"""
Global results for the revised paper: per-country D, A_C, A_G and A for both
allocation rules, plus per-hexagon aggregates for the global hex maps.

Configuration (paper defaults):
  * H3 resolution 5, 120-min driving time aggregated from the resolution-6
    TravelTime store (data/tt_store.py, travel_time_store/), DIRAC October 2026, 2012 CCORE RTUs,
    C_linac = 450 patients/yr;
  * decay: Gaussian with x_max = 120 min (main) and a 120-min step (sensitivity,
    and the quantity reported by Wawrzuta et al. 2025);
  * allocation: nearest-first and capacity-capped 3SFCA ("proportional").

Countries with facilities but no TravelTime matrix (not served by the API)
fall back to great-circle distance with the calibrated 120-min equivalent
(compute.minutes_to_proxy_km, fitted by scripts/calibrate_fit.py) and are flagged time_source = "proxy".
Countries with no facilities get genuine zeros.

Outputs
  figures/global_results.csv           one row per country
  data/country_results.parquet         the same table, read by the app's World Choropleth
  figures/global_hex_res3.parquet      per res-3 hexagon: D, A_G and A sums for
                                       each configuration (aggregated from res 5)

    python scripts/global_results.py [--countries GBR IND ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "travel_time_cache"
_FIG = _ROOT / "figures"

from analysis.accessibility import compute_accessibility  # noqa: E402
from compute import minutes_to_proxy_km  # noqa: E402
from data.cancer import DERIVED_CANCER_TYPES, apportion_cancer_to_h3, get_cancer_types  # noqa: E402
from data.linacs import load_linacs_from_dirac_db  # noqa: E402
from data.population import load_population_at_resolution  # noqa: E402
from data.regions import get_region  # noqa: E402
from data.tt_store import store_matrix  # noqa: E402

RES, MAP_RES, CAP, T_MAX = 5, 3, 450.0, 120.0
# calibrated great-circle equivalents of each driving-time budget (scripts/calibrate_fit.py)
KM_MAP = {t: minutes_to_proxy_km(t) for t in (30, 60, 90, 120)}  # compute._PROXY_KM
PROXY_KM = KM_MAP[120]
_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
CONFIGS = [  # (label, decay model, allocation rule)
    ("gauss_nf", "gaussian", "nearest_first"),
    ("gauss_fca", "gaussian", "proportional"),
    ("step_nf", "step", "nearest_first"),
    ("step_fca", "step", "proportional"),
]


def _demand(cgdf):
    cols = [c for c in cgdf.columns if c.endswith("_optimal_rt")
            and c[:-len("_optimal_rt")].strip().lower() not in _AGG]
    return cgdf[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64) if cols else None


def _tt(iso3, locs, hex_ids):
    return store_matrix(iso3, hex_ids, locs)  # res-6 store (data/tt_store.py)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*", help="ISO3 codes (default: World)")
    args = ap.parse_args()
    wb = pd.read_csv(_ROOT / "data" / "wb_income_groups.csv").set_index("iso3")
    cancers = get_cancer_types() + DERIVED_CANCER_TYPES

    if args.countries:
        objs = [pycountry.countries.get(alpha_3=c) for c in args.countries]
    else:
        objs = [pycountry.countries.get(alpha_2=a2) for a2 in get_region("World").member_alpha2]
    rows, hexes = [], []

    for i, obj in enumerate(o for o in objs if o is not None):
        name, iso3 = obj.name, obj.alpha_3
        try:
            gdf = load_population_at_resolution(name, target_resolution=RES)
            cgdf = apportion_cancer_to_h3(gdf, iso3, cancers, use_actual_rt=False, rtu_set="barton_2014")
        except Exception as exc:  # noqa: BLE001
            print(f"{iso3}: skipped ({str(exc)[:80]})", flush=True)
            continue
        dem = _demand(cgdf)
        if dem is None or dem.sum() <= 0:
            continue
        try:
            locs, _ = load_linacs_from_dirac_db(name)
        except Exception:  # noqa: BLE001
            locs = []
        D = float(dem.sum())
        n_mac = float(sum(w for *_, w in locs)) if locs else 0.0
        row = {"iso3": iso3, "name": name, "population": float(gdf["population"].sum()), "D": D,
               "n_facilities": len(locs), "n_machines": n_mac, "A_C": min(D, n_mac * CAP),
               "income_group": wb["income_group"].get(iso3), "wb_region": wb["wb_region"].get(iso3)}
        parents = np.array([h3.cell_to_parent(c, MAP_RES) for c in gdf["h3"]])
        hx = pd.DataFrame({"h3": parents, "iso3": iso3, "D": dem, "pop": gdf["population"].to_numpy(float)})

        if not locs:
            row["time_source"] = "none"
            for lab, *_ in CONFIGS:
                row[f"A_G_{lab.split('_')[0]}"] = 0.0
                row[f"A_{lab}"] = 0.0
                hx[f"A_{lab}"] = 0.0
            hx["A_G_gauss"] = hx["A_G_step"] = 0.0
            for t in KM_MAP:
                hx[f"pop_{t}"] = 0.0
        else:
            tt = _tt(iso3, locs, list(gdf["h3"]))
            row["time_source"] = "traveltime" if tt is not None else "proxy"
            thr = T_MAX if tt is not None else PROXY_KM
            for lab, model, rule in CONFIGS:
                g, s = compute_accessibility(
                    gdf, locs, model=model, max_distance_km=thr, capacity_per_machine_per_year=CAP,
                    demand=dem, h3_resolution=RES, travel_time_matrix=tt, allocation=rule,
                )
                a_g = (g["access_probability"] * g["rt_demand"]).to_numpy(float)
                row[f"A_G_{model[:5] if model == 'gaussian' else model}"] = float(a_g.sum())
                row[f"A_{lab}"] = float(s["total_rt_treated"])
                hx[f"A_{lab}"] = g["rt_treated"].to_numpy(float)
                hx[f"A_G_{'gauss' if model == 'gaussian' else 'step'}"] = a_g
            # population within each budget of its nearest facility (Wawrzuta et al. metric)
            if tt is not None:
                near = np.min(tt, axis=1)
                for t in KM_MAP:
                    hx[f"pop_{t}"] = np.where(near <= t, hx["pop"], 0.0)
            else:
                near = g["nearest_linac_km"].to_numpy(float)
                for t, km in KM_MAP.items():
                    hx[f"pop_{t}"] = np.where(near <= km, hx["pop"], 0.0)
        for t in KM_MAP:
            row[f"pop_{t}"] = float(hx[f"pop_{t}"].sum())
        rows.append(row)
        hexes.append(hx.groupby(["h3", "iso3"], as_index=False).sum(numeric_only=True))
        print(f"[{i + 1:>3}] {iso3} {name[:22]:22} D={D:>9.0f}  A_C={row['A_C'] / D:5.0%}  "
              f"A_G={row['A_G_gauss'] / D:5.0%}  A_nf={row['A_gauss_nf'] / D:5.0%}  "
              f"A_fca={row['A_gauss_fca'] / D:5.0%}  ({row['time_source']})", flush=True)

    _FIG.mkdir(exist_ok=True)
    suffix = "" if not args.countries else "_subset"
    pd.DataFrame(rows).to_csv(_FIG / f"global_results{suffix}.csv", index=False)
    if not args.countries:  # the app's World Choropleth reads this copy
        pd.DataFrame(rows).to_parquet(_ROOT / "data" / "country_results.parquet", index=False)
    pd.concat(hexes).to_parquet(_FIG / f"global_hex_res3{suffix}.parquet")
    df = pd.DataFrame(rows)
    tot = df[["D", "A_C", "A_G_gauss", "A_G_step", "A_gauss_nf", "A_gauss_fca", "A_step_nf", "A_step_fca"]].sum()
    print("\nGlobal (% of D):", (100 * tot / tot["D"]).round(1).to_dict())


if __name__ == "__main__":
    main()
