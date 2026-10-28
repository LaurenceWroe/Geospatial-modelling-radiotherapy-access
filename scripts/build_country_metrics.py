"""
Build the per-country choropleth lookup table: national demand-weighted access
metrics for every GLOBOCAN country, at each distance and driving-time threshold.

Metrics (all fractions of national RT demand D, 0..1):
  A_C   = min(1, machines * 450 / D)            supply adequacy (threshold-free)
  A_G   = Σ (geographic_access · demand) / D     demand-weighted geographic reach
  A_RM  = Σ treated / D                          realised treatment (supply × access)

Thresholds:
  distance : 50, 100, 200 km        (haversine — computed for every country)
  time     : 30, 60, 120 min        (res-5 matrix aggregated from the res-6
                                      store, data/tt_store.py;
                                      a fitted distance proxy for the countries
                                      TravelTime does not cover)

The 2 h matrix already spans every requested time threshold, so 30/60/120 min
are obtained by thresholding it — no new API calls. Countries with no LINACs get
genuine zeros (A_C = A_G = A_RM = 0), not "no data".

Output: data/country_metrics.parquet  (tracked; a few KB, ~174 rows)
Also prints a summary and a self-check against figures/table3_metrics.csv.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "travel_time_cache"
_OUT = _ROOT / "data" / "country_metrics.parquet"

from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
from data.cancer import apportion_cancer_to_h3, get_cancer_types, DERIVED_CANCER_TYPES
from data.regions import get_region
from analysis.accessibility import compute_accessibility
from data.tt_store import store_matrix

_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
_CAP = 450.0
RES = 5
DIST_KM = [50, 100, 200]
TT_MIN = [30, 60, 120]
# Empirically fitted km whose distance-A_G matches each travel-time A_G
# (from the 150 measured countries; see scripts/calibrate_fit.py).
KM_MAP = {30: 12.6, 60: 28.4, 90: 41.6, 120: 64.4}


def _demand(cgdf):
    cols = [c for c in cgdf.columns if c.endswith("_optimal_rt")
            and c[:-len("_optimal_rt")].strip().lower() not in _AGG]
    return cgdf[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64) if cols else None


def _cache_matrix(iso3, locs, hex_ids):
    return store_matrix(iso3, hex_ids, locs)  # res-6 store (data/tt_store.py)


def _ag_arm(gdf, locs, dem, thr, tt=None):
    """Return (A_G, A_RM) as fractions of demand at threshold *thr*."""
    g, s = compute_accessibility(
        gdf, locs, model="step", max_distance_km=thr,
        capacity_per_machine_per_year=_CAP, demand=dem,
        h3_resolution=RES, travel_time_matrix=tt,
    )
    D = s["total_rt_demand"]
    if D <= 0:
        return np.nan, np.nan
    a_g = float((g["access_probability"] * g["rt_demand"]).sum()) / D
    a_rm = s["total_rt_treated"] / D
    return a_g, a_rm


def main():
    all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES
    members = get_region("World").member_alpha2
    rows = []

    for i, a2 in enumerate(members):
        obj = pycountry.countries.get(alpha_2=a2)
        if obj is None:
            continue
        name, iso3 = obj.name, obj.alpha_3
        try:
            gdf = load_population_at_resolution(name, target_resolution=RES)
            cgdf = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False, rtu_set="barton_2014")
        except Exception:
            continue
        dem = _demand(cgdf)
        if dem is None or float(dem.sum()) <= 0:
            continue
        D = float(dem.sum())

        try:
            locs, _ = load_linacs_from_dirac_db(name)
        except Exception:
            locs = []

        row = {"iso3": iso3, "name": name, "demand": D,
               "n_machines": float(sum(w for *_, w in locs)) if locs else 0.0}

        if not locs:
            # genuine zero access everywhere
            row["A_C"] = 0.0
            for d in DIST_KM:
                row[f"A_G_{d}km"] = 0.0; row[f"A_RM_{d}km"] = 0.0
            for t in TT_MIN:
                row[f"A_G_{t}min"] = 0.0; row[f"A_RM_{t}min"] = 0.0
            row["time_source"] = "none"
            rows.append(row)
            print(f"[{i+1:>3}/{len(members)}] {iso3} {name[:24]:24} no facilities -> zeros", flush=True)
            continue

        row["A_C"] = min(1.0, row["n_machines"] * _CAP / D)
        for d in DIST_KM:
            ag, arm = _ag_arm(gdf, locs, dem, d)
            row[f"A_G_{d}km"] = ag; row[f"A_RM_{d}km"] = arm

        tt = _cache_matrix(iso3, locs, list(gdf["h3"]))
        row["time_source"] = "traveltime" if tt is not None else "proxy"
        for t in TT_MIN:
            if tt is not None:
                ag, arm = _ag_arm(gdf, locs, dem, t, tt=tt)
            else:
                ag, arm = _ag_arm(gdf, locs, dem, KM_MAP[t])  # distance proxy
            row[f"A_G_{t}min"] = ag; row[f"A_RM_{t}min"] = arm

        rows.append(row)
        print(f"[{i+1:>3}/{len(members)}] {iso3} {name[:24]:24} "
              f"A_C={row['A_C']:.2f}  A_G120={row['A_G_120min']:.2f}  "
              f"A_RM120={row['A_RM_120min']:.2f}  ({row['time_source']})", flush=True)

    df = pd.DataFrame(rows).sort_values("iso3").reset_index(drop=True)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT, index=False)
    print(f"\nWrote {_OUT}  ({len(df)} countries)")
    print("time_source:", df["time_source"].value_counts().to_dict())

    # ---- self-check against Table 3 (2 h driving == 120 min) ----
    t3 = _ROOT / "figures" / "table3_metrics.csv"
    if t3.exists():
        chk = pd.read_csv(t3)
        chk["A_C_f"] = chk["A_C"] / chk["D"]
        chk["A_G_f"] = chk["A_G"] / chk["D"]
        chk["A_RM_f"] = chk["A_RM"] / chk["D"]
        chk = chk[["country", "A_C_f", "A_G_f", "A_RM_f"]]  # avoid clashing with df's A_C
        m = df.merge(chk, left_on="name", right_on="country", how="inner")
        print("\n=== self-check vs Table 3 (fractions; should match within rounding) ===")
        print(f"{'name':16} {'A_C Δ':>7} {'A_G120 Δ':>9} {'A_RM120 Δ':>10}")
        for _, r in m.iterrows():
            print(f"{r['name'][:16]:16} {r['A_C']-r['A_C_f']:>+7.3f} "
                  f"{r['A_G_120min']-r['A_G_f']:>+9.3f} {r['A_RM_120min']-r['A_RM_f']:>+10.3f}")


if __name__ == "__main__":
    main()
