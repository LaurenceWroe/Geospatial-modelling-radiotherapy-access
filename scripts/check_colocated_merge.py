"""
Quantify the effect of merging co-located DIRAC records on the national
access metrics (Table 3 configuration: res 5, 2 h driving step, 450 pts/linac).

Merging changes the facility list, so the cached travel-time matrix cannot be
reused directly; instead the merged matrix is built from the cached one by
collapsing columns whose facilities share coordinates (their travel times are
identical by construction) and summing the machine counts.

    scripts/remote_run.sh python scripts/check_colocated_merge.py
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

from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db, merge_colocated_facilities
from data.cancer import apportion_cancer_to_h3, get_cancer_types, DERIVED_CANCER_TYPES
from analysis.accessibility import compute_accessibility

_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
RES, HRS, CAP = 5, 2, 450.0
COUNTRIES = ["Australia", "India", "Nigeria", "Oman", "United Kingdom", "United States"]


def _demand(cgdf):
    cols = [c for c in cgdf.columns if c.endswith("_optimal_rt")
            and c[:-len("_optimal_rt")].strip().lower() not in _AGG]
    return cgdf[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)


def _tt(iso3, locs):
    ll = [(lat, lon) for lat, lon, _ in locs]
    lh = hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]
    p = _CACHE / f"{iso3}_res{RES}_{HRS}h_{lh}_driving.npz"
    return np.load(p)["matrix"] if p.exists() else None


def main() -> None:
    all_c = get_cancer_types() + DERIVED_CANCER_TYPES
    rows = []
    print(f"{'Country':15} {'fac':>5} {'merged':>6} {'A_G%':>7} {'A_G% m':>7} {'A_RM%':>7} {'A_RM% m':>8} {'dA_RM pts':>10}")
    for name in COUNTRIES:
        obj = pycountry.countries.lookup(name)
        iso3 = obj.alpha_3
        gdf = load_population_at_resolution(obj.name, target_resolution=RES)
        cgdf = apportion_cancer_to_h3(gdf, iso3, all_c, use_actual_rt=False, rtu_set="barton_2014")
        dem = _demand(cgdf)
        locs, fdf = load_linacs_from_dirac_db(obj.name)
        tt = _tt(iso3, locs)
        # merged facility list + merged matrix (collapse identical-coordinate columns)
        mdf = merge_colocated_facilities(fdf)
        key = fdf[["lat", "lon"]].round(5).apply(tuple, axis=1)
        mkey = mdf[["lat", "lon"]].round(5).apply(tuple, axis=1)
        first_col = {k: i for i, k in reversed(list(enumerate(key)))}
        cols = [first_col[k] for k in mkey]
        tt_m = tt[:, cols] if tt is not None else None
        mlocs = [(r.lat, r.lon, r.n_linacs) for r in mdf.itertuples()]
        out = {}
        for tag, L, M in (("orig", locs, tt), ("merged", mlocs, tt_m)):
            g, s = compute_accessibility(gdf, L, model="step", max_distance_km=HRS * 60,
                                         capacity_per_machine_per_year=CAP, demand=dem,
                                         h3_resolution=RES, travel_time_matrix=M)
            D = s["total_rt_demand"]
            out[tag] = (100 * float((g["access_probability"] * g["rt_demand"]).sum()) / D,
                        100 * s["total_rt_treated"] / D, s["total_machines"])
        rows.append({"country": name, "facilities": len(locs), "merged": len(mlocs),
                     "machines": out["orig"][2], "machines_merged": out["merged"][2],
                     "A_G_pct": round(out["orig"][0], 2), "A_G_pct_merged": round(out["merged"][0], 2),
                     "A_RM_pct": round(out["orig"][1], 2), "A_RM_pct_merged": round(out["merged"][1], 2)})
        print(f"{name:15} {len(locs):5d} {len(mlocs):6d} {out['orig'][0]:7.2f} {out['merged'][0]:7.2f} "
              f"{out['orig'][1]:7.2f} {out['merged'][1]:8.2f} {out['merged'][1]-out['orig'][1]:+10.3f}")
        del gdf, cgdf, g
    outp = _ROOT / "figures" / "colocated_merge_check.csv"
    pd.DataFrame(rows).to_csv(outp, index=False)
    print(f"written {outp}")


if __name__ == "__main__":
    main()
