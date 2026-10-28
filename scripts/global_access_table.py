"""
Standalone global radiotherapy-access table (NOT wired into the app).

Computes, per-country (domestic only) and aggregated to the world:
  A_G   geographic access probability   = Σ pop_with_access / Σ population
  A_C   capacity-only ratio             = min(1, Σ capacity / Σ demand)
  A_RM  modelled access ratio           = Σ rt_treated / Σ rt_demand

for a set of step-function thresholds, at a chosen H3 resolution, using either
straight-line distance (km) or travel time (minutes). Definitions mirror the
app's per-country region aggregation (data/../app.py) and analysis.accessibility.

Usage:
  python scripts/global_access_table.py --res 3 --distance 100 200
  python scripts/global_access_table.py --res 5 --distance 100 200
  python scripts/global_access_table.py --res 5 --tt 30 60 90 120   (needs API + cache)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
from data.cancer import apportion_cancer_to_h3, get_cancer_types
from data.cancer import DERIVED_CANCER_TYPES
from data.regions import get_region
from analysis.accessibility import compute_accessibility

_AGG_KEYS = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
_CAP_PER_MACHINE = 450.0


def _optimal_demand(cancer_gdf) -> np.ndarray | None:
    rt_cols = [
        c for c in cancer_gdf.columns
        if c.endswith("_optimal_rt")
        and c[:-len("_optimal_rt")].strip().lower() not in _AGG_KEYS
    ]
    if not rt_cols:
        return None
    return cancer_gdf[rt_cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)


def _country_inputs(alpha2: str, res: int, tt_matrix_loader=None):
    """Load one country once: returns (gdf, demand, locs, tt) or None."""
    obj = pycountry.countries.get(alpha_2=alpha2)
    if obj is None:
        return None
    name, iso3 = obj.name, obj.alpha_3
    try:
        gdf = load_population_at_resolution(name, target_resolution=res)
    except Exception:
        return None
    all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES
    try:
        cancer_gdf = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False, rtu_set="barton_2014")
    except Exception:
        return None
    demand = _optimal_demand(cancer_gdf)
    if demand is None:
        return None
    tt = tt_matrix_loader(name, iso3, gdf) if tt_matrix_loader is not None else None
    try:
        locs, _ = load_linacs_from_dirac_db(name)
    except Exception:
        locs = []
    return gdf, demand, locs, tt


def _country_totals(inp, res, model, threshold):
    gdf, demand, locs, tt = inp
    if locs:
        _, stats = compute_accessibility(
            gdf, locs, model=model, max_distance_km=threshold,
            capacity_per_machine_per_year=_CAP_PER_MACHINE, demand=demand,
            h3_resolution=res, travel_time_matrix=tt,
        )
        return {"pop": stats["total_population"], "pop_access": stats["pop_with_access"],
                "demand": stats["total_rt_demand"], "treated": stats["total_rt_treated"],
                "machines": stats["total_machines"]}
    return {"pop": float(gdf["population"].sum()), "pop_access": 0.0,
            "demand": float(demand.sum()), "treated": 0.0, "machines": 0.0}


def run_multi(res: int, thresholds: list[float], unit: str, tt_matrix_loader=None,
              verbose: bool = False):
    """Load each country once; evaluate every threshold. Returns one row per threshold."""
    members = get_region("World").member_alpha2
    tots = {thr: {"pop": 0.0, "pop_access": 0.0, "demand": 0.0, "treated": 0.0, "machines": 0.0}
            for thr in thresholds}
    skipped = []
    for i, a2 in enumerate(members):
        inp = _country_inputs(a2, res, tt_matrix_loader)
        if inp is None:
            skipped.append(a2)
            continue
        for thr in thresholds:
            r = _country_totals(inp, res, "step", thr)
            for k in tots[thr]:
                tots[thr][k] += r[k]
        if verbose:
            print(f"  [{i+1}/{len(members)}] {a2} done", flush=True)

    rows = []
    for thr in thresholds:
        t = tots[thr]
        a_g = t["pop_access"] / t["pop"] if t["pop"] > 0 else float("nan")
        cap = t["machines"] * _CAP_PER_MACHINE
        a_c = min(1.0, cap / t["demand"]) if t["demand"] > 0 else float("nan")
        a_rm = t["treated"] / t["demand"] if t["demand"] > 0 else float("nan")
        rows.append({"res": res, "threshold": f"{thr:g} {unit}",
                     "A_C": a_c, "A_G": a_g, "A_RM": a_rm, "n_skipped": len(skipped)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--distance", type=float, nargs="*", default=[])
    ap.add_argument("--tt", type=float, nargs="*", default=[])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rows = []
    if args.distance:
        t = time.time()
        drows = run_multi(args.res, args.distance, "km", verbose=args.verbose)
        for row in drows:
            rows.append(row)
            print(f"res {args.res}  {row['threshold']:>7}  ->  "
                  f"A_C={row['A_C']:.1%}  A_G={row['A_G']:.1%}  A_RM={row['A_RM']:.1%}  "
                  f"(skipped {row['n_skipped']})", flush=True)
        print(f"  [distance sweep {round(time.time()-t)}s]", flush=True)

    if args.tt:
        from tt_loader import make_tt_loader
        loader = make_tt_loader(args.res)
        t = time.time()
        trows = run_multi(args.res, args.tt, "min", tt_matrix_loader=loader, verbose=args.verbose)
        for row in trows:
            rows.append(row)
            print(f"res {args.res}  {row['threshold']:>7}  ->  "
                  f"A_C={row['A_C']:.1%}  A_G={row['A_G']:.1%}  A_RM={row['A_RM']:.1%}", flush=True)
        print(f"  [travel-time sweep {round(time.time()-t)}s]", flush=True)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / f"global_access_res{args.res}.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
