"""
Table 3 of the RadMaps paper recomputed under alternative demand and capacity
assumptions, for the response to reviewers.

Rows per country:
  published      Delaney 2005 RTU set, 450 patients/linac/yr      (= paper Table 3)
  ccore_2012     CCORE 2012 RTU set (Barton 2014 / Wong 2016), 450 patients/linac/yr
  ccore_2012+fx  CCORE 2012 RTU set, capacity from 8,730 fractions/linac/yr converted
                 with the country's own case mix (Wong 2016 fractions per course)

All at H3 resolution 5 with the cached 2 h driving-time matrices (step model),
exactly as in scripts/table3_percountry.py. Memory: one country at a time;
run it under scripts/run_capped.py on a shared machine.

    python scripts/run_capped.py -- python scripts/table3_rtu_variants.py [--out figures/table3_rtu_variants.csv]
"""
from __future__ import annotations

import argparse
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
from data.linacs import load_linacs_from_dirac_db
from data.cancer import (
    apportion_cancer_to_h3, get_cancer_types, DERIVED_CANCER_TYPES,
    effective_capacity_per_linac, DEFAULT_FRACTIONS_PER_LINAC_PER_YEAR,
)
from analysis.accessibility import compute_accessibility

_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
RES, HRS = 5, 2
COUNTRIES = ["Australia", "India", "Nigeria", "Oman", "United Kingdom", "United States"]
VARIANTS = [
    ("published", "delaney_2005", None),
    ("delaney_corr", "delaney_2005_corrected", None),
    ("ccore_2012", "barton_2014", None),
    ("ccore_2012+fx", "barton_2014", DEFAULT_FRACTIONS_PER_LINAC_PER_YEAR),
]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "figures" / "table3_rtu_variants.csv"))
    args = ap.parse_args()

    all_c = get_cancer_types() + DERIVED_CANCER_TYPES
    rows = []
    print(f"{'Country':15} {'variant':14} {'pts/linac':>9} {'D(k)':>7} {'A_C(k)':>7} {'A_G(k)':>7} {'A_RM(k)':>8} {'A_RM%':>6}")
    for name in COUNTRIES:
        obj = pycountry.countries.lookup(name)
        iso3 = obj.alpha_3
        gdf = load_population_at_resolution(obj.name, target_resolution=RES)
        locs, _ = load_linacs_from_dirac_db(obj.name)
        tt = _tt(iso3, locs)
        for label, rtu, fx in VARIANTS:
            cap = 450.0 if fx is None else effective_capacity_per_linac(iso3, fx, rtu, 1.0)["patients_per_linac"]
            cgdf = apportion_cancer_to_h3(gdf, iso3, all_c, use_actual_rt=False, rtu_set=rtu)
            dem = _demand(cgdf)
            g, s = compute_accessibility(
                gdf, locs, model="step", max_distance_km=HRS * 60,
                capacity_per_machine_per_year=cap, demand=dem,
                h3_resolution=RES, travel_time_matrix=tt,
            )
            D = s["total_rt_demand"]
            A_C = min(D, s["total_machines"] * cap)
            A_G = float((g["access_probability"] * g["rt_demand"]).sum())
            A_RM = s["total_rt_treated"]
            rows.append({"country": name, "variant": label, "rtu_set": rtu, "patients_per_linac": round(cap, 1),
                         "D_k": round(D / 1e3, 1), "A_C_k": round(A_C / 1e3, 1), "A_G_k": round(A_G / 1e3, 1),
                         "A_RM_k": round(A_RM / 1e3, 1), "A_RM_pct": round(100 * A_RM / D, 1) if D > 0 else 0.0,
                         "travel_time": tt is not None})
            print(f"{name:15} {label:14} {cap:9.0f} {D/1e3:7.1f} {A_C/1e3:7.1f} {A_G/1e3:7.1f} {A_RM/1e3:8.1f} {100*A_RM/D:6.1f}")
        del gdf, cgdf, g  # keep one country resident at a time

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
