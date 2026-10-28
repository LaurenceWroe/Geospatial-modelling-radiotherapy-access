"""
Table 3 — per-country radiotherapy access metrics, absolute counts (thousands),
at H3 resolution 5 with a 2 h step-function driving-time threshold.

Reproduces the per-country table (Australia, India, Nigeria, Oman, UK, US):

  D    = Σ demand                                  (total RT demand)
  A_C  = min(Σ demand, machines · 450)             (capacity-limited supply)
  A_G  = Σ (geographic_access · demand)            (demand-weighted access, 2 h drive)
  A_RM = Σ treated                                 (capacity-limited, 2 h drive)

All in patients/yr; printed in thousands (k). A_G and A_RM use the real cached
2 h driving-time matrix for each country. Mirrors the recipe in final_table.py /
demand_table.py but broken out per country and reported as absolute counts.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "travel_time_cache"

from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
from data.cancer import apportion_cancer_to_h3, get_cancer_types, DERIVED_CANCER_TYPES
from analysis.accessibility import compute_accessibility

_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
_CAP = 450.0
RES = 5
HRS = 2

COUNTRIES = ["Australia", "India", "Nigeria", "Oman", "United Kingdom", "United States"]


def _demand(cgdf):
    cols = [c for c in cgdf.columns if c.endswith("_optimal_rt")
            and c[:-len("_optimal_rt")].strip().lower() not in _AGG]
    return cgdf[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64) if cols else None


def _cache_matrix(iso3, locs):
    ll = [(lat, lon) for lat, lon, _ in locs]
    lh = hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]
    p = _CACHE / f"{iso3}_res{RES}_{HRS}h_{lh}_driving.npz"
    if not p.exists():
        return None
    m = np.load(p)["matrix"]
    return m if np.isfinite(m).any() else None


def main():
    all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES
    rows = []
    print(f"{'Country':16} {'D (k)':>8} {'A_C (k)':>8} {'A_G (k)':>8} {'A_RM (k)':>9}")
    print("-" * 54)
    for name in COUNTRIES:
        obj = pycountry.countries.lookup(name)
        iso3 = obj.alpha_3
        gdf = load_population_at_resolution(obj.name, target_resolution=RES)
        cgdf = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False, rtu_set="barton_2014")
        dem = _demand(cgdf)
        locs, _ = load_linacs_from_dirac_db(obj.name)
        tt = _cache_matrix(iso3, locs)
        src = "TravelTime" if tt is not None else "NO CACHE (proxy!)"

        g, s = compute_accessibility(
            gdf, locs, model="step", max_distance_km=(HRS * 60),
            capacity_per_machine_per_year=_CAP, demand=dem,
            h3_resolution=RES, travel_time_matrix=tt,
        )
        D = s["total_rt_demand"]
        A_C = min(D, s["total_machines"] * _CAP)
        A_G = float((g["access_probability"] * g["rt_demand"]).sum())
        A_RM = s["total_rt_treated"]
        rows.append((name, D, A_C, A_G, A_RM))
        print(f"{name:16} {D/1e3:>8.1f} {A_C/1e3:>8.1f} {A_G/1e3:>8.1f} {A_RM/1e3:>9.1f}   {src}")

    # LaTeX body for pasting back into the paper
    print("\n% --- LaTeX table body ---")
    for name, D, A_C, A_G, A_RM in rows:
        print(f"{name} & {D/1e3:.1f} & {A_C/1e3:.1f} & {A_G/1e3:.1f} & {A_RM/1e3:.1f} \\\\")


if __name__ == "__main__":
    main()
