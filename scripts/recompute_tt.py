"""
Recompute the global travel-time table from CACHED matrices (no API calls).

Correctly separates two cases that the first run conflated:
  - country has NO facilities            -> genuine zero domestic access (INCLUDE)
  - country HAS facilities but TT fetch failed (unsupported by TravelTime, e.g.
    China/Russia/North Korea) -> data gap, cannot assess (EXCLUDE + report)

Prints the corrected A_C / A_G / A_RM over the assessable population and lists
the excluded countries with their population share.
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
from data.regions import get_region
from analysis.accessibility import compute_accessibility

_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
_CAP = 450.0
RES = 5
HRS = 2  # 120-min budget


def _demand(cgdf):
    cols = [c for c in cgdf.columns if c.endswith("_optimal_rt")
            and c[:-len("_optimal_rt")].strip().lower() not in _AGG]
    return cgdf[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64) if cols else None


def _cache_matrix(iso3, hex_ids, locs):
    ll = [(lat, lon) for lat, lon, _ in locs]
    lh = hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]
    p = _CACHE / f"{iso3}_res{RES}_{HRS}h_{lh}_driving.npz"
    if not p.exists():
        return None
    m = np.load(p)["matrix"]
    return m if np.isfinite(m).any() else None  # all-inf => failed


def main():
    thresholds = [30, 60, 90, 120]
    members = get_region("World").member_alpha2
    tot = {t: {"pop": 0.0, "acc": 0.0, "dem": 0.0, "tre": 0.0, "mac": 0.0} for t in thresholds}
    excluded, no_fac = [], 0
    all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES

    for a2 in members:
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
        if dem is None:
            continue
        pop = float(gdf["population"].sum())

        try:
            locs, _ = load_linacs_from_dirac_db(name)
        except Exception:
            locs = []

        if not locs:
            no_fac += 1
            for t in thresholds:  # genuine zero access
                tot[t]["pop"] += pop
                tot[t]["dem"] += float(dem.sum())
            continue

        tt = _cache_matrix(iso3, list(gdf["h3"]), locs)
        if tt is None:  # has facilities but TT fetch failed -> data gap
            excluded.append((name, pop))
            continue

        for t in thresholds:
            _, s = compute_accessibility(gdf, locs, model="step", max_distance_km=t,
                                         capacity_per_machine_per_year=_CAP, demand=dem,
                                         h3_resolution=RES, travel_time_matrix=tt)
            tot[t]["pop"] += s["total_population"]
            tot[t]["acc"] += s["pop_with_access"]
            tot[t]["dem"] += s["total_rt_demand"]
            tot[t]["tre"] += s["total_rt_treated"]
            tot[t]["mac"] += s["total_machines"]

    print("=== Corrected travel-time table (assessable population only) ===")
    for t in thresholds:
        d = tot[t]
        a_g = d["acc"] / d["pop"] if d["pop"] else float("nan")
        a_c = min(1.0, d["mac"] * _CAP / d["dem"]) if d["dem"] else float("nan")
        a_rm = d["tre"] / d["dem"] if d["dem"] else float("nan")
        print(f"  {t:>3} min ->  A_C={a_c:.1%}  A_G={a_g:.1%}  A_RM={a_rm:.1%}")

    expop = sum(p for _, p in excluded)
    inpop = tot[120]["pop"]
    print(f"\nCountries with genuinely no domestic facility (counted as 0): {no_fac}")
    print(f"EXCLUDED (facilities exist but TravelTime unsupported): {len(excluded)}")
    for name, pop in sorted(excluded, key=lambda x: -x[1]):
        print(f"   {name:35} pop {pop:,.0f}")
    print(f"\nExcluded population: {expop:,.0f}  "
          f"({100*expop/(expop+inpop):.1f}% of world with-facility population unassessable)")


if __name__ == "__main__":
    main()
