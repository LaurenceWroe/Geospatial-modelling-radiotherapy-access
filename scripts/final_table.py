"""
Final global travel-time table with a distance proxy for TravelTime-unsupported
countries (China, Russia, North Korea, + any other failed fetch).

- Supported countries with a cached TT matrix -> real driving time.
- Failed/unsupported countries -> straight-line distance step at a mapped km
  threshold (user mapping: 30/60/90/120 min = 15/30/45/60 km, i.e. 30 km/h).
- Countries with no facilities -> genuine zero access.

Also prints a CALIBRATION: distance-based A_G at the mapped km over the
assessable (real-TT) countries, so the proxy ratio can be checked against the
measured travel-time numbers.
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
HRS = 2
TT_MIN = [30, 60, 90, 120]
# Empirically fitted from the 150 measured countries (km whose A_G matches each
# travel-time A_G). Not a constant speed: ~21 km/h short trips, ~34 km/h long.
KM_MAP = {30: 12.6, 60: 28.4, 90: 41.6, 120: 64.4}


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


def _accum(acc, key, s):
    a = acc[key]
    a["pop"] += s["total_population"]; a["acc"] += s["pop_with_access"]
    a["dem"] += s["total_rt_demand"]; a["tre"] += s["total_rt_treated"]
    a["mac"] += s["total_machines"]


def main():
    members = get_region("World").member_alpha2
    # final = TT for supported + distance-proxy for failed; calib = distance over assessable only
    final = {t: dict(pop=0.0, acc=0.0, dem=0.0, tre=0.0, mac=0.0) for t in TT_MIN}
    calib = {t: dict(pop=0.0, acc=0.0, dem=0.0, tre=0.0, mac=0.0) for t in TT_MIN}
    tt_assess = {t: dict(pop=0.0, acc=0.0, dem=0.0, tre=0.0, mac=0.0) for t in TT_MIN}
    proxied = []
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
            for t in TT_MIN:
                final[t]["pop"] += pop; final[t]["dem"] += float(dem.sum())
            continue

        tt = _cache_matrix(iso3, locs)
        if tt is None:
            # distance proxy at mapped km
            proxied.append((name, pop))
            for t in TT_MIN:
                _, s = compute_accessibility(gdf, locs, model="step", max_distance_km=KM_MAP[t],
                                             capacity_per_machine_per_year=_CAP, demand=dem,
                                             h3_resolution=RES)
                _accum(final, t, s)
        else:
            for t in TT_MIN:
                _, s = compute_accessibility(gdf, locs, model="step", max_distance_km=t,
                                             capacity_per_machine_per_year=_CAP, demand=dem,
                                             h3_resolution=RES, travel_time_matrix=tt)
                _accum(final, t, s)
                _accum(tt_assess, t, s)
                # calibration: same country, distance at mapped km
                _, sd = compute_accessibility(gdf, locs, model="step", max_distance_km=KM_MAP[t],
                                              capacity_per_machine_per_year=_CAP, demand=dem,
                                              h3_resolution=RES)
                _accum(calib, t, sd)

    def _row(d):
        ag = d["acc"] / d["pop"] if d["pop"] else float("nan")
        ac = min(1.0, d["mac"] * _CAP / d["dem"]) if d["dem"] else float("nan")
        arm = d["tre"] / d["dem"] if d["dem"] else float("nan")
        return ac, ag, arm

    print("=== CALIBRATION (assessable countries): real travel time vs distance proxy ===")
    print(f"{'min':>4} {'km':>4} | {'A_G(time)':>9} {'A_G(dist)':>9} | {'A_RM(time)':>10} {'A_RM(dist)':>10}")
    for t in TT_MIN:
        _, ag_t, arm_t = _row(tt_assess[t])
        _, ag_d, arm_d = _row(calib[t])
        print(f"{t:>4} {KM_MAP[t]:>4.0f} | {ag_t:>8.1%} {ag_d:>8.1%} | {arm_t:>9.1%} {arm_d:>9.1%}")

    print("\n=== FINAL GLOBAL TABLE (real travel time + distance proxy for unsupported) ===")
    for t in TT_MIN:
        ac, ag, arm = _row(final[t])
        print(f"  {t:>3} min (proxy {KM_MAP[t]:.0f} km) ->  A_C={ac:.1%}  A_G={ag:.1%}  A_RM={arm:.1%}")

    print(f"\nDistance-proxied countries ({len(proxied)}):")
    for name, pop in sorted(proxied, key=lambda x: -x[1]):
        print(f"   {name:35} pop {pop:,.0f}")


if __name__ == "__main__":
    main()
