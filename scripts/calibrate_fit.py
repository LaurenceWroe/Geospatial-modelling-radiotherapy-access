"""
Empirically fit the travel-time <-> straight-line-distance mapping, and
sanity-check specific countries against the paper's Table 2.

Uses the travel-time store (data/tt_store.py) + geodesic distance (no API). Population-weighted A_G
only (geography), which is what the mapping/paper compare.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "travel_time_cache"

import h3
from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
from data.regions import get_region

RES = 5
HRS = 2
_R = 6371.0088


def _nearest_km(gdf, locs):
    lats = np.radians([h3.cell_to_latlng(h)[0] for h in gdf["h3"]])
    lons = np.radians([h3.cell_to_latlng(h)[1] for h in gdf["h3"]])
    cl = np.cos(lats)
    nk = np.full(len(gdf), np.inf)
    for lat, lon, _ in locs:
        p, l = math.radians(lat), math.radians(lon)
        a = np.sin((lats - p) / 2) ** 2 + math.cos(p) * cl * np.sin((lons - l) / 2) ** 2
        d = 2 * _R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        np.minimum(nk, d, out=nk)
    return nk


def _nearest_min(gdf, iso3, locs):
    """Driving minutes to the nearest facility from the travel-time store (data/tt_store.py)."""
    from data.tt_store import store_matrix
    m = store_matrix(iso3, list(gdf["h3"]), locs)
    if m is None:
        return None
    nm = m.min(axis=1)
    nm[~np.isfinite(nm)] = np.inf
    return nm


def main():
    members = get_region("World").member_alpha2
    km_grid = np.arange(5, 81, 1.0)
    tt_thr = [30, 60, 90, 120]

    pop_dist = np.zeros(len(km_grid))   # covered pop within each km (assessable)
    pop_time = {t: 0.0 for t in tt_thr}
    tot_pop = 0.0
    n_countries = 0
    table2 = {"United Kingdom": 45, "United States": 60, "Canada": 120}
    t2_out = {}

    for a2 in members:
        obj = pycountry.countries.get(alpha_2=a2)
        if obj is None:
            continue
        name, iso3 = obj.name, obj.alpha_3
        try:
            gdf = load_population_at_resolution(name, target_resolution=RES)
        except Exception:
            continue
        try:
            locs, _ = load_linacs_from_dirac_db(name)
        except Exception:
            continue
        if not locs:
            continue
        nm = _nearest_min(gdf, iso3, locs)
        if nm is None:
            continue  # unsupported/failed -> not part of calibration
        nk = _nearest_km(gdf, locs)
        n_countries += 1
        pop = gdf["population"].to_numpy(np.float64)
        tot = pop.sum()
        tot_pop += tot
        for i, km in enumerate(km_grid):
            pop_dist[i] += pop[nk <= km].sum()
        for t in tt_thr:
            pop_time[t] += pop[nm <= t].sum()
        if name in table2:
            thr = table2[name]
            t2_out[name] = (100 * pop[nm <= thr].sum() / tot, thr)

    ag_dist = pop_dist / tot_pop
    print(f"countries with driving times: {n_countries}")
    print("=== Empirical fit: km whose A_G matches each travel-time A_G (assessable pop) ===")
    print(f"{'min':>4} {'A_G(time)':>9} {'fitted km':>9} {'A_G(dist@km)':>12}")
    for t in tt_thr:
        agt = pop_time[t] / tot_pop
        # invert ag_dist(km) to find km where dist coverage == time coverage
        km_fit = float(np.interp(agt, ag_dist, km_grid))
        agd = float(np.interp(km_fit, km_grid, ag_dist))
        print(f"{t:>4} {agt:>8.1%} {km_fit:>8.1f} {agd:>11.1%}")

    print("\n=== Table 2 sanity check (our TravelTime vs paper's OSM routing) ===")
    paper = {"United Kingdom": (45, 89), "United States": (60, 87), "Canada": (120, 90)}
    for name, (thr, ppct) in paper.items():
        if name in t2_out:
            ours, _ = t2_out[name]
            print(f"  {name:16} {thr:>3} min:  ours {ours:4.0f}%   paper {ppct}%   (diff {ours-ppct:+.0f})")
        else:
            print(f"  {name:16}: no cached TT")


if __name__ == "__main__":
    main()
