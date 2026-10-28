"""
Global A_C / A_G / A_RM table, ALL demand-weighted, for distance and travel time.

  A_C  = min(1, Σ capacity / Σ demand)                  (supply adequacy)
  A_G  = Σ (geographic_access · demand) / Σ demand       (demand-weighted access)
  A_RM = Σ treated / Σ demand                            (demand-weighted, capacity-limited)

Distance thresholds work for every country. Travel-time thresholds use real
cached TravelTime where available and a fitted straight-line-distance proxy for
the countries TravelTime does not cover (China, Russia, Koreas, Macao, Uzbekistan).
Now A_RM <= A_G always (capacity can only reduce demand-weighted access).
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
from data.linacs import load_linacs_from_dirac_db
from data.cancer import apportion_cancer_to_h3, get_cancer_types, DERIVED_CANCER_TYPES
from data.regions import get_region
from analysis.accessibility import compute_accessibility

_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
_CAP = 450.0
RES = 5
DIST_KM = [100.0, 200.0]
TT_MIN = [30, 60, 90, 120]
KM_MAP = {30: 12.6, 60: 28.4, 90: 41.6, 120: 64.4}  # empirically fitted


def _demand(cg):
    cols = [c for c in cg.columns if c.endswith("_optimal_rt")
            and c[:-len("_optimal_rt")].strip().lower() not in _AGG]
    return cg[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64) if cols else None


def _cache(iso3, locs):
    ll = [(a, b) for a, b, _ in locs]
    h = hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]
    p = _CACHE / f"{iso3}_res{RES}_2h_{h}_driving.npz"
    if not p.exists():
        return None
    m = np.load(p)["matrix"]
    return m if np.isfinite(m).any() else None


def _accum(acc, gdf_out, s):
    # demand-weighted geographic access = Σ access_prob * demand
    acc["dem"] += s["total_rt_demand"]
    acc["gacc"] += float((gdf_out["access_probability"] * gdf_out["rt_demand"]).sum())
    acc["tre"] += s["total_rt_treated"]
    acc["mac"] += s["total_machines"]


def main():
    keys = [("dist", d) for d in DIST_KM] + [("time", t) for t in TT_MIN]
    tot = {k: dict(dem=0.0, gacc=0.0, tre=0.0, mac=0.0) for k in keys}
    all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES

    for a2 in get_region("World").member_alpha2:
        obj = pycountry.countries.get(alpha_2=a2)
        if obj is None:
            continue
        name, iso3 = obj.name, obj.alpha_3
        try:
            gdf = load_population_at_resolution(name, target_resolution=RES)
            cg = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False, rtu_set="barton_2014")
        except Exception:
            continue
        dem = _demand(cg)
        if dem is None:
            continue
        try:
            locs, _ = load_linacs_from_dirac_db(name)
        except Exception:
            locs = []
        if not locs:
            for k in keys:  # no facilities -> demand counted, zero access
                tot[k]["dem"] += float(dem.sum())
            continue
        tt = _cache(iso3, locs)

        for d in DIST_KM:
            g, s = compute_accessibility(gdf, locs, model="step", max_distance_km=d,
                                         capacity_per_machine_per_year=_CAP, demand=dem,
                                         h3_resolution=RES)
            _accum(tot[("dist", d)], g, s)
        for t in TT_MIN:
            if tt is not None:
                g, s = compute_accessibility(gdf, locs, model="step", max_distance_km=t,
                                             capacity_per_machine_per_year=_CAP, demand=dem,
                                             h3_resolution=RES, travel_time_matrix=tt)
            else:  # fitted distance proxy
                g, s = compute_accessibility(gdf, locs, model="step", max_distance_km=KM_MAP[t],
                                             capacity_per_machine_per_year=_CAP, demand=dem,
                                             h3_resolution=RES)
            _accum(tot[("time", t)], g, s)

    rows = []
    for kind, thr in keys:
        d = tot[(kind, thr)]
        a_c = min(1.0, d["mac"] * _CAP / d["dem"]) if d["dem"] else float("nan")
        a_g = d["gacc"] / d["dem"] if d["dem"] else float("nan")
        a_rm = d["tre"] / d["dem"] if d["dem"] else float("nan")
        unit = "km" if kind == "dist" else "min"
        rows.append(dict(metric=kind, threshold=f"{thr:g} {unit}",
                         A_C=a_c, A_G=a_g, A_RM=a_rm))
        print(f"{kind:4} {thr:>4g} {unit:>3} ->  A_C={a_c:.1%}  A_G={a_g:.1%}  A_RM={a_rm:.1%}"
              f"{'   (A_RM>A_G!)' if a_rm > a_g + 1e-9 else ''}", flush=True)

    pd.DataFrame(rows).to_csv(_ROOT / "scripts" / "demand_weighted_table.csv", index=False)


if __name__ == "__main__":
    main()
