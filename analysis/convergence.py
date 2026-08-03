"""
Resolution-convergence analysis for a single country.

Computes A_C / A_G / A_RM (all demand-weighted) across H3 resolutions 3–8 for
both straight-line distance and driving-time step thresholds, so you can see at
what resolution each metric stabilises.

- Distance: computed natively at every resolution (local, no API).
- Travel time: fetched natively from the TravelTime H3 API at res 5–8 (the
  supported range) and aggregated down (population-weighted) to res 3–4, mirroring
  how the app produces sub-res-5 travel times.

A_G is demand-weighted: Σ(geographic_access · demand) / Σ demand, consistent with
the app's A_C / A_G / A_RM definitions. A_RM = Σ treated / Σ demand.
A_C = min(1, Σ capacity / Σ demand) — resolution-invariant by construction.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Optional

import numpy as np
import pandas as pd

from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
from data.cancer import apportion_cancer_to_h3, get_cancer_types, DERIVED_CANCER_TYPES
from data.travel_time import compute_travel_time_matrix, aggregate_tt_matrix
from analysis.accessibility import compute_accessibility

_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
_CAP = 450.0
RES_ALL = [3, 4, 5, 6, 7, 8]
TT_NATIVE = [5, 6, 7, 8]          # resolutions the TravelTime API supports directly
TT_AGGREGATE_FROM = 5             # res 3–4 are aggregated from this
DIST_KM = [30.0, 60.0, 90.0, 120.0]
TIME_MIN = [30, 60, 90, 120]
_TT_BUDGET_SEC = 7200             # 120-minute isochrones


def _optimal_demand(cancer_gdf) -> Optional[np.ndarray]:
    cols = [c for c in cancer_gdf.columns if c.endswith("_optimal_rt")
            and c[:-len("_optimal_rt")].strip().lower() not in _AGG]
    if not cols:
        return None
    return cancer_gdf[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)


def _metrics(gdf_out, stats) -> tuple[float, float, float]:
    dem = stats["total_rt_demand"]
    a_g = float((gdf_out["access_probability"] * gdf_out["rt_demand"]).sum() / dem) if dem > 0 else float("nan")
    a_c = min(1.0, stats["total_national_capacity"] / dem) if dem > 0 else float("nan")
    a_rm = stats["total_rt_treated"] / dem if dem > 0 else float("nan")
    return a_c, a_g, a_rm


def run_convergence(country: str, iso3: str, app_id: str, api_key: str,
                    progress: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
    """Return a tidy DataFrame: [metric, resolution, threshold, unit, A_C, A_G, A_RM]."""
    def _tick(frac, msg):
        if progress:
            progress(frac, msg)

    try:
        locs, _ = load_linacs_from_dirac_db(country)
    except (ValueError, FileNotFoundError) as exc:
        raise ValueError(f"No radiotherapy facilities (DIRAC) found for {country}.") from exc
    if not locs:
        raise ValueError(f"No radiotherapy facilities (DIRAC) found for {country}.")
    linac_ll = [(lat, lon) for lat, lon, _ in locs]
    lh = hashlib.md5(json.dumps(sorted(linac_ll)).encode()).hexdigest()[:8]
    all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES

    # Load population once per resolution (stable hex order reused everywhere).
    gdf_by_res = {}
    for i, res in enumerate(RES_ALL):
        _tick(0.05 + 0.10 * i / len(RES_ALL), f"Loading population at res {res}…")
        gdf_by_res[res] = load_population_at_resolution(country, target_resolution=res)

    # Travel-time matrices: fetch native res 5–8, then aggregate down to 3–4.
    tt_by_res = {}
    errors = []
    for i, res in enumerate(TT_NATIVE):
        _tick(0.20 + 0.45 * i / len(TT_NATIVE), f"Fetching driving times at res {res}…")
        gdf = gdf_by_res[res]
        ck = f"{iso3}_res{res}_2h_{lh}"
        mat, errs = compute_travel_time_matrix(
            list(gdf["h3"]), linac_ll, res, "driving", app_id, api_key,
            cache_key=ck, max_travel_time_sec=_TT_BUDGET_SEC,
        )
        tt_by_res[res] = mat
        if errs:
            errors.append(f"res{res}: {len(errs)} batch error(s)")

    gdf5 = gdf_by_res[TT_AGGREGATE_FROM]
    pops5 = gdf5["population"].to_numpy(np.float64)
    for res in (3, 4):
        _tick(0.66, f"Aggregating driving times to res {res}…")
        tt_by_res[res] = aggregate_tt_matrix(
            tt_by_res[TT_AGGREGATE_FROM], list(gdf5["h3"]), pops5,
            list(gdf_by_res[res]["h3"]), res, unreachable_fill_min=_TT_BUDGET_SEC / 60.0,
        )

    rows = []
    for i, res in enumerate(RES_ALL):
        _tick(0.70 + 0.28 * i / len(RES_ALL), f"Computing access metrics at res {res}…")
        gdf = gdf_by_res[res]
        cg = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False)
        demand = _optimal_demand(cg)
        if demand is None:
            continue

        for thr in DIST_KM:
            g, s = compute_accessibility(gdf, locs, model="step", max_distance_km=thr,
                                         capacity_per_machine_per_year=_CAP, demand=demand,
                                         h3_resolution=res)
            a_c, a_g, a_rm = _metrics(g, s)
            rows.append(dict(metric="distance", resolution=res, threshold=thr,
                             unit="km", A_C=a_c, A_G=a_g, A_RM=a_rm))

        mat = tt_by_res[res]
        for thr in TIME_MIN:
            g, s = compute_accessibility(gdf, locs, model="step", max_distance_km=thr,
                                         capacity_per_machine_per_year=_CAP, demand=demand,
                                         h3_resolution=res, travel_time_matrix=mat)
            a_c, a_g, a_rm = _metrics(g, s)
            rows.append(dict(metric="time", resolution=res, threshold=thr,
                             unit="min", A_C=a_c, A_G=a_g, A_RM=a_rm))

    _tick(1.0, "Done")
    df = pd.DataFrame(rows)
    df.attrs["errors"] = errors
    df.attrs["n_facilities"] = len(locs)
    return df
