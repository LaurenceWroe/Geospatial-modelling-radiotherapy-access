"""
H3-native radiotherapy accessibility computation.

Models
------
**Exponential** (default):
    P_total = 1 - ∏_i (1 - exp(-d_i / λ))^w_i

**Step function**:
    P_total = 1 if any facility within max_distance_km, else 0

**Uniform**:
    P_total = 1 for all hexes (no distance barrier)

where:
    d_i   geodesic distance (km) from centroid to facility i
    λ     distance-decay parameter (km)
    w_i   weight (≥ 1, scales with number of linacs at facility i)

Distances are computed vectorised with pyproj.Geod for WGS-84 accuracy.
Facilities beyond ``cutoff_km`` (default 5λ) are treated as zero probability
in the exponential model.

Two probability outputs
-----------------------
``access_probability``
    Geographic accessibility — probability of reaching a facility,
    ignoring capacity.

``capacity_limited_probability``
    Greedy nearest-first allocation.  RT demand per hex is supplied by the
    caller (typically summed Optimal RT cases from GLOBOCAN data).  For each
    facility, hexes are processed in order of increasing distance; the demand
    placed on facility j from hex i is ``rt_demand_i × p_ij``.  This is
    subtracted from the facility's annual capacity until exhausted.
    ``capacity_limited_probability = rt_treated_i / rt_demand_i``.
    If no demand array is provided, raw population is used as a fallback.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from pyproj import Geod


_REFERENCE_CAPACITY: float = 450.0  # patients / machine / year — weight normalisation


def compute_accessibility(
    gdf: gpd.GeoDataFrame,
    linac_locations: List[Tuple[float, float, float]],
    lambda_km: float = 30.0,
    cutoff_km: Optional[float] = None,
    use_weights: bool = True,
    model: Literal["exponential", "step"] = "exponential",
    max_distance_km: float = 50.0,
    capacity_per_machine_per_year: float = _REFERENCE_CAPACITY,
    demand: Optional[np.ndarray] = None,
    snap_linacs_to_hex: bool = False,
    h3_resolution: int = 8,
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """Compute radiotherapy access probability for each H3 cell.

    Parameters
    ----------
    gdf : GeoDataFrame
        H3 population GeoDataFrame with columns ``h3`` and ``population``.
    linac_locations : list of (lat, lon, n_linacs)
        LINAC facility positions and machine counts.
    lambda_km : float
        Distance-decay half-length in km (exponential model only).
    cutoff_km : float, optional
        Hard cut-off distance; defaults to 5 × lambda_km.
    use_weights : bool
        Scale facility contribution by machine count in the geographic model.
    model : "exponential" | "step"
        Probability model.
    max_distance_km : float
        Maximum reach for the step-function model.
    capacity_per_machine_per_year : float
        Throughput per machine (patients/year).
    demand : np.ndarray, optional
        Annual RT patient demand per hex (same length as gdf rows).
        If provided, used directly for capacity allocation.
        If None, raw population is used as a fallback.

    Returns
    -------
    gdf_out : GeoDataFrame with added columns:
        ``nearest_linac_km``             (float32, km)
        ``access_probability``           (float32, 0–1)
        ``capacity_limited_probability`` (float32, 0–1)
        ``rt_demand``                    (float32, RT patients/yr per hex)
        ``rt_treated``                   (float32, RT patients/yr treated)
        ``rt_untreated``                 (float32, RT patients/yr untreated)
        ``pop_with_access``              (float32, backward compat)
    stats : dict
    """
    if cutoff_km is None:
        cutoff_km = 10.0 * lambda_km

    # Optionally snap each linac to the centroid of its H3 cell at the target resolution.
    # Multiple linacs in the same hex are merged into a single entry with summed machine count.
    if snap_linacs_to_hex:
        snapped: dict[str, float] = {}
        for lat_f, lon_f, w in linac_locations:
            cell = h3.latlng_to_cell(lat_f, lon_f, h3_resolution)
            snapped[cell] = snapped.get(cell, 0.0) + w
        linac_locations = [
            (h3.cell_to_latlng(cell)[0], h3.cell_to_latlng(cell)[1], w)
            for cell, w in snapped.items()
        ]

    geod = Geod(ellps="WGS84")
    g = gdf.copy()

    centroids = g["h3"].apply(lambda h: h3.cell_to_latlng(h))
    g["centroid_lat"] = centroids.apply(lambda c: c[0])
    g["centroid_lon"] = centroids.apply(lambda c: c[1])

    lats = g["centroid_lat"].to_numpy(dtype=np.float64)
    lons = g["centroid_lon"].to_numpy(dtype=np.float64)
    n = len(g)

    pop = pd.to_numeric(g["population"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    pop = np.where(pop > 0, pop, 0.0)

    # RT demand per hex: use supplied cancer data, or fall back to raw population
    if demand is not None:
        rt_demand = np.where(demand > 0, demand, 0.0).astype(np.float64)
    else:
        rt_demand = pop.copy()

    nearest_km = np.full(n, np.inf, dtype=np.float64)
    raw_weights = np.array([w for _, _, w in linac_locations], dtype=np.float64)

    # Geographic access probability accumulator
    cap_factor = capacity_per_machine_per_year / _REFERENCE_CAPACITY
    complement = np.ones(n, dtype=np.float64)

    # Capacity-limited allocation accumulator (RT patients/year)
    total_allocated = np.zeros(n, dtype=np.float64)
    # Track unmet demand per hex so each facility only serves genuinely unmet need.
    # Without this, multiple facilities each "see" the full demand of a nearby hex,
    # wasting their capacity on already-served patients and leaving adjacent hexes
    # with nothing.
    remaining_demand = rt_demand.copy()

    for j, (lat_f, lon_f, w) in enumerate(linac_locations):
        _, _, dists_m = geod.inv(
            np.full(n, lon_f),
            np.full(n, lat_f),
            lons,
            lats,
        )
        dists_km = dists_m * 1e-3
        np.minimum(nearest_km, dists_km, out=nearest_km)

        # --- geographic access probability ---
        if model == "step":
            p = np.where(dists_km <= max_distance_km, 1.0, 0.0)
        elif model == "uniform":
            p = np.ones(n, dtype=np.float64)
        else:  # exponential
            p = np.exp(-dists_km / lambda_km)
            p = np.where(dists_km <= cutoff_km, p, 0.0)

        eff_w = (w * cap_factor) if use_weights else 1.0
        complement *= np.power(np.maximum(1.0 - p, 0.0), eff_w)

        # --- greedy nearest-first capacity allocation ---
        cap_j = w * capacity_per_machine_per_year
        if cap_j <= 0:
            continue

        sorted_idx = np.argsort(dists_km)
        p_sorted = p[sorted_idx]                   # cutoff already applied
        remaining_sorted = remaining_demand[sorted_idx]

        demands_sorted = remaining_sorted * p_sorted

        cum = np.cumsum(demands_sorted)
        prev_cum = np.empty_like(cum)
        prev_cum[0] = 0.0
        prev_cum[1:] = cum[:-1]

        remaining_before = np.maximum(0.0, cap_j - prev_cum)
        allocated_sorted = np.minimum(demands_sorted, remaining_before)

        total_allocated[sorted_idx] += allocated_sorted
        remaining_demand[sorted_idx] -= allocated_sorted

    # --- assemble results ---
    prob = 1.0 - complement

    rt_treated = np.minimum(rt_demand, total_allocated)
    rt_untreated = rt_demand - rt_treated
    cap_limited_prob = np.where(rt_demand > 0, rt_treated / rt_demand, 0.0)

    g["access_probability"] = prob.astype(np.float32)
    g["capacity_limited_probability"] = cap_limited_prob.astype(np.float32)
    nearest_km[np.isinf(nearest_km)] = np.nan
    g["nearest_linac_km"] = nearest_km.astype(np.float32)
    g["rt_demand"] = rt_demand.astype(np.float32)
    g["rt_treated"] = rt_treated.astype(np.float32)
    g["rt_untreated"] = rt_untreated.astype(np.float32)
    g["pop_with_access"] = (prob * pop).astype(np.float32)

    total_pop = float(np.nansum(pop))
    total_rt_demand = float(np.nansum(rt_demand))
    total_rt_treated = float(np.nansum(rt_treated))
    total_machines = float(raw_weights.sum())

    stats = {
        "n_facilities": len(linac_locations),
        "total_machines": total_machines,
        "capacity_per_machine_per_year": capacity_per_machine_per_year,
        "total_national_capacity": total_machines * capacity_per_machine_per_year,
        "total_rt_demand": total_rt_demand,
        "total_rt_treated": total_rt_treated,
        "total_rt_untreated": total_rt_demand - total_rt_treated,
        "model": model,
        "lambda_km": lambda_km,
        "cutoff_km": cutoff_km,
        "max_distance_km": max_distance_km,
        "total_population": total_pop,
        "pop_with_access": float(np.nansum(prob * pop)),
        "mean_access_probability": float(np.nansum(prob * pop)) / total_pop if total_pop > 0 else 0.0,
        "mean_capacity_limited_probability": total_rt_treated / total_rt_demand if total_rt_demand > 0 else 0.0,
        "n_hexagons": n,
    }
    return g, stats
