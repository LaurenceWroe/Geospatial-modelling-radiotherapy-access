"""
H3-native radiotherapy accessibility computation.

Models
------
**Exponential** (default):
    P(d) = exp(-d / λ)

**Step function**:
    P(d) = 1 if d ≤ threshold, else 0

**Weibull survival**:
    P(d) = exp(-(d / λ)^k)

**Uniform**:
    P(d) = 1 for all hexes (no distance barrier)

where d is either geodesic distance (km) or travel time (minutes) depending
on the ``travel_time_matrix`` argument.

When using distance:  λ in km,      cutoff in km,      threshold in km.
When using travel time: λ in minutes, cutoff in minutes, threshold in minutes.

Geographic access probability
------------------------------
Each centre counts once regardless of linac count:
    P_total = 1 - ∏_i (1 - P(d_i))

Capacity allocation
-------------------
Ring-based proportional allocation using geodesic distance rings (even when
travel time is used for the probability model). Rings are processed nearest
first; each ring is served in full until capacity is exhausted, at which
point remaining capacity is split proportionally by demand weight.

    capacity_limited_probability = rt_treated_i / rt_demand_i
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from pyproj import Geod


_REFERENCE_CAPACITY: float = 450.0  # patients / machine / year


def compute_accessibility(
    gdf: gpd.GeoDataFrame,
    linac_locations: List[Tuple[float, float, float]],
    lambda_km: float = 30.0,
    cutoff_km: Optional[float] = None,
    model: Literal["exponential", "step", "weibull"] = "exponential",
    max_distance_km: float = 50.0,
    weibull_k: float = 2.0,
    capacity_per_machine_per_year: float = _REFERENCE_CAPACITY,
    demand: Optional[np.ndarray] = None,
    snap_linacs_to_hex: bool = False,
    h3_resolution: int = 8,
    travel_time_matrix: Optional[np.ndarray] = None,
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """Compute radiotherapy access probability for each H3 cell.

    Parameters
    ----------
    gdf : GeoDataFrame
        H3 population GeoDataFrame with columns ``h3`` and ``population``.
    linac_locations : list of (lat, lon, n_linacs)
        LINAC facility positions and machine counts.
    lambda_km : float
        Scale parameter for exponential/Weibull models.
        In km when using distance; in minutes when using travel_time_matrix.
    cutoff_km : float, optional
        Hard cut-off; defaults to 10 × lambda_km (same units as lambda_km).
    model : "exponential" | "step" | "weibull"
        Probability model.
    max_distance_km : float
        Step-function threshold (km or minutes depending on mode).
    weibull_k : float
        Weibull shape parameter.
    capacity_per_machine_per_year : float
        Throughput per machine (patients/year).
    demand : np.ndarray, optional
        Annual RT patient demand per hex. Falls back to population if None.
    snap_linacs_to_hex : bool
        Snap each linac to its H3 cell centroid and merge same-cell linacs.
    h3_resolution : int
        H3 resolution of the gdf (used for ring-step calculation).
    travel_time_matrix : np.ndarray, optional
        Shape (n_hexes, n_linacs), travel time in minutes from each hex to
        each LINAC. When provided, replaces geodesic distance in the
        probability model. Geodesic distance is still used for ring-based
        capacity allocation. np.inf = unreachable.

    Returns
    -------
    gdf_out : GeoDataFrame with added columns:
        ``nearest_linac_km``             (float32, km — always geodesic)
        ``nearest_linac_min``            (float32, minutes — only if travel_time_matrix provided)
        ``access_probability``           (float32, 0–1)
        ``capacity_limited_probability`` (float32, 0–1)
        ``rt_demand``                    (float32, patients/yr)
        ``rt_treated``                   (float32, patients/yr)
        ``rt_untreated``                 (float32, patients/yr)
        ``pop_with_access``              (float32)
    stats : dict
    """
    if cutoff_km is None:
        cutoff_km = 10.0 * lambda_km

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

    rt_demand = np.where(demand > 0, demand, 0.0).astype(np.float64) if demand is not None else pop.copy()

    nearest_km = np.full(n, np.inf, dtype=np.float64)
    nearest_min = np.full(n, np.inf, dtype=np.float64) if travel_time_matrix is not None else None
    raw_weights = np.array([w for _, _, w in linac_locations], dtype=np.float64)

    complement = np.ones(n, dtype=np.float64)
    total_allocated = np.zeros(n, dtype=np.float64)
    remaining_demand = rt_demand.copy()

    try:
        _edge_km = h3.average_hexagon_edge_length(h3_resolution, unit="km")
        _ring_step_km = _edge_km * math.sqrt(3)
    except Exception:
        _area_km2 = h3.average_hexagon_area(h3_resolution, unit="km^2")
        _ring_step_km = math.sqrt(2.0 * _area_km2 / math.sqrt(3.0))

    for j, (lat_f, lon_f, w) in enumerate(linac_locations):
        # Always compute geodesic distances for ring allocation and nearest_km
        _, _, dists_m = geod.inv(
            np.full(n, lon_f),
            np.full(n, lat_f),
            lons,
            lats,
        )
        dists_km = dists_m * 1e-3
        np.minimum(nearest_km, dists_km, out=nearest_km)

        # Effective distances for probability model
        if travel_time_matrix is not None:
            eff_dists = travel_time_matrix[:, j].astype(np.float64)
            np.minimum(nearest_min, eff_dists, out=nearest_min)
        else:
            eff_dists = dists_km

        # --- geographic access probability ---
        if model == "step":
            p = np.where(eff_dists <= max_distance_km, 1.0, 0.0)
        elif model == "uniform":
            p = np.ones(n, dtype=np.float64)
        elif model == "weibull":
            p = np.exp(-np.power(
                np.where(np.isfinite(eff_dists), eff_dists, 1e9) / lambda_km,
                weibull_k,
            ))
            p = np.where(eff_dists <= cutoff_km, p, 0.0)
        else:  # exponential
            p = np.exp(-np.where(np.isfinite(eff_dists), eff_dists, 1e9) / lambda_km)
            p = np.where(eff_dists <= cutoff_km, p, 0.0)

        # Each centre counts once regardless of linac count
        complement *= np.maximum(1.0 - p, 0.0)

        # --- ring-based proportional capacity allocation (always geodesic) ---
        cap_j = w * capacity_per_machine_per_year
        if cap_j <= 0:
            continue

        in_range = p > 0.0
        if not in_range.any():
            continue

        ring_nums = np.round(dists_km / _ring_step_km).astype(np.int32)
        unique_rings = np.unique(ring_nums[in_range])

        for ring_k in unique_rings:
            if cap_j <= 0.0:
                break

            ring_idx = np.where(in_range & (ring_nums == ring_k))[0]
            if ring_idx.size == 0:
                continue

            ring_demands = remaining_demand[ring_idx] * p[ring_idx]
            total_ring_demand = float(ring_demands.sum())
            if total_ring_demand <= 0.0:
                continue

            if total_ring_demand <= cap_j:
                total_allocated[ring_idx] += ring_demands
                remaining_demand[ring_idx] -= ring_demands
                cap_j -= total_ring_demand
            else:
                fracs = ring_demands / total_ring_demand
                allocated = cap_j * fracs
                total_allocated[ring_idx] += allocated
                remaining_demand[ring_idx] -= allocated
                cap_j = 0.0

    # --- assemble results ---
    prob = 1.0 - complement
    rt_treated = np.minimum(rt_demand, total_allocated)
    rt_untreated = rt_demand - rt_treated
    cap_limited_prob = np.where(rt_demand > 0, rt_treated / rt_demand, 0.0)

    g["access_probability"] = prob.astype(np.float32)
    g["capacity_limited_probability"] = cap_limited_prob.astype(np.float32)
    nearest_km[np.isinf(nearest_km)] = np.nan
    g["nearest_linac_km"] = nearest_km.astype(np.float32)
    if nearest_min is not None:
        nearest_min[np.isinf(nearest_min)] = np.nan
        g["nearest_linac_min"] = nearest_min.astype(np.float32)
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
        "using_travel_time": travel_time_matrix is not None,
    }
    return g, stats
