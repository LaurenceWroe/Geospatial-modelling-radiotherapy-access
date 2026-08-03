"""
H3-native radiotherapy accessibility computation.

Models
------
**Exponential**:
    P(d) = exp(-d / λ)

**Step function**:
    P(d) = 1 if d ≤ threshold, else 0

**Weibull survival**:
    P(d) = exp(-(d / λ)^k)

**Uniform**:
    P(d) = 1 for all hexes (no distance barrier)

where d is either haversine distance (km) or travel time (minutes) depending
on the ``travel_time_matrix`` argument.

When using distance:  λ in km,      cutoff in km,      threshold in km.
When using travel time: λ in minutes, cutoff in minutes, threshold in minutes.

Distances are great-circle (haversine, spherical Earth). The difference from
full ellipsoidal geodesics is < 0.5 % — negligible relative to model
uncertainty — and haversine is an order of magnitude faster.

Geographic access probability
------------------------------
Each centre counts once regardless of linac count:
    P_total = 1 - ∏_i (1 - P(d_i))

Capacity allocation
-------------------
Global-ring proportional allocation. Distance rings (one hex wide) are
processed in ascending order *across all facilities simultaneously*, so a
patient 10 km from facility A is served before a patient 200 km from
facility B. Within a ring, each facility serves demand weighted by
``remaining_demand × P(d)`` until its capacity is exhausted. This makes the
result independent of the order facilities appear in the input file (up to
same-ring ties, which are at most one hex width apart).

For the uniform model, any capacity left over after the ring pass is
distributed proportionally across all remaining unmet demand (no distance
barrier means no hex is out of reach).

    capacity_limited_probability = rt_treated_i / rt_demand_i
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon as _Polygon


_REFERENCE_CAPACITY: float = 450.0  # patients / machine / year

# Allocation entries with P(d) below this contribute nothing meaningful and
# are pruned to bound memory (weibull k=4 reaches 1e-9 by ~2.3 λ).
_P_MIN: float = 1e-9

# For the uniform model (P=1 everywhere) prune each facility's candidate
# hexes once cumulative initial demand exceeds this multiple of its capacity;
# the post-pass proportional distribution catches anything missed.
_UNIFORM_CAPTURE_FACTOR: float = 25.0

_EARTH_RADIUS_KM = 6371.0088


def _haversine_km(
    lat_deg: float,
    lon_deg: float,
    lats_rad: np.ndarray,
    lons_rad: np.ndarray,
    cos_lats: np.ndarray,
) -> np.ndarray:
    """Great-circle distance (km) from one point to arrays of pre-radianised points."""
    phi1 = math.radians(lat_deg)
    lmb1 = math.radians(lon_deg)
    dphi = lats_rad - phi1
    dlmb = lons_rad - lmb1
    a = np.sin(dphi / 2.0) ** 2 + math.cos(phi1) * cos_lats * np.sin(dlmb / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_accessibility(
    gdf: gpd.GeoDataFrame,
    linac_locations: List[Tuple[float, float, float]],
    lambda_km: float = 30.0,
    cutoff_km: Optional[float] = None,
    model: Literal["exponential", "step", "weibull", "uniform"] = "exponential",
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
    model : "exponential" | "step" | "weibull" | "uniform"
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
        each LINAC. When provided, replaces haversine distance in the
        probability model. Haversine distance is still used for ring-based
        capacity allocation. np.inf = unreachable.

    Returns
    -------
    gdf_out : GeoDataFrame with added columns:
        ``nearest_linac_km``             (float32, km — always haversine)
        ``nearest_linac_min``            (float32, minutes — only if travel_time_matrix provided)
        ``access_probability``           (float32, 0–1)
        ``capacity_limited_probability`` (float32, 0–1)
        ``rt_demand``                    (float32, patients/yr)
        ``rt_treated``                   (float32, patients/yr)
        ``rt_untreated``                 (float32, patients/yr)
        ``pop_with_access``              (float32)
    stats : dict
        Includes ``demand_fallback`` — True when *demand* was None and raw
        population was used as the demand proxy.
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

    g = gdf.copy()

    centroids = g["h3"].apply(lambda h: h3.cell_to_latlng(h))
    g["centroid_lat"] = centroids.apply(lambda c: c[0])
    g["centroid_lon"] = centroids.apply(lambda c: c[1])

    lats = g["centroid_lat"].to_numpy(dtype=np.float64)
    lons = g["centroid_lon"].to_numpy(dtype=np.float64)
    n = len(g)

    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    cos_lats = np.cos(lats_rad)

    pop = pd.to_numeric(g["population"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    pop = np.where(pop > 0, pop, 0.0)

    demand_fallback = demand is None
    rt_demand = np.where(demand > 0, demand, 0.0).astype(np.float64) if demand is not None else pop.copy()

    nearest_km = np.full(n, np.inf, dtype=np.float64)
    nearest_min = np.full(n, np.inf, dtype=np.float64) if travel_time_matrix is not None else None
    raw_weights = np.array([w for _, _, w in linac_locations], dtype=np.float64)
    capacities = raw_weights * capacity_per_machine_per_year

    complement = np.ones(n, dtype=np.float64)
    total_allocated = np.zeros(n, dtype=np.float64)
    remaining_demand = rt_demand.copy()

    try:
        _edge_km = h3.average_hexagon_edge_length(h3_resolution, unit="km")
        _ring_step_km = _edge_km * math.sqrt(3)
    except Exception:
        _area_km2 = h3.average_hexagon_area(h3_resolution, unit="km^2")
        _ring_step_km = math.sqrt(2.0 * _area_km2 / math.sqrt(3.0))

    # ------------------------------------------------------------------
    # Pass 1 — per facility: probability, nearest distance, and sparse
    # allocation candidates (facility, hex, ring, p).
    # ------------------------------------------------------------------
    ent_ring: List[np.ndarray] = []
    ent_fac: List[np.ndarray] = []
    ent_hex: List[np.ndarray] = []
    ent_p: List[np.ndarray] = []

    for j, (lat_f, lon_f, w) in enumerate(linac_locations):
        dists_km = _haversine_km(lat_f, lon_f, lats_rad, lons_rad, cos_lats)
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

        if capacities[j] <= 0:
            continue

        # --- collect sparse allocation candidates ---
        cand = np.where(p > _P_MIN)[0]
        if cand.size == 0:
            continue

        if model == "uniform" and cand.size == n:
            # Prune: keep nearest hexes until cumulative demand safely
            # exceeds capacity; the leftover pass catches the rest.
            order = np.argsort(dists_km)
            cum = np.cumsum(rt_demand[order])
            cut = int(np.searchsorted(cum, capacities[j] * _UNIFORM_CAPTURE_FACTOR)) + 1
            cand = order[:min(cut, n)]

        rings = np.round(dists_km[cand] / _ring_step_km).astype(np.int32)
        ent_ring.append(rings)
        ent_fac.append(np.full(cand.size, j, dtype=np.int32))
        ent_hex.append(cand.astype(np.int32))
        ent_p.append(p[cand].astype(np.float64))

    # ------------------------------------------------------------------
    # Pass 2 — global ring order: nearest patient-facility pairs first,
    # across ALL facilities. Result is independent of facility input order
    # (up to same-ring ties).
    # ------------------------------------------------------------------
    caps = capacities.copy()
    if ent_ring:
        a_ring = np.concatenate(ent_ring)
        a_fac = np.concatenate(ent_fac)
        a_hex = np.concatenate(ent_hex)
        a_p = np.concatenate(ent_p)

        # Canonical facility order (by coordinates) so same-ring ties are
        # broken deterministically regardless of input file order.
        canon = sorted(
            range(len(linac_locations)),
            key=lambda j: (linac_locations[j][0], linac_locations[j][1], linac_locations[j][2]),
        )
        rank = np.empty(len(linac_locations), dtype=np.int32)
        for r, j in enumerate(canon):
            rank[j] = r
        a_rank = rank[a_fac]

        order = np.lexsort((a_rank, a_ring))
        a_ring, a_fac, a_hex, a_p, a_rank = (
            a_ring[order], a_fac[order], a_hex[order], a_p[order], a_rank[order]
        )

        # Group boundaries where (ring, facility) changes
        key_change = np.empty(a_ring.size, dtype=bool)
        key_change[0] = True
        key_change[1:] = (np.diff(a_ring) != 0) | (np.diff(a_rank) != 0)
        starts = np.flatnonzero(key_change)
        ends = np.append(starts[1:], a_ring.size)

        for s, e in zip(starts, ends):
            j = int(a_fac[s])
            cap_j = caps[j]
            if cap_j <= 0.0:
                continue
            idx = a_hex[s:e]
            weights = remaining_demand[idx] * a_p[s:e]
            total_w = float(weights.sum())
            if total_w <= 0.0:
                continue
            if total_w <= cap_j:
                total_allocated[idx] += weights
                remaining_demand[idx] -= weights
                caps[j] = cap_j - total_w
            else:
                allocated = weights * (cap_j / total_w)
                total_allocated[idx] += allocated
                remaining_demand[idx] -= allocated
                caps[j] = 0.0

    # ------------------------------------------------------------------
    # Pass 3 — uniform model only: leftover capacity has no distance
    # barrier, so distribute it proportionally over remaining demand.
    # ------------------------------------------------------------------
    if model == "uniform":
        leftover = float(caps.sum())
        unmet = float(remaining_demand.sum())
        if leftover > 0.0 and unmet > 0.0:
            frac = min(1.0, leftover / unmet)
            extra = remaining_demand * frac
            total_allocated += extra
            remaining_demand -= extra

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
        "demand_fallback": demand_fallback,
    }
    return g, stats


def aggregate_access_gdf(gdf: gpd.GeoDataFrame, target_resolution: int) -> gpd.GeoDataFrame:
    """Aggregate an accessibility result GeoDataFrame to a coarser H3 resolution.

    Count columns (population, rt_demand, rt_treated, rt_untreated,
    pop_with_access) are summed. Probabilities are recomputed from the summed
    counts so they stay internally consistent. Distances/travel times are
    population-weighted means (weight floor of 1 so empty hexes still carry
    a distance).
    """
    native_res = h3.get_resolution(str(gdf["h3"].iloc[0]))
    if target_resolution >= native_res:
        return gdf

    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    df["h3_parent"] = df["h3"].apply(lambda h: h3.cell_to_parent(h, target_resolution))

    w = df["population"].clip(lower=1.0)
    df["_w"] = w
    df["_wp"] = df["access_probability"] * w
    # Distance means must exclude NaN children (no facility reachable) from
    # both numerator and denominator, so track a separate distance weight.
    km_valid = df["nearest_linac_km"].notna()
    df["_wd_km"] = (df["nearest_linac_km"] * w).where(km_valid, 0.0)
    df["_w_km"] = w.where(km_valid, 0.0)
    has_min = "nearest_linac_min" in df.columns
    if has_min:
        min_valid = df["nearest_linac_min"].notna()
        df["_wd_min"] = (df["nearest_linac_min"] * w).where(min_valid, 0.0)
        df["_w_min"] = w.where(min_valid, 0.0)

    grp = df.groupby("h3_parent")
    out = grp.agg(
        population=("population", "sum"),
        rt_demand=("rt_demand", "sum"),
        rt_treated=("rt_treated", "sum"),
        rt_untreated=("rt_untreated", "sum"),
        pop_with_access=("pop_with_access", "sum"),
        _w=("_w", "sum"),
        _wd_km=("_wd_km", "sum"),
        _w_km=("_w_km", "sum"),
        _wp=("_wp", "sum"),
    ).reset_index().rename(columns={"h3_parent": "h3"})

    with np.errstate(invalid="ignore", divide="ignore"):
        out["nearest_linac_km"] = (
            out["_wd_km"] / out["_w_km"].replace(0.0, np.nan)
        ).astype(np.float32)
        if has_min:
            wd_min = grp["_wd_min"].sum().to_numpy()
            w_min = grp["_w_min"].sum().to_numpy()
            out["nearest_linac_min"] = (
                wd_min / np.where(w_min > 0, w_min, np.nan)
            ).astype(np.float32)

    # Probability = population-weighted mean of child probabilities (floor-1
    # weights so uninhabited hexes still carry a value for display).
    out["access_probability"] = (out["_wp"] / out["_w"]).astype(np.float32)
    out = out.drop(columns=["_w", "_wd_km", "_w_km", "_wp"])

    dem = out["rt_demand"].to_numpy(np.float64)
    out["capacity_limited_probability"] = np.where(
        dem > 0, out["rt_treated"] / np.maximum(dem, 1e-9), 0.0
    ).astype(np.float32)

    # Carry through country attribution if present (mode of children)
    if "country" in df.columns:
        country_mode = grp["country"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
        out = out.merge(country_mode.reset_index().rename(columns={"h3_parent": "h3"}), on="h3", how="left")

    out["geometry"] = out["h3"].apply(
        lambda h: _Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h)])
    )
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
