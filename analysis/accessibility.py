"""
H3-native radiotherapy accessibility computation.

Decay models
------------
See :mod:`analysis.decay`. ``x`` is either haversine distance (km) or travel
time (minutes) depending on the ``travel_time_matrix`` argument, and every
decay parameter is expressed in the same unit.

Distances are great-circle (haversine, spherical Earth). The difference from
full ellipsoidal geodesics is < 0.5 % — negligible relative to model
uncertainty — and haversine is an order of magnitude faster.

Geographic access probability
------------------------------
Each centre counts once regardless of linac count:
    P_total = 1 - ∏_i (1 - P(x_i))

Capacity allocation (``allocation`` argument)
---------------------------------------------
``nearest_first`` (default, the RadMaps rule)
    Global-ring proportional allocation. Rings of the access variable x (one
    hex wide for distance, ``ring_step_min`` for travel time) are processed in ascending order *across all facilities simultaneously*, so a
    patient 10 km from facility A is served before a patient 200 km from
    facility B. Within a ring, each facility serves demand weighted by
    ``remaining_demand × P(x)`` until its capacity is exhausted. The result is
    independent of the order facilities appear in the input (same-ring ties
    are broken by a canonical coordinate sort). For the uniform model, any
    capacity left after the ring pass is spread proportionally over all
    remaining unmet demand.

``proportional`` (capacity-capped balanced floating catchment)
    Each hexagon's geographically accessible demand ``D_h · P_h`` is split
    across the facilities it can reach in proportion to the decay weight
    (optionally × facility capacity, Huff-style, with ``size_weighting``).
    Each facility then serves the demand it receives at the ratio
    ``min(1, capacity / demand received)``. Shortage is shared equally by
    everyone in a facility's reach rather than rationed nearest-first. This is
    the row/column-normalised allocation of Paez, Higgins & Vivona (2019) with
    a hard capacity cap, so patients and capacity are each counted once.

``optimal`` (upper bound)
    Linear programme maximising total treated patients subject to the same
    per-hexagon accessible demand, per-pair ``D_h · P(x_hj)`` and per-facility
    capacity limits, with total travel as a tie-breaker. Shows how much
    capacity the greedy rule strands when catchments overlap.

In every mode:
    rt_treated_h ≤ rt_demand_h,  Σ_h rt_treated_h ≤ Σ_j capacity_j,
    capacity_limited_probability_h = rt_treated_h / rt_demand_h
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon as _Polygon

from analysis.decay import DECAY_MODELS, decay_probability, default_cutoff

_REFERENCE_CAPACITY: float = 450.0  # patients / machine / year

ALLOCATION_RULES = ("nearest_first", "proportional", "optimal")

# Allocation entries with P(d) below this contribute nothing meaningful and
# are pruned to bound memory (weibull k=4 reaches 1e-9 by ~2.3 λ).
_P_MIN: float = 1e-9

# For the uniform model (P=1 everywhere) prune each facility's candidate
# hexes once cumulative initial demand exceeds this multiple of its capacity;
# the post-pass proportional distribution catches anything missed.
_UNIFORM_CAPTURE_FACTOR: float = 25.0

# Largest sparse hex-facility problem the LP solver will accept interactively.
_OPTIMAL_MAX_ENTRIES: int = 3_000_000

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


def snap_linacs(
    linac_locations: List[Tuple[float, float, float]], h3_resolution: int
) -> List[Tuple[float, float, float]]:
    """Snap each facility to its H3 cell centroid and merge same-cell facilities."""
    snapped: dict[str, float] = {}
    for lat_f, lon_f, w in linac_locations:
        cell = h3.latlng_to_cell(lat_f, lon_f, h3_resolution)
        snapped[cell] = snapped.get(cell, 0.0) + w
    return [
        (h3.cell_to_latlng(cell)[0], h3.cell_to_latlng(cell)[1], w)
        for cell, w in snapped.items()
    ]


def hex_centroids(gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) float64 arrays of H3 cell centroids for *gdf*."""
    centroids = gdf["h3"].apply(lambda h: h3.cell_to_latlng(h))
    lats = np.fromiter((c[0] for c in centroids), dtype=np.float64, count=len(gdf))
    lons = np.fromiter((c[1] for c in centroids), dtype=np.float64, count=len(gdf))
    return lats, lons


def iter_facility_access(
    lats: np.ndarray,
    lons: np.ndarray,
    linac_locations: List[Tuple[float, float, float]],
    *,
    model: str,
    scale: float,
    shape: float,
    threshold: float,
    cutoff: Optional[float],
    travel_time_matrix: Optional[np.ndarray] = None,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(j, p, dist_km, x)`` for each facility *j*.

    ``p`` is the decay probability for every hexagon, ``dist_km`` the haversine
    distance and ``x`` the geographic variable the decay was evaluated on
    (``dist_km`` or the travel-time column).
    """
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    cos_lats = np.cos(lats_rad)
    for j, (lat_f, lon_f, _w) in enumerate(linac_locations):
        dists_km = _haversine_km(lat_f, lon_f, lats_rad, lons_rad, cos_lats)
        if travel_time_matrix is not None:
            x = travel_time_matrix[:, j].astype(np.float64)
        else:
            x = dists_km
        p = decay_probability(x, model, scale=scale, shape=shape,
                              threshold=threshold, cutoff=cutoff)
        yield j, p, dists_km, x


def _ring_step_km(h3_resolution: int) -> float:
    try:
        edge_km = h3.average_hexagon_edge_length(h3_resolution, unit="km")
        return edge_km * math.sqrt(3)
    except Exception:
        area_km2 = h3.average_hexagon_area(h3_resolution, unit="km^2")
        return math.sqrt(2.0 * area_km2 / math.sqrt(3.0))


# ---------------------------------------------------------------------------
# Allocation rules
# ---------------------------------------------------------------------------

def _allocate_nearest_first(
    n: int,
    rt_demand: np.ndarray,
    capacities: np.ndarray,
    linac_locations: List[Tuple[float, float, float]],
    ent_ring: List[np.ndarray],
    ent_fac: List[np.ndarray],
    ent_hex: List[np.ndarray],
    ent_p: List[np.ndarray],
    model: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Global-ring nearest-first allocation. Returns (allocated, remaining caps)."""
    total_allocated = np.zeros(n, dtype=np.float64)
    remaining_demand = rt_demand.copy()
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

    # Uniform model only: leftover capacity has no distance barrier, so
    # distribute it proportionally over remaining demand.
    if model == "uniform":
        leftover = float(caps.sum())
        unmet = float(remaining_demand.sum())
        if leftover > 0.0 and unmet > 0.0:
            frac = min(1.0, leftover / unmet)
            extra = remaining_demand * frac
            total_allocated += extra
            remaining_demand -= extra
            caps *= (1.0 - min(1.0, unmet / leftover)) if leftover > 0 else 0.0

    return total_allocated, caps


def _allocate_proportional(
    n: int,
    accessible: np.ndarray,
    capacities: np.ndarray,
    a_hex: np.ndarray,
    a_fac: np.ndarray,
    a_p: np.ndarray,
    size_weighting: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Capacity-capped balanced-FCA allocation. Returns (allocated, remaining caps,
    demand Q_f directed to each facility)."""
    n_fac = capacities.size
    allocated = np.zeros(n, dtype=np.float64)
    if a_hex.size == 0:
        return allocated, capacities.copy(), np.zeros(n_fac)

    w = a_p * (capacities[a_fac] if size_weighting else 1.0)
    row_sum = np.bincount(a_hex, weights=w, minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(row_sum[a_hex] > 0, w / row_sum[a_hex], 0.0)

    demand_sent = accessible[a_hex] * share                      # hex -> facility
    demand_on_fac = np.bincount(a_fac, weights=demand_sent, minlength=n_fac)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(demand_on_fac > 0, np.minimum(1.0, capacities / demand_on_fac), 1.0)

    served = demand_sent * ratio[a_fac]
    allocated = np.bincount(a_hex, weights=served, minlength=n)
    used = np.bincount(a_fac, weights=served, minlength=n_fac)
    return allocated, np.maximum(capacities - used, 0.0), demand_on_fac


def _allocate_optimal(
    n: int,
    rt_demand: np.ndarray,
    accessible: np.ndarray,
    capacities: np.ndarray,
    a_hex: np.ndarray,
    a_fac: np.ndarray,
    a_p: np.ndarray,
    a_dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """LP upper bound on treated patients (min-cost flow). Returns (allocated, remaining caps)."""
    from scipy import sparse
    from scipy.optimize import linprog

    n_fac = capacities.size
    m = a_hex.size
    if m == 0:
        return np.zeros(n, dtype=np.float64), capacities.copy()
    if m > _OPTIMAL_MAX_ENTRIES:
        raise ValueError(
            f"Optimal allocation problem too large ({m:,} hex–facility pairs > "
            f"{_OPTIMAL_MAX_ENTRIES:,}). Use a coarser resolution, a shorter cut-off, "
            "or the nearest-first / proportional rules."
        )

    # Keep only hexes with accessible demand and facilities with capacity.
    keep = (accessible[a_hex] > 0) & (capacities[a_fac] > 0) & (a_p > 0)
    a_hex, a_fac, a_p, a_dist = a_hex[keep], a_fac[keep], a_p[keep], a_dist[keep]
    m = a_hex.size
    if m == 0:
        return np.zeros(n, dtype=np.float64), capacities.copy()

    hex_ids, hex_inv = np.unique(a_hex, return_inverse=True)
    fac_ids, fac_inv = np.unique(a_fac, return_inverse=True)
    n_h, n_f = hex_ids.size, fac_ids.size

    rows = np.concatenate([hex_inv, n_h + fac_inv])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    data = np.ones(2 * m, dtype=np.float64)
    A_ub = sparse.csr_matrix((data, (rows, cols)), shape=(n_h + n_f, m))
    b_ub = np.concatenate([accessible[hex_ids], capacities[fac_ids]])

    # Each unit treated is worth more than any travel cost; distance breaks ties.
    big = float(np.nanmax(a_dist[np.isfinite(a_dist)]) if np.isfinite(a_dist).any() else 1.0) + 1.0
    cost = np.where(np.isfinite(a_dist), a_dist, big) - 2.0 * big
    bounds = np.column_stack([np.zeros(m), rt_demand[a_hex] * a_p])

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Optimal allocation LP failed: {res.message}")
    x = np.clip(res.x, 0.0, None)

    allocated = np.bincount(a_hex, weights=x, minlength=n)
    used = np.bincount(a_fac, weights=x, minlength=n_fac)
    return allocated, np.maximum(capacities - used, 0.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_accessibility(
    gdf: gpd.GeoDataFrame,
    linac_locations: List[Tuple[float, float, float]],
    lambda_km: float = 30.0,
    cutoff_km: Optional[float] = None,
    model: str = "exponential",
    max_distance_km: float = 50.0,
    weibull_k: float = 2.0,
    capacity_per_machine_per_year: float = _REFERENCE_CAPACITY,
    demand: Optional[np.ndarray] = None,
    snap_linacs_to_hex: bool = False,
    h3_resolution: int = 8,
    travel_time_matrix: Optional[np.ndarray] = None,
    allocation: Literal["nearest_first", "proportional", "optimal"] = "nearest_first",
    size_weighting: bool = True,
    ring_step_min: float = 5.0,
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """Compute radiotherapy access probability for each H3 cell.

    Parameters
    ----------
    gdf : GeoDataFrame
        H3 population GeoDataFrame with columns ``h3`` and ``population``.
    linac_locations : list of (lat, lon, n_linacs)
        LINAC facility positions and machine counts (fractional counts are
        allowed and represent capacity overrides).
    lambda_km : float
        Scale parameter for exponential / Weibull / power models.
        In km when using distance; in minutes when using travel_time_matrix.
    cutoff_km : float, optional
        Hard cut-off; defaults to 10 × lambda_km (same units as lambda_km).
    model : str
        One of :data:`analysis.decay.DECAY_MODELS`.
    max_distance_km : float
        Threshold for the step and Gaussian models (km or minutes).
    weibull_k : float
        Shape parameter (Weibull k, or power-law exponent β).
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
        each LINAC. When provided, replaces haversine distance in the decay
        model and in the ring ordering of the nearest-first rule.
        np.inf = unreachable.
    allocation : "nearest_first" | "proportional" | "optimal"
        Capacity rationing rule; see module docstring.
    size_weighting : bool
        Proportional rule only: split demand by decay weight × facility
        capacity (Huff-style attraction) instead of decay weight alone.
    ring_step_min : float
        Nearest-first ring width in minutes when travel_time_matrix is
        provided; with distance the ring width is one hexagon width.

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
        ``sfca3_ratio``                  (float32, plain 3SFCA index, uncapped)
        ``sfca3_supply``                 (float32, sfca3_ratio × rt_demand; sums on aggregation)
    stats : dict
        Includes ``demand_fallback`` — True when *demand* was None and raw
        population was used as the demand proxy — plus ``unused_capacity``
        and the demand-weighted geographic access ``demand_weighted_access``.
    """
    if model not in DECAY_MODELS:
        raise ValueError(f"Unknown decay model {model!r}")
    if allocation not in ALLOCATION_RULES:
        raise ValueError(f"Unknown allocation rule {allocation!r}; expected one of {ALLOCATION_RULES}")
    if cutoff_km is None:
        cutoff_km = default_cutoff(model, lambda_km, max_distance_km)

    if snap_linacs_to_hex:
        linac_locations = snap_linacs(linac_locations, h3_resolution)

    g = gdf.copy()
    lats, lons = hex_centroids(g)
    g["centroid_lat"] = lats
    g["centroid_lon"] = lons
    n = len(g)

    pop = pd.to_numeric(g["population"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    pop = np.where(pop > 0, pop, 0.0)

    demand_fallback = demand is None
    rt_demand = np.where(demand > 0, demand, 0.0).astype(np.float64) if demand is not None else pop.copy()

    nearest_km = np.full(n, np.inf, dtype=np.float64)
    nearest_min = np.full(n, np.inf, dtype=np.float64) if travel_time_matrix is not None else None
    raw_weights = np.array([w for _, _, w in linac_locations], dtype=np.float64)
    capacities = raw_weights * capacity_per_machine_per_year

    complement = np.ones(n, dtype=np.float64)
    # Rings are in units of the access variable x: one hexagon width (km)
    # for distance, ring_step_min (minutes) for travel time.
    ring_step = ring_step_min if travel_time_matrix is not None else _ring_step_km(h3_resolution)

    # ------------------------------------------------------------------
    # Pass 1 — per facility: probability, nearest distance, and sparse
    # allocation candidates (facility, hex, ring, p, distance).
    # ------------------------------------------------------------------
    ent_ring: List[np.ndarray] = []
    ent_fac: List[np.ndarray] = []
    ent_hex: List[np.ndarray] = []
    ent_p: List[np.ndarray] = []
    ent_dist: List[np.ndarray] = []

    for j, p, dists_km, x in iter_facility_access(
        lats, lons, linac_locations, model=model, scale=lambda_km, shape=weibull_k,
        threshold=max_distance_km, cutoff=cutoff_km, travel_time_matrix=travel_time_matrix,
    ):
        np.minimum(nearest_km, dists_km, out=nearest_km)
        if nearest_min is not None:
            np.minimum(nearest_min, x, out=nearest_min)

        # Each centre counts once regardless of linac count
        complement *= np.maximum(1.0 - p, 0.0)

        if capacities[j] <= 0:
            continue

        cand = np.where(p > _P_MIN)[0]
        if cand.size == 0:
            continue

        if model == "uniform" and cand.size == n and allocation == "nearest_first":
            # Prune: keep nearest hexes until cumulative demand safely
            # exceeds capacity; the leftover pass catches the rest.
            order = np.argsort(x)
            cum = np.cumsum(rt_demand[order])
            cut = int(np.searchsorted(cum, capacities[j] * _UNIFORM_CAPTURE_FACTOR)) + 1
            cand = order[:min(cut, n)]

        ent_ring.append(np.round(x[cand] / ring_step).astype(np.int32))
        ent_fac.append(np.full(cand.size, j, dtype=np.int32))
        ent_hex.append(cand.astype(np.int32))
        ent_p.append(p[cand].astype(np.float64))
        ent_dist.append((x[cand] if travel_time_matrix is not None else dists_km[cand]).astype(np.float64))

    prob = 1.0 - complement
    accessible = rt_demand * prob

    # Plain 3SFCA index (Wan, Zou & Sternberg 2012) from the same decay
    # weights: demand split by selection weight W_hf = P_hf / sum_g P_hg,
    # R_f = C_f / sum_h W_hf P_hf D_h, index_h = sum_f W_hf P_hf R_f.
    # Uncapped (can exceed 1); kept as supply credited to the hexagon so it
    # aggregates by summation.
    sfca3_supply = np.zeros(n, dtype=np.float64)
    if ent_hex:
        e_h, e_f, e_p = np.concatenate(ent_hex), np.concatenate(ent_fac), np.concatenate(ent_p)
        row_p = np.bincount(e_h, weights=e_p, minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(row_p[e_h] > 0, e_p / row_p[e_h], 0.0) * e_p
            q = np.bincount(e_f, weights=rt_demand[e_h] * eff, minlength=capacities.size)
            r_f = np.where(q > 0, capacities / q, 0.0)
        sfca3_index = np.bincount(e_h, weights=r_f[e_f] * eff, minlength=n)
        sfca3_supply = sfca3_index * rt_demand
    else:
        sfca3_index = np.zeros(n, dtype=np.float64)

    # ------------------------------------------------------------------
    # Pass 2 — allocation
    # ------------------------------------------------------------------
    facility_demand = None  # Q_f, from the capped 3SFCA rule only
    if allocation == "nearest_first":
        total_allocated, caps = _allocate_nearest_first(
            n, rt_demand, capacities, linac_locations,
            ent_ring, ent_fac, ent_hex, ent_p, model,
        )
    elif model == "uniform":
        # Every facility reaches every hex with P = 1, so both the balanced
        # and the optimal rule reduce to a closed form: total capacity is
        # shared across all demand.
        total_cap = float(capacities.sum())
        total_dem = float(rt_demand.sum())
        frac = min(1.0, total_cap / total_dem) if total_dem > 0 else 0.0
        total_allocated = rt_demand * frac
        caps = capacities * (1.0 - min(1.0, total_dem / total_cap)) if total_cap > 0 else capacities.copy()
    else:
        a_hex = np.concatenate(ent_hex) if ent_hex else np.zeros(0, dtype=np.int32)
        a_fac = np.concatenate(ent_fac) if ent_fac else np.zeros(0, dtype=np.int32)
        a_p = np.concatenate(ent_p) if ent_p else np.zeros(0, dtype=np.float64)
        if allocation == "proportional":
            total_allocated, caps, facility_demand = _allocate_proportional(
                n, accessible, capacities, a_hex, a_fac, a_p, size_weighting,
            )
        else:
            a_dist = np.concatenate(ent_dist) if ent_dist else np.zeros(0, dtype=np.float64)
            total_allocated, caps = _allocate_optimal(
                n, rt_demand, accessible, capacities, a_hex, a_fac, a_p, a_dist,
            )

    # --- assemble results ---
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
    g["sfca3_ratio"] = sfca3_index.astype(np.float32)
    g["sfca3_supply"] = sfca3_supply.astype(np.float32)

    total_pop = float(np.nansum(pop))
    total_rt_demand = float(np.nansum(rt_demand))
    total_rt_treated = float(np.nansum(rt_treated))
    total_machines = float(raw_weights.sum())
    total_capacity = total_machines * capacity_per_machine_per_year
    geo_accessible = float(np.nansum(accessible))

    stats = {
        "n_facilities": len(linac_locations),
        "total_machines": total_machines,
        "capacity_per_machine_per_year": capacity_per_machine_per_year,
        "total_national_capacity": total_capacity,
        "total_rt_demand": total_rt_demand,
        "total_rt_treated": total_rt_treated,
        "total_rt_untreated": total_rt_demand - total_rt_treated,
        "unused_capacity": max(total_capacity - total_rt_treated, 0.0),
        "model": model,
        "allocation": allocation,
        "size_weighting": bool(size_weighting),
        "lambda_km": lambda_km,
        "cutoff_km": cutoff_km,
        "max_distance_km": max_distance_km,
        "shape": weibull_k,
        "total_population": total_pop,
        "pop_with_access": float(np.nansum(prob * pop)),
        "mean_access_probability": float(np.nansum(prob * pop)) / total_pop if total_pop > 0 else 0.0,
        "geo_accessible_demand": geo_accessible,
        "demand_weighted_access": geo_accessible / total_rt_demand if total_rt_demand > 0 else 0.0,
        "mean_capacity_limited_probability": total_rt_treated / total_rt_demand if total_rt_demand > 0 else 0.0,
        "n_hexagons": n,
        "using_travel_time": travel_time_matrix is not None,
        "demand_fallback": demand_fallback,
        # per facility, in the order of linac_locations
        "facility_capacity": np.asarray(capacities, dtype=np.float64).tolist(),
        "facility_treated": np.maximum(np.asarray(capacities) - np.asarray(caps), 0.0).tolist(),
        "facility_demand": None if facility_demand is None else np.asarray(facility_demand).tolist(),
    }
    return g, stats


FRACTION_COLUMNS = ("rt_fractions_demand", "rt_fractions_treated", "rt_fractions_untreated")


def attach_fraction_metrics(
    gdf: gpd.GeoDataFrame,
    stats: Dict,
    fractions_demand: np.ndarray,
    fractions_per_linac: float,
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """Add workload (fraction) columns and totals to an accessibility result.

    ``fractions_demand`` is the annual number of treatment fractions each
    hexagon's RT demand would require (Σ_i demand_{h,i} × fractions per course_i).
    Treated fractions follow the patient allocation at each hexagon's own
    mean course length, so ``rt_fractions_treated / rt_fractions_demand``
    equals the patient access ratio wherever the cancer mix is uniform.

    Stats gain ``total_fractions_demand``, ``total_fractions_treated``,
    ``total_fractions_untreated``, ``fractions_per_linac``,
    ``fractions_capacity`` (machines × fractions per linac),
    ``mean_fractions_per_course`` (demand-weighted) and
    ``linac_equivalents_deficit`` (= untreated fractions / fractions per linac:
    the machines, at benchmark throughput and assuming they could be reached,
    that the unmet workload represents).
    """
    g = gdf
    dem = g["rt_demand"].to_numpy(np.float64)
    tr = g["rt_treated"].to_numpy(np.float64)
    fd = np.where(np.asarray(fractions_demand, dtype=np.float64) > 0, fractions_demand, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_course = np.where(dem > 0, fd / dem, 0.0)
    ft = tr * per_course
    g["rt_fractions_demand"] = fd.astype(np.float32)
    g["rt_fractions_treated"] = ft.astype(np.float32)
    g["rt_fractions_untreated"] = (fd - ft).astype(np.float32)

    total_fd = float(fd.sum())
    total_ft = float(ft.sum())
    total_dem = float(dem.sum())
    fpl = float(fractions_per_linac) if fractions_per_linac and fractions_per_linac > 0 else None
    stats.update({
        "total_fractions_demand": total_fd,
        "total_fractions_treated": total_ft,
        "total_fractions_untreated": total_fd - total_ft,
        "mean_fractions_per_course": total_fd / total_dem if total_dem > 0 else 0.0,
        "fractions_per_linac": fpl,
        "fractions_capacity": (stats.get("total_machines", 0.0) * fpl) if fpl else None,
        "linac_equivalents_deficit": ((total_fd - total_ft) / fpl) if fpl else None,
        "linac_equivalents_demand": (total_fd / fpl) if fpl else None,
    })
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
    _agg_spec = dict(
        population=("population", "sum"),
        rt_demand=("rt_demand", "sum"),
        rt_treated=("rt_treated", "sum"),
        rt_untreated=("rt_untreated", "sum"),
        pop_with_access=("pop_with_access", "sum"),
        _w=("_w", "sum"),
        _wd_km=("_wd_km", "sum"),
        _w_km=("_w_km", "sum"),
        _wp=("_wp", "sum"),
    )
    for c in FRACTION_COLUMNS + ("sfca3_supply",):
        if c in df.columns:
            _agg_spec[c] = (c, "sum")
    out = grp.agg(**_agg_spec).reset_index().rename(columns={"h3_parent": "h3"})

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
    if "sfca3_supply" in out.columns:
        out["sfca3_ratio"] = np.where(dem > 0, out["sfca3_supply"] / np.maximum(dem, 1e-9), 0.0).astype(np.float32)

    # Carry through country attribution if present (mode of children)
    if "country" in df.columns:
        country_mode = grp["country"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
        out = out.merge(country_mode.reset_index().rename(columns={"h3_parent": "h3"}), on="h3", how="left")

    out["geometry"] = out["h3"].apply(
        lambda h: _Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h)])
    )

    # Re-derive parent-cell centroids (dropped by the groupby) so downstream
    # consumers — notably the machine-planning optimiser — can rely on them
    # being present on every accessibility gdf, aggregated or not.
    _centroids = out["h3"].apply(lambda h: h3.cell_to_latlng(h))
    out["centroid_lat"] = _centroids.apply(lambda c: c[0])
    out["centroid_lon"] = _centroids.apply(lambda c: c[1])

    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
