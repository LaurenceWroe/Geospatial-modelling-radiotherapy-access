"""
Floating-catchment-area benchmarks for comparison with the RadMaps allocation.

These are the conventional *potential accessibility* indices from the spatial
accessibility literature. Unlike the RadMaps allocation they do not cap
treatment at capacity or at demand: the output is a supply-to-demand ratio per
hexagon that can exceed 1, and overlapping catchments count the same demand
(and the same supply) more than once.

Variants
--------
``2sfca``   Luo & Wang (2003). Binary catchment of radius ``threshold``.
            Step 1: R_j = S_j / Σ_{h in catchment j} D_h
            Step 2: A_h = Σ_{j: h in catchment j} R_j
``e2sfca``  Luo & Qi (2009) / Wang (2012) generalised form. Same two steps
            with a decay weight w(x) in place of the binary catchment, using
            whichever decay model the caller selects (Gaussian is the
            classic E2SFCA kernel).
            Step 1: R_j = S_j / Σ_h D_h w(x_hj)
            Step 2: A_h = Σ_j R_j w(x_hj)
``3sfca``   Wan, Zou & Sternberg (2012). Adds a selection weight
            G_hj = w(x_hj) / Σ_k w(x_hk) so that each hexagon's demand is
            split across the facilities it can reach rather than counted in
            full against every one of them (the demand side of the balanced
            FCA; supply is still not normalised).
            Step 1: R_j = S_j / Σ_h D_h G_hj w(x_hj)
            Step 2: A_h = Σ_j G_hj R_j w(x_hj)
``gravity`` Joseph & Bantock (1982) gravity model: ``e2sfca`` with a
            power-law impedance.

Interpretation
--------------
``fca_ratio`` is capacity (patients/yr) available per patient of demand within
reach. ``fca_ratio_capped`` = min(1, ratio) is the fraction of local demand the
index implies could be treated, which is the quantity comparable with the
RadMaps ``capacity_limited_probability``. ``fca_implied_treated`` and
``fca_implied_deficit`` multiply that fraction (and its complement) by the
hexagon's demand. Because each facility's ratio is re-summed at every hexagon
it reaches, the national sum of implied treated patients is *not* bounded by
total capacity — see Paez, Higgins & Vivona (2019).

Implementation is sparse: only hexagon–facility pairs with w(x) > 0 are kept.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from analysis.accessibility import (
    _REFERENCE_CAPACITY,
    hex_centroids,
    iter_facility_access,
    snap_linacs,
)
from analysis.decay import default_cutoff

FCA_VARIANTS = ("2sfca", "e2sfca", "3sfca", "gravity")

FCA_LABELS = {
    "2sfca": "2SFCA (Luo & Wang 2003)",
    "e2sfca": "E2SFCA / generalised 2SFCA",
    "3sfca": "3SFCA (Wan et al. 2012)",
    "gravity": "Gravity model (power-law impedance)",
}

_W_MIN = 1e-9


def compute_fca(
    gdf: gpd.GeoDataFrame,
    linac_locations: List[Tuple[float, float, float]],
    *,
    variant: str = "e2sfca",
    model: str = "gaussian",
    lambda_km: float = 30.0,
    weibull_k: float = 2.0,
    max_distance_km: float = 120.0,
    cutoff_km: Optional[float] = None,
    capacity_per_machine_per_year: float = _REFERENCE_CAPACITY,
    demand: Optional[np.ndarray] = None,
    snap_linacs_to_hex: bool = False,
    h3_resolution: int = 8,
    travel_time_matrix: Optional[np.ndarray] = None,
    gravity_beta: float = 1.5,
) -> Tuple[pd.DataFrame, Dict]:
    """Return per-hexagon floating-catchment ratios and summary statistics.

    Parameters mirror :func:`analysis.accessibility.compute_accessibility`.
    For ``variant="2sfca"`` the decay is forced to a step at
    ``max_distance_km``; for ``variant="gravity"`` it is forced to a power law
    with exponent ``gravity_beta`` and plateau ``lambda_km``.
    """
    if variant not in FCA_VARIANTS:
        raise ValueError(f"Unknown FCA variant {variant!r}; expected one of {FCA_VARIANTS}")

    if variant == "2sfca":
        model, shape = "step", weibull_k
    elif variant == "gravity":
        model, shape = "power", gravity_beta
    else:
        shape = weibull_k
    if cutoff_km is None:
        cutoff_km = default_cutoff(model, lambda_km, max_distance_km)

    if snap_linacs_to_hex:
        linac_locations = snap_linacs(linac_locations, h3_resolution)

    lats, lons = hex_centroids(gdf)
    n = len(gdf)
    n_fac = len(linac_locations)
    pop = pd.to_numeric(gdf["population"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    pop = np.where(pop > 0, pop, 0.0)
    rt_demand = np.where(demand > 0, demand, 0.0).astype(np.float64) if demand is not None else pop.copy()

    raw_weights = np.array([w for _, _, w in linac_locations], dtype=np.float64)
    capacities = raw_weights * capacity_per_machine_per_year

    # Sparse weight matrix: one entry per (hex, facility) pair with w > 0.
    e_hex: List[np.ndarray] = []
    e_fac: List[np.ndarray] = []
    e_w: List[np.ndarray] = []
    for j, p, _d, _x in iter_facility_access(
        lats, lons, linac_locations, model=model, scale=lambda_km, shape=shape,
        threshold=max_distance_km, cutoff=cutoff_km, travel_time_matrix=travel_time_matrix,
    ):
        idx = np.flatnonzero(p > _W_MIN)
        if idx.size == 0:
            continue
        e_hex.append(idx.astype(np.int32))
        e_fac.append(np.full(idx.size, j, dtype=np.int32))
        e_w.append(p[idx].astype(np.float64))
    a_hex = np.concatenate(e_hex) if e_hex else np.zeros(0, dtype=np.int32)
    a_fac = np.concatenate(e_fac) if e_fac else np.zeros(0, dtype=np.int32)
    a_w = np.concatenate(e_w) if e_w else np.zeros(0, dtype=np.float64)

    if variant == "3sfca":
        row_sum = np.bincount(a_hex, weights=a_w, minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(row_sum[a_hex] > 0, a_w / row_sum[a_hex], 0.0)   # selection weight G_hj
        eff = g * a_w
    else:
        eff = a_w

    # Step 1: facility supply-to-demand ratio over its weighted catchment.
    demand_in_reach = np.bincount(a_fac, weights=rt_demand[a_hex] * eff, minlength=n_fac)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(demand_in_reach > 0, capacities / demand_in_reach, 0.0)

    # Step 2: sum each reachable facility's ratio at the hexagon.
    a = np.bincount(a_hex, weights=ratios[a_fac] * eff, minlength=n)

    capped = np.minimum(a, 1.0)
    out = pd.DataFrame({
        "h3": gdf["h3"].to_numpy(),
        "fca_ratio": a.astype(np.float32),
        "fca_ratio_capped": capped.astype(np.float32),
        "fca_implied_treated": (capped * rt_demand).astype(np.float32),
        "fca_implied_deficit": ((1.0 - capped) * rt_demand).astype(np.float32),
    })

    total_demand = float(rt_demand.sum())
    stats = {
        "variant": variant,
        "model": model,
        "lambda_km": lambda_km,
        "shape": shape,
        "max_distance_km": max_distance_km,
        "cutoff_km": cutoff_km,
        "total_rt_demand": total_demand,
        "total_national_capacity": float(capacities.sum()),
        "demand_weighted_ratio": float((a * rt_demand).sum()) / total_demand if total_demand > 0 else 0.0,
        "implied_treated": float((capped * rt_demand).sum()),
        "implied_treated_fraction": float((capped * rt_demand).sum()) / total_demand if total_demand > 0 else 0.0,
        "implied_deficit": float(((1.0 - capped) * rt_demand).sum()),
        "demand_with_ratio_ge_1": float(rt_demand[a >= 1.0].sum()),
        "demand_unreached": float(rt_demand[a <= 0.0].sum()),
        # Σ_h A_h D_h re-sums each facility's ratio across its catchment, so it
        # equals Σ_j S_j over facilities that reach any demand — the identity
        # that holds in aggregate despite the per-hex distortion.
        "aggregate_level_of_service": float((a * rt_demand).sum()),
        "supply_reaching_demand": float(capacities[demand_in_reach > 0].sum()),
        "n_hexagons": n,
        "n_pairs": int(a_hex.size),
        "using_travel_time": travel_time_matrix is not None,
    }
    return out, stats
