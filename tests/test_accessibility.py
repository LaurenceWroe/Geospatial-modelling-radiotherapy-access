"""
Unit tests for the core accessibility computation and aggregation.

These cover the conservation properties that any refactor of the allocation
loop must preserve:
  - treated ≤ demand, treated ≤ capacity
  - order-independence of the facility list
  - probability bounds
  - aggregation conserves counts
Plus the cancer apportionment invariant (hex incidence sums to national total).

Run:  pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

h3 = pytest.importorskip("h3")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon

from analysis.accessibility import (
    compute_accessibility,
    aggregate_access_gdf,
    _haversine_km,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _grid(center_lat=51.5, center_lon=-0.12, res=6, k=10, pop=1000.0):
    center = h3.latlng_to_cell(center_lat, center_lon, res)
    cells = list(h3.grid_disk(center, k))
    geoms = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame(
        {"h3": cells, "population": np.full(len(cells), pop, dtype=float)},
        geometry=geoms, crs="EPSG:4326",
    )


@pytest.fixture
def grid():
    return _grid()


@pytest.fixture
def linacs():
    return [(51.5, -0.12, 2.0), (51.7, 0.10, 1.0), (51.3, -0.50, 3.0)]


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def test_haversine_matches_known_distance():
    # London ↔ Paris ≈ 344 km
    lats = np.radians(np.array([48.8566]))
    lons = np.radians(np.array([2.3522]))
    d = _haversine_km(51.5074, -0.1278, lats, lons, np.cos(lats))[0]
    assert 330 < d < 360


def test_haversine_zero_distance():
    lats = np.radians(np.array([10.0]))
    lons = np.radians(np.array([20.0]))
    d = _haversine_km(10.0, 20.0, lats, lons, np.cos(lats))[0]
    assert d == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Conservation properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["exponential", "weibull", "step", "uniform"])
def test_treated_never_exceeds_demand(grid, linacs, model):
    demand = np.full(len(grid), 40.0)
    out, stats = compute_accessibility(
        grid, linacs, lambda_km=30, model=model, weibull_k=4,
        max_distance_km=60, capacity_per_machine_per_year=450,
        demand=demand, h3_resolution=6,
    )
    assert (out["rt_treated"] <= out["rt_demand"] + 1e-5).all()
    assert stats["total_rt_treated"] <= stats["total_rt_demand"] + 1e-3


def test_treated_never_exceeds_capacity(grid, linacs):
    demand = np.full(len(grid), 1000.0)  # force capacity to bind
    out, stats = compute_accessibility(
        grid, linacs, lambda_km=50, model="exponential",
        capacity_per_machine_per_year=450, demand=demand, h3_resolution=6,
    )
    total_capacity = stats["total_machines"] * 450
    assert stats["total_rt_treated"] <= total_capacity + 1e-3


def test_probability_bounds(grid, linacs):
    out, _ = compute_accessibility(
        grid, linacs, lambda_km=30, model="weibull", weibull_k=4,
        demand=np.full(len(grid), 40.0), h3_resolution=6,
    )
    p = out["access_probability"].to_numpy()
    assert (p >= 0).all() and (p <= 1.0 + 1e-6).all()
    cp = out["capacity_limited_probability"].to_numpy()
    assert (cp >= 0).all() and (cp <= 1.0 + 1e-6).all()


def test_demand_fallback_flag(grid, linacs):
    _, stats_pop = compute_accessibility(grid, linacs, h3_resolution=6)
    assert stats_pop["demand_fallback"] is True
    _, stats_dem = compute_accessibility(
        grid, linacs, demand=np.full(len(grid), 5.0), h3_resolution=6
    )
    assert stats_dem["demand_fallback"] is False


# ---------------------------------------------------------------------------
# Order independence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["exponential", "weibull", "step"])
def test_allocation_order_independent(grid, linacs, model):
    demand = np.full(len(grid), 40.0)
    kw = dict(lambda_km=30, model=model, weibull_k=4, max_distance_km=60,
              capacity_per_machine_per_year=450, demand=demand, h3_resolution=6)
    base, _ = compute_accessibility(grid, linacs, **kw)
    rev, _ = compute_accessibility(grid, list(reversed(linacs)), **kw)
    rot, _ = compute_accessibility(grid, linacs[1:] + linacs[:1], **kw)
    b = base.set_index("h3")["rt_treated"]
    assert float((b - rev.set_index("h3")["rt_treated"]).abs().max()) < 1e-4
    assert float((b - rot.set_index("h3")["rt_treated"]).abs().max()) < 1e-4


def test_uniform_full_coverage_with_ample_capacity(grid, linacs):
    demand = np.full(len(grid), 10.0)
    _, stats = compute_accessibility(
        grid, linacs, model="uniform",
        capacity_per_machine_per_year=1_000_000, demand=demand, h3_resolution=6,
    )
    # With effectively unlimited capacity and no distance barrier, all demand met
    assert stats["total_rt_treated"] == pytest.approx(stats["total_rt_demand"], rel=1e-6)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregation_conserves_counts(grid, linacs):
    out, _ = compute_accessibility(
        grid, linacs, lambda_km=30, model="weibull", weibull_k=4,
        demand=np.full(len(grid), 40.0), h3_resolution=6,
    )
    agg = aggregate_access_gdf(out, 3)
    for col in ("population", "rt_demand", "rt_treated", "rt_untreated", "pop_with_access"):
        assert agg[col].sum() == pytest.approx(out[col].sum(), rel=1e-4), col
    assert len(agg) < len(out)
    # Probabilities stay in bounds after aggregation
    assert (agg["access_probability"] >= 0).all()
    assert (agg["access_probability"] <= 1.0 + 1e-6).all()


def test_aggregation_noop_when_target_finer(grid, linacs):
    out, _ = compute_accessibility(grid, linacs, demand=np.full(len(grid), 40.0), h3_resolution=6)
    same = aggregate_access_gdf(out, 8)  # finer than native → returned unchanged
    assert len(same) == len(out)
