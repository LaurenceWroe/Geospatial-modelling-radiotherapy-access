"""
Unit tests for the data loaders: cancer apportionment, linac parsing,
and travel-time aggregation.

Run:  pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

h3 = pytest.importorskip("h3")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon

from data.cancer import apportion_cancer_to_h3, get_national_cases
from data.travel_time import aggregate_tt_matrix


def _grid(res=6, k=6, pops=None):
    center = h3.latlng_to_cell(51.5, -0.12, res)
    cells = list(h3.grid_disk(center, k))
    if pops is None:
        pops = np.arange(1, len(cells) + 1, dtype=float) * 100
    geoms = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame(
        {"h3": cells, "population": pops}, geometry=geoms, crs="EPSG:4326"
    )


# ---------------------------------------------------------------------------
# Cancer apportionment
# ---------------------------------------------------------------------------

def test_incidence_sums_to_national_total():
    """Apportioned hex incidence must sum back to the national case count."""
    gdf = _grid()
    cancers = ["Lung", "Breast"]
    national = get_national_cases("GBR", cancers)
    out = apportion_cancer_to_h3(gdf, "GBR", cancers, use_actual_rt=False)
    for c in cancers:
        col = f"{c}_incidence"
        assert out[col].sum() == pytest.approx(national[c], rel=1e-4), c


def test_incidence_proportional_to_population():
    """A hex with 2× the population should get 2× the incidence."""
    cells_pop = np.array([100.0, 200.0, 300.0])
    center = h3.latlng_to_cell(51.5, -0.12, 6)
    cells = list(h3.grid_disk(center, 1))[:3]
    geoms = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells]
    gdf = gpd.GeoDataFrame({"h3": cells, "population": cells_pop}, geometry=geoms, crs="EPSG:4326")
    out = apportion_cancer_to_h3(gdf, "GBR", ["Lung"], use_actual_rt=False)
    inc = out["Lung_incidence"].to_numpy()
    # ratios should match population ratios
    assert inc[1] / inc[0] == pytest.approx(2.0, rel=1e-3)
    assert inc[2] / inc[0] == pytest.approx(3.0, rel=1e-3)


def test_nan_national_cases_become_zero():
    """Vagina cancer has NaN for the World aggregate; must resolve to 0, not NaN."""
    cases = get_national_cases("WLD", ["Vagina"])
    assert np.isfinite(cases["Vagina"])
    assert cases["Vagina"] == 0.0


# ---------------------------------------------------------------------------
# Travel-time aggregation
# ---------------------------------------------------------------------------

def test_aggregate_tt_population_weighted_mean():
    """Two children of one parent → population-weighted mean travel time."""
    parent = h3.latlng_to_cell(51.5, -0.12, 3)
    children = h3.cell_to_children(parent, 5)[:2]
    # child0 pop 100 @ 10 min; child1 pop 300 @ 30 min → weighted mean 25 min
    matrix = np.array([[10.0], [30.0]], dtype=np.float32)
    pops = np.array([100.0, 300.0])
    result = aggregate_tt_matrix(matrix, list(children), pops, [parent], target_resolution=3)
    assert result[0, 0] == pytest.approx(25.0, rel=1e-4)


def test_aggregate_tt_unreachable_fill_biases_upward():
    """With unreachable fill, an unreachable child raises the parent's mean."""
    parent = h3.latlng_to_cell(51.5, -0.12, 3)
    children = h3.cell_to_children(parent, 5)[:2]
    matrix = np.array([[10.0], [np.inf]], dtype=np.float32)  # one reachable, one not
    pops = np.array([100.0, 100.0])
    # Without fill: only the reachable child counts → 10 min
    no_fill = aggregate_tt_matrix(matrix, list(children), pops, [parent], target_resolution=3)
    assert no_fill[0, 0] == pytest.approx(10.0, rel=1e-4)
    # With fill at 120 min: mean of 10 and 120 → 65 min
    with_fill = aggregate_tt_matrix(
        matrix, list(children), pops, [parent], target_resolution=3,
        unreachable_fill_min=120.0,
    )
    assert with_fill[0, 0] == pytest.approx(65.0, rel=1e-4)


def test_aggregate_tt_all_unreachable_stays_inf():
    parent = h3.latlng_to_cell(51.5, -0.12, 3)
    children = h3.cell_to_children(parent, 5)[:2]
    matrix = np.array([[np.inf], [np.inf]], dtype=np.float32)
    pops = np.array([100.0, 100.0])
    result = aggregate_tt_matrix(
        matrix, list(children), pops, [parent], target_resolution=3,
        unreachable_fill_min=120.0,
    )
    assert not np.isfinite(result[0, 0])
