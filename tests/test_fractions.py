"""Tests for the workload (fraction) metrics and the fractions-per-course data."""

from __future__ import annotations

import numpy as np
import pytest

h3 = pytest.importorskip("h3")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon

from analysis.accessibility import (
    compute_accessibility, attach_fraction_metrics, aggregate_access_gdf, FRACTION_COLUMNS,
)
from data.cancer import (
    get_fractions_per_course, fractions_per_course_map, effective_capacity_per_linac,
    get_optimal_rt_fractions, list_rtu_sets, DEFAULT_FRACTIONS_PER_COURSE,
)


def _grid(res=6, k=6):
    center = h3.latlng_to_cell(51.5, -0.12, res)
    cells = list(h3.grid_disk(center, k))
    geoms = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame({"h3": cells, "population": np.full(len(cells), 1000.0)},
                            geometry=geoms, crs="EPSG:4326")


LINACS = [(51.5, -0.12, 2.0), (51.6, 0.05, 1.0)]


def test_fraction_table_is_populated_from_wong_2016():
    fr, missing, total = get_fractions_per_course()
    assert total >= 30
    assert missing <= 2  # Liver and NMSC have no RT indication
    assert fr["breast"] == pytest.approx(16.4)
    assert fr["prostate"] == pytest.approx(28.1)
    assert fr["braincns"] == pytest.approx(29.1)


def test_both_rtu_sets_complete_and_2012_overall_matches_paper():
    ids = {s["set_id"] for s in list_rtu_sets()}
    assert {"delaney_2005", "delaney_2005_corrected", "barton_2014"} <= ids
    f = get_optimal_rt_fractions("barton_2014")
    assert f["Breast"] == pytest.approx(0.87)
    assert f["All cancers excl. NMSC"] == pytest.approx(0.484)


def test_site_override_and_factor():
    m = fractions_per_course_map((("Breast", 5.0),), factor=0.5)
    assert m["breast"] == pytest.approx(2.5)
    assert m["prostate"] == pytest.approx(28.1 * 0.5)


def test_effective_capacity_reproduces_450_at_benchmark_mix():
    # 8,730 fractions at the 19.4 overall mean gives exactly 450.
    r = effective_capacity_per_linac("GBR", 8730.0, "barton_2014", 1.0)
    assert 400 < r["patients_per_linac"] < 520
    assert r["demand_share_with_site_data"] == pytest.approx(1.0)
    half = effective_capacity_per_linac("GBR", 8730.0, "barton_2014", 0.5)
    assert half["patients_per_linac"] == pytest.approx(2 * r["patients_per_linac"], rel=1e-6)


def test_attach_fraction_metrics_consistency_and_aggregation():
    gdf = _grid()
    demand = np.linspace(5, 50, len(gdf))
    out, s = compute_accessibility(gdf, LINACS, model="step", max_distance_km=25,
                                   capacity_per_machine_per_year=200, demand=demand, h3_resolution=6)
    fractions = demand * 19.4
    out, s = attach_fraction_metrics(out, s, fractions, fractions_per_linac=8730.0)
    for c in FRACTION_COLUMNS:
        assert c in out.columns
    # Uniform mix: fraction access ratio equals patient access ratio everywhere
    fd = out["rt_fractions_demand"].to_numpy(np.float64)
    ft = out["rt_fractions_treated"].to_numpy(np.float64)
    ratio_f = np.where(fd > 0, ft / fd, 0)
    np.testing.assert_allclose(ratio_f, out["capacity_limited_probability"].to_numpy(np.float64), atol=1e-5)
    assert s["total_fractions_demand"] == pytest.approx(fractions.sum(), rel=1e-6)
    assert s["mean_fractions_per_course"] == pytest.approx(19.4, rel=1e-6)
    assert s["linac_equivalents_deficit"] == pytest.approx(s["total_fractions_untreated"] / 8730.0)
    assert s["fractions_capacity"] == pytest.approx(3.0 * 8730.0)
    # Aggregation sums the fraction columns
    agg = aggregate_access_gdf(out, 4)
    for c in FRACTION_COLUMNS:
        assert agg[c].sum() == pytest.approx(out[c].sum(), rel=1e-4), c


def test_attach_handles_zero_demand_hexes():
    gdf = _grid()
    demand = np.zeros(len(gdf)); demand[0] = 10.0
    out, s = compute_accessibility(gdf, LINACS, model="uniform", demand=demand, h3_resolution=6)
    out, s = attach_fraction_metrics(out, s, demand * DEFAULT_FRACTIONS_PER_COURSE, 8730.0)
    assert np.isfinite(out["rt_fractions_treated"]).all()
    assert s["total_fractions_treated"] == pytest.approx(10.0 * DEFAULT_FRACTIONS_PER_COURSE)
