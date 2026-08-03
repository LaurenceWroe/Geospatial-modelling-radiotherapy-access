"""
Tests for analysis.convergence — the resolution-convergence helpers.

These avoid the TravelTime API: they exercise the demand-weighting invariant
(A_RM ≤ A_G) via the real accessibility engine on a synthetic grid, and the
no-facility guard on the public entry point.
"""
from __future__ import annotations

import numpy as np
import pytest

h3 = pytest.importorskip("h3")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon

from analysis.accessibility import compute_accessibility
from analysis.convergence import _metrics, _optimal_demand, run_convergence


def _grid(res=6, k=8, pop=1000.0):
    center = h3.latlng_to_cell(51.5, -0.12, res)
    cells = list(h3.grid_disk(center, k))
    geoms = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame(
        {"h3": cells, "population": np.full(len(cells), pop, dtype=float)},
        geometry=geoms, crs="EPSG:4326",
    )


def test_metrics_demand_weighted_arm_le_ag():
    """A_RM (demand-weighted, capacity-limited) must never exceed A_G (demand-weighted access)."""
    gdf = _grid()
    demand = np.full(len(gdf), 40.0)
    locs = [(51.5, -0.12, 2.0), (51.7, 0.1, 1.0)]
    for thr in (30, 60, 120):
        g, s = compute_accessibility(gdf, locs, model="step", max_distance_km=thr,
                                     capacity_per_machine_per_year=450, demand=demand,
                                     h3_resolution=6)
        a_c, a_g, a_rm = _metrics(g, s)
        assert 0.0 <= a_g <= 1.0 + 1e-9
        assert 0.0 <= a_rm <= 1.0 + 1e-9
        assert a_rm <= a_g + 1e-9, f"A_RM {a_rm} > A_G {a_g} at {thr} km"


def test_metrics_ag_is_demand_weighted():
    """A_G weights geographic access by demand, not population."""
    gdf = _grid()
    # uneven demand: concentrate demand where access is guaranteed (near the centre linac)
    demand = np.full(len(gdf), 1.0)
    g, s = compute_accessibility(gdf, [(51.5, -0.12, 5.0)], model="step", max_distance_km=60,
                                 capacity_per_machine_per_year=450, demand=demand, h3_resolution=6)
    _, a_g, _ = _metrics(g, s)
    manual = float((g["access_probability"] * g["rt_demand"]).sum() / g["rt_demand"].sum())
    assert a_g == pytest.approx(manual, rel=1e-6)


def test_optimal_demand_none_when_no_rt_columns():
    import pandas as pd
    df = pd.DataFrame({"population": [1, 2, 3]})
    assert _optimal_demand(df) is None


def test_run_convergence_raises_for_country_without_facilities():
    # Chad has GLOBOCAN data but no DIRAC facilities -> must raise cleanly, not crash.
    with pytest.raises(ValueError):
        run_convergence("Chad", "TCD", "", "")
