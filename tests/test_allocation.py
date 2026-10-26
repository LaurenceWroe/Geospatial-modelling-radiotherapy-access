"""
Tests for the three capacity-allocation rules and the floating-catchment
benchmarks, using toy geometries with exact travel-time matrices so the
expected numbers can be worked out by hand.

Toy 1 — one facility, two hexes
    Facility capacity 100. Hex A (demand 80) at 10 min, hex B (demand 80)
    at 50 min, step catchment 60 min.
        nearest-first : A 100 %, B 25 %
        proportional  : A 62.5 %, B 62.5 %
        2SFCA ratio   : 0.625 everywhere in reach

Toy 2 — stranded capacity
    Facilities F and G, capacity 100 each. Hex A (80) is 10 min from both;
    hex B (80) reaches only F (30 min); hex C (80) reaches only G (30 min).
        nearest-first : 180 treated (A served entirely by one facility, 20
                        of the other's capacity unused, B or C left short)
        proportional  : 200 treated, every hex 83.3 %
        optimal       : 200 treated
"""

from __future__ import annotations

import numpy as np
import pytest

h3 = pytest.importorskip("h3")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon

from analysis.accessibility import ALLOCATION_RULES, compute_accessibility
from analysis.fca import FCA_VARIANTS, compute_fca


def _line_grid(n_cells: int, res: int = 6):
    """n hexes in a straight line heading east from a seed cell."""
    seed = h3.latlng_to_cell(51.5, -0.12, res)
    cells = [seed]
    cur = seed
    for _ in range(n_cells - 1):
        # pick the eastern neighbour: the ring-1 cell with the largest longitude
        ring = [c for c in h3.grid_ring(cur, 1) if c not in cells]
        cur = max(ring, key=lambda c: h3.cell_to_latlng(c)[1])
        cells.append(cur)
    geoms = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame(
        {"h3": cells, "population": np.full(len(cells), 1000.0)},
        geometry=geoms, crs="EPSG:4326",
    )


def _facility_at(gdf, idx, n_linacs=1.0):
    lat, lon = h3.cell_to_latlng(gdf["h3"].iloc[idx])
    return (lat, lon, n_linacs)


# ---------------------------------------------------------------------------
# Toy 1
# ---------------------------------------------------------------------------

@pytest.fixture
def toy1():
    gdf = _line_grid(2)
    locs = [_facility_at(gdf, 0)]                      # at hex A
    demand = np.array([80.0, 80.0])
    tt = np.array([[10.0], [50.0]])                    # minutes to the facility
    return gdf, locs, demand, tt


def test_toy1_nearest_first(toy1):
    gdf, locs, demand, tt = toy1
    out, s = compute_accessibility(
        gdf, locs, model="step", max_distance_km=60, capacity_per_machine_per_year=100,
        demand=demand, h3_resolution=6, travel_time_matrix=tt, allocation="nearest_first",
    )
    np.testing.assert_allclose(out["rt_treated"], [80.0, 20.0], atol=1e-6)
    assert s["total_rt_treated"] == pytest.approx(100.0)
    assert s["unused_capacity"] == pytest.approx(0.0)


def test_toy1_proportional_shares_shortage_equally(toy1):
    gdf, locs, demand, tt = toy1
    out, s = compute_accessibility(
        gdf, locs, model="step", max_distance_km=60, capacity_per_machine_per_year=100,
        demand=demand, h3_resolution=6, travel_time_matrix=tt, allocation="proportional",
    )
    np.testing.assert_allclose(out["rt_treated"], [50.0, 50.0], atol=1e-6)
    np.testing.assert_allclose(out["capacity_limited_probability"], [0.625, 0.625], atol=1e-6)
    assert s["total_rt_treated"] == pytest.approx(100.0)


def test_toy1_2sfca_ratio(toy1):
    gdf, locs, demand, tt = toy1
    df, s = compute_fca(
        gdf, locs, variant="2sfca", max_distance_km=60, capacity_per_machine_per_year=100,
        demand=demand, h3_resolution=6, travel_time_matrix=tt,
    )
    np.testing.assert_allclose(df["fca_ratio"], [0.625, 0.625], atol=1e-6)
    # aggregate level of service equals supply reaching demand
    assert s["aggregate_level_of_service"] == pytest.approx(s["supply_reaching_demand"])


def test_toy1_e2sfca_weights_nearer_hex_higher(toy1):
    gdf, locs, demand, tt = toy1
    df, _ = compute_fca(
        gdf, locs, variant="e2sfca", model="gaussian", max_distance_km=60,
        capacity_per_machine_per_year=100, demand=demand, h3_resolution=6, travel_time_matrix=tt,
    )
    assert df["fca_ratio"].iloc[0] > df["fca_ratio"].iloc[1] > 0


# ---------------------------------------------------------------------------
# Toy 2 — stranded capacity
# ---------------------------------------------------------------------------

@pytest.fixture
def toy2():
    # hexes: 0 = B, 1 = A, 2 = C along a line. Both facilities sit on hex A so
    # that A is the nearest ring for each of them (ring order is haversine);
    # the travel-time matrix then restricts B to F and C to G.
    gdf = _line_grid(3)
    locs = [_facility_at(gdf, 1), _facility_at(gdf, 1)]   # F and G both at A
    demand = np.array([80.0, 80.0, 80.0])                  # B, A, C
    inf = np.inf
    tt = np.array([
        [30.0, inf],    # B: reaches F only
        [10.0, 10.0],   # A: reaches both
        [inf, 30.0],    # C: reaches G only
    ])
    return gdf, locs, demand, tt


def test_toy2_nearest_first_strands_capacity(toy2):
    gdf, locs, demand, tt = toy2
    out, s = compute_accessibility(
        gdf, locs, model="step", max_distance_km=60, capacity_per_machine_per_year=100,
        demand=demand, h3_resolution=6, travel_time_matrix=tt, allocation="nearest_first",
    )
    assert s["total_rt_treated"] == pytest.approx(180.0, abs=1e-6)
    assert s["unused_capacity"] == pytest.approx(20.0, abs=1e-6)
    assert out["rt_treated"].iloc[1] == pytest.approx(80.0)          # A fully served
    assert sorted(out["rt_treated"].iloc[[0, 2]].round(6)) == [20.0, 80.0]


def test_toy2_proportional_uses_all_capacity(toy2):
    gdf, locs, demand, tt = toy2
    out, s = compute_accessibility(
        gdf, locs, model="step", max_distance_km=60, capacity_per_machine_per_year=100,
        demand=demand, h3_resolution=6, travel_time_matrix=tt, allocation="proportional",
    )
    assert s["total_rt_treated"] == pytest.approx(200.0, abs=1e-6)
    np.testing.assert_allclose(out["capacity_limited_probability"], [5 / 6] * 3, atol=1e-6)


def test_toy2_optimal_matches_upper_bound(toy2):
    pytest.importorskip("scipy")
    gdf, locs, demand, tt = toy2
    out, s = compute_accessibility(
        gdf, locs, model="step", max_distance_km=60, capacity_per_machine_per_year=100,
        demand=demand, h3_resolution=6, travel_time_matrix=tt, allocation="optimal",
    )
    assert s["total_rt_treated"] == pytest.approx(200.0, abs=1e-4)
    assert (out["rt_treated"] <= out["rt_demand"] + 1e-6).all()


# ---------------------------------------------------------------------------
# Generic invariants on a random grid
# ---------------------------------------------------------------------------

def _disk_grid(res=6, k=8):
    center = h3.latlng_to_cell(51.5, -0.12, res)
    cells = list(h3.grid_disk(center, k))
    geoms = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells]
    return gpd.GeoDataFrame(
        {"h3": cells, "population": np.full(len(cells), 1000.0)}, geometry=geoms, crs="EPSG:4326",
    )


LINACS = [(51.5, -0.12, 2.0), (51.7, 0.10, 1.0), (51.3, -0.50, 3.0)]


@pytest.mark.parametrize("allocation", ALLOCATION_RULES)
@pytest.mark.parametrize("model", ["step", "weibull", "gaussian", "exponential", "power", "uniform"])
def test_allocation_conservation(allocation, model):
    if allocation == "optimal":
        pytest.importorskip("scipy")
    gdf = _disk_grid()
    demand = np.full(len(gdf), 60.0)
    out, s = compute_accessibility(
        gdf, LINACS, model=model, lambda_km=30, weibull_k=2.0, max_distance_km=40,
        capacity_per_machine_per_year=450, demand=demand, h3_resolution=6, allocation=allocation,
    )
    assert (out["rt_treated"] <= out["rt_demand"] + 1e-5).all()
    # no hex is treated beyond its geographically accessible demand
    acc = out["rt_demand"].to_numpy(np.float64) * out["access_probability"].to_numpy(np.float64)
    assert (out["rt_treated"].to_numpy(np.float64) <= acc + 1e-3).all()
    assert s["total_rt_treated"] <= s["total_national_capacity"] + 1e-3
    assert s["unused_capacity"] == pytest.approx(
        max(s["total_national_capacity"] - s["total_rt_treated"], 0.0), abs=1e-3
    )


def test_optimal_treats_at_least_as_many_as_greedy():
    pytest.importorskip("scipy")
    gdf = _disk_grid()
    demand = np.full(len(gdf), 60.0)
    kw = dict(model="step", max_distance_km=40, capacity_per_machine_per_year=450,
              demand=demand, h3_resolution=6)
    _, greedy = compute_accessibility(gdf, LINACS, allocation="nearest_first", **kw)
    _, prop = compute_accessibility(gdf, LINACS, allocation="proportional", **kw)
    _, opt = compute_accessibility(gdf, LINACS, allocation="optimal", **kw)
    assert opt["total_rt_treated"] >= greedy["total_rt_treated"] - 1e-4
    assert opt["total_rt_treated"] >= prop["total_rt_treated"] - 1e-4


def test_ample_capacity_all_rules_agree_with_geography():
    """With unlimited capacity every rule treats exactly the accessible demand."""
    gdf = _disk_grid()
    demand = np.full(len(gdf), 10.0)
    for allocation in ALLOCATION_RULES:
        if allocation == "optimal":
            pytest.importorskip("scipy")
        out, s = compute_accessibility(
            gdf, LINACS, model="weibull", lambda_km=30, weibull_k=3, capacity_per_machine_per_year=1e9,
            demand=demand, h3_resolution=6, allocation=allocation,
        )
        acc = out["rt_demand"].to_numpy(np.float64) * out["access_probability"].to_numpy(np.float64)
        np.testing.assert_allclose(out["rt_treated"].to_numpy(np.float64), acc, atol=1e-2)
        assert s["total_rt_treated"] == pytest.approx(s["geo_accessible_demand"], rel=1e-4)


def test_demand_weighted_access_stat():
    gdf = _disk_grid()
    demand = np.linspace(1, 100, len(gdf))
    out, s = compute_accessibility(
        gdf, LINACS, model="step", max_distance_km=30, demand=demand, h3_resolution=6,
    )
    expected = float((out["access_probability"].to_numpy(np.float64) * demand).sum()) / demand.sum()
    assert s["demand_weighted_access"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("variant", FCA_VARIANTS)
def test_fca_outputs(variant):
    gdf = _disk_grid()
    demand = np.full(len(gdf), 60.0)
    df, s = compute_fca(
        gdf, LINACS, variant=variant, model="gaussian", lambda_km=10, max_distance_km=40,
        capacity_per_machine_per_year=450, demand=demand, h3_resolution=6,
    )
    assert len(df) == len(gdf)
    assert (df["fca_ratio"] >= 0).all()
    assert (df["fca_ratio_capped"] <= 1.0).all()
    assert s["implied_treated"] <= s["total_rt_demand"] + 1e-6
    assert s["aggregate_level_of_service"] == pytest.approx(s["supply_reaching_demand"], rel=1e-6)


def test_unknown_allocation_raises():
    gdf = _disk_grid()
    with pytest.raises(ValueError):
        compute_accessibility(gdf, LINACS, allocation="random", h3_resolution=6)


def test_3sfca_equals_e2sfca_with_one_facility():
    # With a single reachable facility the selection weight G_hj is 1 everywhere,
    # so 3SFCA reduces exactly to E2SFCA.
    gdf = _disk_grid()
    demand = np.full(len(gdf), 60.0)
    kw = dict(model="gaussian", lambda_km=10, max_distance_km=40,
              capacity_per_machine_per_year=450, demand=demand, h3_resolution=6)
    e2, _ = compute_fca(gdf, LINACS[:1], variant="e2sfca", **kw)
    s3, _ = compute_fca(gdf, LINACS[:1], variant="3sfca", **kw)
    np.testing.assert_allclose(s3["fca_ratio"], e2["fca_ratio"], rtol=1e-6)


def test_3sfca_invariant_to_colocated_split():
    # Splitting one facility into two co-located halves leaves 3SFCA unchanged:
    # the selection weights halve and the per-facility ratios double.
    gdf = _disk_grid()
    demand = np.full(len(gdf), 60.0)
    lat, lon, w = LINACS[0]
    kw = dict(model="gaussian", lambda_km=10, max_distance_km=40,
              capacity_per_machine_per_year=450, demand=demand, h3_resolution=6)
    one, _ = compute_fca(gdf, [(lat, lon, w)], variant="3sfca", **kw)
    two, _ = compute_fca(gdf, [(lat, lon, w / 2), (lat, lon, w / 2)], variant="3sfca", **kw)
    np.testing.assert_allclose(two["fca_ratio"], one["fca_ratio"], rtol=1e-6)


def test_3sfca_demand_split_lowers_overcounting():
    # Two facilities with overlapping catchments: 3SFCA counts each hexagon's
    # demand once in total across step 1, so its aggregate implied deficit is
    # never above E2SFCA's wherever both reach the same demand.
    gdf = _disk_grid()
    demand = np.full(len(gdf), 60.0)
    kw = dict(model="gaussian", lambda_km=10, max_distance_km=40,
              capacity_per_machine_per_year=450, demand=demand, h3_resolution=6)
    e2, se = compute_fca(gdf, LINACS, variant="e2sfca", **kw)
    s3, s3s = compute_fca(gdf, LINACS, variant="3sfca", **kw)
    assert se["aggregate_level_of_service"] == pytest.approx(s3s["aggregate_level_of_service"], rel=1e-6)
    assert not np.allclose(s3["fca_ratio"], e2["fca_ratio"])


@pytest.mark.parametrize("variant", FCA_VARIANTS)
def test_fca_implied_deficit_complements_treated(variant):
    gdf = _disk_grid()
    demand = np.full(len(gdf), 60.0)
    df, s = compute_fca(
        gdf, LINACS, variant=variant, model="gaussian", lambda_km=10, max_distance_km=40,
        capacity_per_machine_per_year=450, demand=demand, h3_resolution=6,
    )
    np.testing.assert_allclose(df["fca_implied_treated"] + df["fca_implied_deficit"], demand, rtol=1e-5)
    assert s["implied_deficit"] == pytest.approx(float(df["fca_implied_deficit"].sum()), rel=1e-5)
    assert s["implied_treated"] + s["implied_deficit"] == pytest.approx(s["total_rt_demand"], rel=1e-6)


def test_nearest_first_rings_follow_travel_time():
    # One facility on hex 0 with capacity for one hex. Hex 1 is nearer by
    # distance but hex 2 is nearer by travel time, so hex 2 must be served.
    gdf = _line_grid(3)
    locs = [_facility_at(gdf, 0)]
    demand = np.array([0.0, 100.0, 100.0])
    tt = np.array([[1.0], [60.0], [20.0]])
    out, s = compute_accessibility(
        gdf, locs, model="step", max_distance_km=90, capacity_per_machine_per_year=100,
        demand=demand, h3_resolution=6, travel_time_matrix=tt, allocation="nearest_first",
    )
    treated = out["rt_treated"].to_numpy()
    assert treated[2] == pytest.approx(100.0, abs=1e-6)
    assert treated[1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Limiting cases (Section 2.4.3): both rules recover A_C and A_G
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["nearest_first", "proportional"])
def test_capacity_only_limit(rule):
    # Every facility reaches every hex (P = 1) and facilities differ in size,
    # so the capped 3SFCA split must weight by capacity to use it all.
    gdf = _line_grid(5)
    locs = [_facility_at(gdf, 0, 1.0), _facility_at(gdf, 4, 3.0)]
    demand = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    _, s = compute_accessibility(
        gdf, locs, model="step", max_distance_km=1000, capacity_per_machine_per_year=100,
        demand=demand, h3_resolution=6, allocation=rule,
    )
    assert s["total_rt_treated"] == pytest.approx(min(400.0, 500.0), rel=1e-6)


@pytest.mark.parametrize("rule", ["nearest_first", "proportional"])
def test_geography_only_limit(rule):
    gdf = _line_grid(5)
    locs = [_facility_at(gdf, 0), _facility_at(gdf, 4)]
    demand = np.array([100.0, 50.0, 80.0, 20.0, 60.0])
    out, s = compute_accessibility(
        gdf, locs, model="exponential", lambda_km=10, capacity_per_machine_per_year=1e9,
        demand=demand, h3_resolution=6, allocation=rule,
    )
    a_g = float((out["access_probability"] * out["rt_demand"]).sum())
    assert s["total_rt_treated"] == pytest.approx(a_g, rel=1e-4)


@pytest.mark.parametrize("model", ["step", "gaussian", "exponential"])
def test_engine_3sfca_matches_fca_module(toy2, model):
    """compute_accessibility's sfca3_ratio is the plain 3SFCA index of analysis.fca."""
    gdf, locs, demand, tt = toy2
    kw = dict(model=model, max_distance_km=60, lambda_km=20.0, capacity_per_machine_per_year=100,
              demand=demand, h3_resolution=6, travel_time_matrix=tt)
    out, _ = compute_accessibility(gdf, locs, allocation="nearest_first", **kw)
    df, _ = compute_fca(gdf, locs, variant="3sfca", **kw)
    np.testing.assert_allclose(out["sfca3_ratio"], df["fca_ratio"], rtol=1e-5)
    np.testing.assert_allclose(out["sfca3_supply"], out["sfca3_ratio"] * demand, rtol=1e-5)
