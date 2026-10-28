"""
Tests for the dashboard layer (compute.py): access profiles, the distance proxy
and its inverse, mixed TravelTime / straight-line matrices, centre loads, the
facility planner, and the pre-computed World results.

Run:  pytest tests/ -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("h3")

from compute import (_regional_tt_matrix, access_profile, centre_loads, compute_members, minutes_to_proxy_km,
                     plan_centres, proxy_km_to_minutes)
from data.linacs import load_linacs_from_dirac_db
from data.population import load_population_at_resolution
from data.tt_store import load_store

_ROOT = Path(__file__).resolve().parent.parent
_WORLD = _ROOT / "cache" / "dashboard_world_countries.parquet"


def _main_config(df: pd.DataFrame, allocation: str = "proportional") -> pd.DataFrame:
    d = df[(df["rtu"] == "barton_2014") & (df["model"] == "gaussian") & (df["t_max"] == 120)
           & (df["capacity"] == 450) & (df["D"] > 0)]
    return d[d["allocation"] == allocation] if "allocation" in d else d


# --- access profiles ---------------------------------------------------------

@pytest.mark.parametrize("a_c, a_g, a, expected", [
    (1.00, 0.97, 0.97, "Strong access"),
    (0.81, 0.99, 0.78, "Capacity constrained"),
    (1.00, 0.74, 0.71, "Geography constrained"),
    (0.43, 0.41, 0.32, "Doubly constrained"),
    (0.79, 0.85, 0.79, "Capacity constrained"),   # one metric below 80 % is not doubly constrained
])
def test_access_profile_rule(a_c, a_g, a, expected):
    assert access_profile(a_c, a_g, a) == expected


@pytest.mark.skipif(not _WORLD.exists(), reason="pre-computed World results not built")
def test_access_profiles_reproduce_the_paper():
    """Section 3.2: 31 / 10 / 51 / 94 countries with 24 / 36 / 13 / 28 % of demand."""
    c = _main_config(pd.read_parquet(_WORLD))
    c = c.assign(p=[access_profile(r.A_C / r.D, r.A_G / r.D, r.A / r.D) for r in c.itertuples()])
    n = c["p"].value_counts()
    share = (100 * c.groupby("p")["D"].sum() / c["D"].sum()).round()
    assert n.to_dict() == {"Doubly constrained": 94, "Geography constrained": 51, "Strong access": 31,
                           "Capacity constrained": 10}
    assert share.to_dict() == {"Capacity constrained": 36, "Doubly constrained": 28,
                               "Geography constrained": 13, "Strong access": 24}


# --- distance proxy ------------------------------------------------------------

@pytest.mark.parametrize("minutes", [0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 180.0])
def test_proxy_round_trip(minutes):
    assert float(proxy_km_to_minutes(minutes_to_proxy_km(minutes))) == pytest.approx(minutes)


def test_proxy_calibration_points():
    assert [minutes_to_proxy_km(m) for m in (30, 60, 90, 120)] == pytest.approx([12.6, 28.4, 41.6, 64.4])


# --- mixed driving-time matrix ---------------------------------------------------

@pytest.mark.skipif(load_store("ISL") is None, reason="no stored driving times for Iceland")
def test_new_centre_gets_straight_line_times():
    """Stored centres keep their TravelTime column; a centre missing from the store gets
    the calibrated straight-line equivalent instead of being unreachable."""
    gdf = load_population_at_resolution("Iceland", target_resolution=5)
    locs, _ = load_linacs_from_dirac_db("Iceland")
    hex_ids = list(gdf["h3"])
    stored = _regional_tt_matrix("ISL", locs, hex_ids, "driving")
    mixed = _regional_tt_matrix("ISL", list(locs) + [(65.68, -18.09, 1.0)], hex_ids, "driving")  # Akureyri
    np.testing.assert_array_equal(mixed[:, : len(locs)], stored)
    assert np.isfinite(mixed[:, -1]).sum() > 0
    assert np.nanmin(mixed[:, -1]) < 15  # the hexagon holding the new centre is minutes away


# --- centre loads and planning ----------------------------------------------------

def test_centre_loads_sum_to_national_access():
    loads = centre_loads("IS")
    _, ct = compute_members(["IS"])
    assert loads["treated"].sum() == pytest.approx(float(ct["A"].iloc[0]), rel=1e-6)
    assert (loads["treated"] <= loads["capacity"] + 1e-6).all()
    assert loads["demand"].notna().all()  # capped 3SFCA reports the demand directed to each centre


def test_nearest_first_bounded_by_single_metrics():
    _, ct = compute_members(["GB"], allocation="nearest_first")
    r = ct.iloc[0]
    assert r["A"] <= min(r["A_C"], r["A_G"]) + 1e-6


def test_planner_adds_the_requested_linacs():
    before, steps, locs = plan_centres("NG", n_linacs=5, linacs_per_step=2, snap_min=30)
    assert [s["linacs_added"] for s in steps] == [2, 2, 1]
    assert sum(n for *_, n in locs) == pytest.approx(before["n_linacs"] + 5)
    access = [100 * before["A"] / before["D"]] + [s["A/D (%)"] for s in steps]
    assert all(b >= a - 1e-9 for a, b in zip(access, access[1:]))  # access never falls


def test_planner_stops_at_the_target():
    before, steps, _ = plan_centres("GB", target_access=0.8, linacs_per_step=4, snap_min=30)
    assert steps[-1]["A/D (%)"] >= 80.0
    assert len(steps) == 1 or steps[-2]["A/D (%)"] < 80.0  # no step beyond the one that reached it


# --- pre-computed results ----------------------------------------------------------

@pytest.mark.skipif(not _WORLD.exists(), reason="pre-computed World results not built")
@pytest.mark.parametrize("allocation, expected", [("proportional", 60.0), ("nearest_first", 59.0)])
def test_prebuilt_world_matches_the_paper(allocation, expected):
    c = _main_config(pd.read_parquet(_WORLD), allocation)
    assert 100 * c["A"].sum() / c["D"].sum() == pytest.approx(expected, abs=0.05)
    assert 100 * c["A_C"].sum() / c["D"].sum() == pytest.approx(68.5, abs=0.05)
    assert 100 * c["A_G"].sum() / c["D"].sum() == pytest.approx(73.0, abs=0.05)
