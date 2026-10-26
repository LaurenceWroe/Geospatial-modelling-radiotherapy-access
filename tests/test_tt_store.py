"""The res-6 travel-time store: reconstruction by cell id and facility coordinates."""
import h3
import numpy as np
import pytest

from data import tt_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(tt_store, "STORE_DIR", tmp_path)
    tt_store.load_store.cache_clear()
    parent = h3.latlng_to_cell(51.5, -0.12, 5)
    kids = sorted(h3.cell_to_children(parent, 6))          # 7 res-6 cells
    pop = np.array([100, 0, 50, 10, 0, 0, 40], float)
    locs = [(51.50, -0.10), (51.60, -0.20), (51.50, -0.10)]  # facility 2 duplicates 0
    rows = np.array([0, 2, 3, 0, 6, 0])
    cols = np.array([0, 0, 0, 1, 1, 2])
    mins = np.array([10, 20, 30, 50, 70, 12], float)
    tt_store.save_store("TST", "driving", kids, pop, locs, rows, cols, mins, cutoff_min=120.0)
    return tt_store.load_store("TST"), parent, kids, locs


def test_res6_reconstruction_is_exact(store):
    s, _, kids, locs = store
    m = s.matrix(kids, locs)
    assert m[0, 0] == 10 and m[2, 0] == 20 and m[3, 0] == 30
    assert np.isinf(m[1, 0])                       # pair not stored -> unreachable
    assert m[0, 2] == 12                           # duplicate coordinate keeps its own column


def test_requested_order_and_missing_facility(store):
    s, _, kids, locs = store
    m, found = s.matrix(kids[::-1], [locs[1], (0.0, 0.0)], return_found=True)
    assert list(found) == [True, False]
    assert m[0, 0] == 70                           # kids[6] is first after reversal
    assert np.isinf(m[:, 1]).all()


def test_aggregation_is_population_weighted_with_cutoff_fill(store):
    s, parent, _, locs = store
    m = s.matrix([parent], locs[:2])
    w = np.maximum([100, 0, 50, 10, 0, 0, 40], 1.0)
    t0 = np.array([10, 120, 20, 30, 120, 120, 120])   # unreachable children at the cut-off
    t1 = np.array([50, 120, 120, 120, 120, 120, 70])
    assert m[0, 0] == pytest.approx((w * t0).sum() / w.sum(), rel=1e-6)
    assert m[0, 1] == pytest.approx((w * t1).sum() / w.sum(), rel=1e-6)
