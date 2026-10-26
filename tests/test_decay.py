"""Unit tests for analysis.decay — the shared geographic decay functions."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.decay import (
    DECAY_MODELS,
    decay_probability,
    default_cutoff,
    describe_model,
)

X = np.array([0.0, 10.0, 50.0, 100.0, 500.0, np.inf])


@pytest.mark.parametrize("model", DECAY_MODELS)
def test_probability_bounds_and_unreachable(model):
    p = decay_probability(X, model, scale=50, shape=2, threshold=100)
    assert p.shape == X.shape
    assert (p >= 0).all() and (p <= 1).all()
    if model != "uniform":
        assert p[-1] == 0.0, "np.inf must be unreachable"
    else:
        assert p[-1] == 1.0


def test_step_is_binary_at_threshold():
    p = decay_probability(X, "step", threshold=100)
    np.testing.assert_array_equal(p, [1, 1, 1, 1, 0, 0])


def test_weibull_k1_equals_exponential():
    a = decay_probability(X, "weibull", scale=40, shape=1.0, cutoff=1000)
    b = decay_probability(X, "exponential", scale=40, cutoff=1000)
    np.testing.assert_allclose(a, b)


def test_weibull_has_no_default_cutoff():
    lam, k = 30.0, 4.0
    p = decay_probability(X, "weibull", scale=lam, shape=k)
    finite = np.isfinite(X)
    np.testing.assert_allclose(p[finite], np.exp(-(X[finite] / lam) ** k))
    assert default_cutoff("weibull", lam, 0) == float("inf")


def test_gaussian_is_one_at_origin_zero_at_catchment_and_monotone():
    thr = 120.0
    x = np.linspace(0, 150, 301)
    p = decay_probability(x, "gaussian", threshold=thr)
    assert p[0] == pytest.approx(1.0)
    assert p[x <= thr][-1] == pytest.approx(0.0, abs=1e-9)
    assert (np.diff(p[x <= thr]) <= 1e-12).all()
    assert (p[x > thr] == 0).all()


def test_gaussian_reproduces_luo_qi_band_weights():
    # Luo & Qi (2009) used Gaussian weights of ~1.00 / 0.68 / 0.22 for the
    # 0–10 / 10–20 / 20–30 min bands of a 30 min catchment (band midpoints).
    p = decay_probability(np.array([0.0, 15.0, 25.0]), "gaussian", threshold=30)
    assert p[0] == pytest.approx(1.0)
    assert 0.6 < p[1] < 0.75
    assert 0.15 < p[2] < 0.3


def test_power_plateau_and_decay():
    p = decay_probability(np.array([0.0, 5.0, 10.0, 20.0, 40.0]), "power", scale=10, shape=2, cutoff=1000)
    np.testing.assert_allclose(p[:3], 1.0)
    assert p[3] == pytest.approx(0.25)
    assert p[4] == pytest.approx(1 / 16)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        decay_probability(X, "logistic")


@pytest.mark.parametrize("model", DECAY_MODELS)
def test_describe_model_returns_string(model):
    s = describe_model(model, scale=30, shape=2, threshold=60, cutoff=None, unit="km")
    assert isinstance(s, str) and s
