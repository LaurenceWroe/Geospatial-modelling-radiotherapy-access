"""
Geographic access decay functions shared by the accessibility engine, the
floating-catchment benchmarks, and the Probability Models tab.

Every function maps a non-negative geographic variable ``x`` (straight-line
km or travel-time minutes) to a probability P(x) in [0, 1] that a patient at
that separation from a facility accesses treatment there. ``np.inf`` marks an
unreachable hexagon and always maps to 0 (except for the uniform model, which
by definition has no geographic barrier).

Models
------
``step``         P = 1 if x <= threshold else 0.
                 Binary catchment used by classic 2SFCA (Luo & Wang 2003).
``uniform``      P = 1 everywhere. No geographic barrier.
``weibull``      P = exp(-(x/scale)^shape), 0 beyond an optional cutoff.
                 shape=1 is exponential; large shape approaches a step at
                 x = scale.
``exponential``  P = exp(-x/scale), 0 beyond an optional cutoff.
``gaussian``     Truncated, normalised Gaussian within the threshold:
                 P = (exp(-0.5 (x/threshold)^2) - exp(-0.5)) / (1 - exp(-0.5)),
                 0 beyond threshold. P(0)=1, P(threshold)=0. This is the kernel
                 used by the enhanced 2SFCA (Luo & Qi 2009; Dai 2010) and the
                 Gaussian 2SFCA literature that followed.
``power``        P = min(1, (x/scale)^-shape), 0 beyond an optional cutoff.
                 Classic gravity-model impedance (Hansen 1959; Joseph &
                 Bantock 1982) with a plateau inside ``scale`` so the
                 singularity at x = 0 is avoided.

Parameters are expressed in the same units as ``x``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DECAY_MODELS = ("step", "uniform", "weibull", "exponential", "gaussian", "power")

# Human-readable labels in the order they appear in the UI.
DECAY_LABELS = {
    "step": "Step function",
    "weibull": "Weibull",
    "exponential": "Exponential",
    "gaussian": "Gaussian (E2SFCA kernel)",
    "power": "Power law (gravity)",
    "uniform": "Uniform (no decay)",
}

# Which parameters each model reads.
MODEL_PARAMS = {
    "step": ("threshold",),
    "uniform": (),
    "weibull": ("scale", "shape", "cutoff"),
    "exponential": ("scale", "cutoff"),
    "gaussian": ("threshold",),
    "power": ("scale", "shape", "cutoff"),
}

_GAUSS_FLOOR = float(np.exp(-0.5))


def default_cutoff(model: str, scale: float, threshold: float) -> float:
    """Hard cut-off beyond which P is forced to zero, when the user sets none.

    Scale-based models have no cut-off by default (the engine still drops
    pairs with P below a negligible floor). Threshold models return their
    threshold.
    """
    if model in ("weibull", "exponential", "power"):
        return float("inf")
    return float(threshold)


def decay_probability(
    x: np.ndarray,
    model: str,
    *,
    scale: float = 30.0,
    shape: float = 2.0,
    threshold: float = 50.0,
    cutoff: Optional[float] = None,
) -> np.ndarray:
    """Return P(x) for the given decay model as a float64 array in [0, 1]."""
    if model not in DECAY_MODELS:
        raise ValueError(f"Unknown decay model {model!r}; expected one of {DECAY_MODELS}")

    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    xs = np.where(finite, x, 1e9)

    if model == "uniform":
        return np.ones_like(x)

    if model == "step":
        return np.where(finite & (xs <= threshold), 1.0, 0.0)

    if model == "gaussian":
        if threshold <= 0:
            return np.zeros_like(x)
        g = (np.exp(-0.5 * (xs / threshold) ** 2) - _GAUSS_FLOOR) / (1.0 - _GAUSS_FLOOR)
        return np.where(finite & (xs <= threshold), np.clip(g, 0.0, 1.0), 0.0)

    if cutoff is None:
        cutoff = default_cutoff(model, scale, threshold)

    if model == "weibull":
        p = np.exp(-np.power(xs / scale, shape))
    elif model == "exponential":
        p = np.exp(-xs / scale)
    else:  # power
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(xs <= scale, 1.0, np.power(xs / scale, -shape))
    return np.where(finite & (xs <= cutoff), np.clip(p, 0.0, 1.0), 0.0)


def describe_model(model: str, *, scale: float, shape: float, threshold: float,
                   cutoff: Optional[float], unit: str) -> str:
    """One-line description of a parameterised model for captions and stats."""
    if model == "step":
        return f"Step function | threshold = {threshold:.0f} {unit}"
    if model == "uniform":
        return "Uniform (no distance decay)"
    if model == "gaussian":
        return f"Gaussian (E2SFCA) | catchment = {threshold:.0f} {unit}"
    c = cutoff if cutoff is not None else default_cutoff(model, scale, threshold)
    tail = f" | cut-off = {c:.0f} {unit}" if np.isfinite(c) else ""
    if model == "weibull":
        return f"Weibull | λ = {scale:g} {unit} | k = {shape:g}{tail}"
    if model == "exponential":
        return f"Exponential | λ = {scale:g} {unit}{tail}"
    return f"Power law | plateau = {scale:g} {unit} | β = {shape:g}{tail}"
