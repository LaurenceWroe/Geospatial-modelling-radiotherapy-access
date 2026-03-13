"""
Cancer incidence loader and H3 apportionment.

Data sources
------------
- ``b_cancer_incidence/globocan_xarray.nc``  — national cancer case totals
  (dims: Cancer × Metric × ISO3).
- ``b_cancer_incidence/optimal_rt_utilisations.csv``  — optimal RT fraction
  per cancer type, format: ``Cancer type, fraction`` (no header).
- ``b_cancer_incidence/actual_data/{ISO3}.csv``  — actual per-country RT
  fraction (optional; falls back to optimal if missing).

Cases are apportioned over H3 hexagons proportional to population, so that
the spatial distribution of cancer matches the population distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

XARRAY_PATH = Path("b_cancer_incidence/globocan_xarray.nc")
OPTIMAL_CSV = Path("b_cancer_incidence/optimal_rt_utilisations.csv")
ACTUAL_DIR = Path("b_cancer_incidence/actual_data")
NEW_CASES_METRIC = "New_Cases_Number"

# Derived cancer types not directly in GLOBOCAN — computed from aggregates
DERIVED_CANCER_TYPES: List[str] = ["NMSC", "Other cancers"]
_GLOBOCAN_AGGREGATE_KEYS = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}


def _norm_key(s: str) -> str:
    """Normalise a cancer name to alphanumeric-only lowercase for robust matching."""
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def get_cancer_types() -> List[str]:
    """Return all cancer type names available in the GLOBOCAN DataArray."""
    da = xr.open_dataarray(XARRAY_PATH)
    return sorted(str(v) for v in da.coords["Cancer"].values)


def has_globocan_data(iso3: str) -> bool:
    """Return True if *iso3* is present in the GLOBOCAN dataset."""
    da = xr.open_dataarray(XARRAY_PATH)
    return iso3 in [str(v) for v in da.coords["ISO3"].values]


def _load_optimal_fractions() -> Dict[str, float]:
    """Load optimal RT utilisation fractions keyed by cancer name (lower)."""
    fracs: Dict[str, float] = {}
    try:
        with open(OPTIMAL_CSV) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(",", 1)
                if len(parts) == 2:
                    cancer = _norm_key(parts[0])
                    try:
                        fracs[cancer] = float(parts[1].strip())
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return fracs


def _load_actual_fractions(iso3: str) -> Optional[Dict[str, float]]:
    """Load actual per-country RT utilisation fractions for *iso3*, or None."""
    path = ACTUAL_DIR / f"{iso3}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, header=None, names=["cancer", "fraction"])
        return {_norm_key(r["cancer"]): float(r["fraction"])
                for _, r in df.iterrows()}
    except Exception:
        return None


def get_optimal_rt_fractions() -> Dict[str, float]:
    """Return optimal RT utilisation fractions keyed by original cancer type name."""
    fracs: Dict[str, float] = {}
    try:
        with open(OPTIMAL_CSV) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(",", 1)
                if len(parts) == 2:
                    try:
                        fracs[parts[0].strip()] = float(parts[1].strip())
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return fracs


def _compute_derived_cases(iso3: str, da) -> Dict[str, float]:
    """Compute NMSC and Other cancers case counts from GLOBOCAN aggregates."""
    def _get(cancer: str) -> float:
        try:
            v = float(da.sel(Cancer=cancer, Metric=NEW_CASES_METRIC, ISO3=iso3).values)
            return v if np.isfinite(v) else 0.0
        except (KeyError, ValueError):
            return 0.0

    all_total = _get("All cancers")
    excl_nmsc = _get("All cancers excl. NMSC")
    nmsc = max(0.0, all_total - excl_nmsc)

    # Sum all individual (non-aggregate, non-derived) site cases
    individual_sum = 0.0
    for cancer in da.coords["Cancer"].values:
        c = str(cancer)
        if c.strip().lower() not in _GLOBOCAN_AGGREGATE_KEYS:
            individual_sum += _get(c)

    other = max(0.0, excl_nmsc - individual_sum)
    return {"NMSC": nmsc, "Other cancers": other}


def get_national_cases(iso3: str, cancer_types: List[str]) -> Dict[str, float]:
    """Return national new-case counts for the given cancer types and ISO3.

    Handles the two derived types ``"NMSC"`` and ``"Other cancers"`` which are
    computed from GLOBOCAN aggregates rather than read directly.
    """
    da = xr.open_dataarray(XARRAY_PATH)
    derived: Dict[str, float] = {}
    if any(c in ("NMSC", "Other cancers") for c in cancer_types):
        derived = _compute_derived_cases(iso3, da)

    cases: Dict[str, float] = {}
    for cancer in cancer_types:
        if cancer in ("NMSC", "Other cancers"):
            cases[cancer] = derived.get(cancer, 0.0)
        else:
            try:
                val = float(da.sel(Cancer=cancer, Metric=NEW_CASES_METRIC, ISO3=iso3).values)
                cases[cancer] = val if np.isfinite(val) else 0.0
            except (KeyError, ValueError):
                cases[cancer] = 0.0
    return cases


# ---------------------------------------------------------------------------
# Apportionment
# ---------------------------------------------------------------------------

def apportion_cancer_to_h3(
    gdf: gpd.GeoDataFrame,
    iso3: str,
    cancer_types: List[str],
    use_actual_rt: bool = True,
) -> gpd.GeoDataFrame:
    """Add cancer incidence and RT treatment columns to an H3 GeoDataFrame.

    For each cancer in *cancer_types*, three columns are added:
    - ``{cancer}_incidence``   — estimated new cases in that hexagon
    - ``{cancer}_optimal_rt``  — cases expected to receive optimal RT
    - ``{cancer}_actual_rt``   — cases receiving actual RT (or optimal if
                                  no actual data available for this country)

    Apportionment is proportional to population.

    Parameters
    ----------
    gdf : GeoDataFrame
        H3 population GeoDataFrame with ``h3`` and ``population`` columns.
    iso3 : str
        Three-letter ISO country code (upper-case).
    cancer_types : list of str
        Cancer type names matching GLOBOCAN ``Cancer`` dimension.
    use_actual_rt : bool
        If True, load actual RT fractions from ``actual_data/{ISO3}.csv``
        and fall back to optimal when absent.

    Returns
    -------
    GeoDataFrame with additional cancer columns.
    """
    national_cases = get_national_cases(iso3, cancer_types)
    optimal_fracs = _load_optimal_fractions()
    actual_fracs: Optional[Dict[str, float]] = (
        _load_actual_fractions(iso3) if use_actual_rt else None
    )

    pop = gdf["population"].to_numpy(dtype=np.float64)
    total_pop = pop.sum()
    pop_share = pop / total_pop if total_pop > 0 else np.zeros_like(pop)

    result = gdf.copy()
    for cancer in cancer_types:
        n_cases = national_cases.get(cancer, 0.0)
        incidence = pop_share * n_cases
        result[f"{cancer}_incidence"] = incidence.astype(np.float32)

        cancer_key = _norm_key(cancer)
        opt_frac = optimal_fracs.get(cancer_key, 0.0)
        result[f"{cancer}_optimal_rt"] = (incidence * opt_frac).astype(np.float32)

        if actual_fracs is not None:
            act_frac = actual_fracs.get(cancer_key, opt_frac)
        else:
            act_frac = opt_frac
        result[f"{cancer}_actual_rt"] = (incidence * act_frac).astype(np.float32)

    return result
