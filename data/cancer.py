"""
Cancer incidence loader and H3 apportionment.

Data sources
------------
- ``inputs/globocan_xarray_2024.nc``  — national cancer case totals
  (dims: Cancer × Metric × ISO3), the default. The GLOBOCAN 2022 file used
  for the preprint (``globocan_xarray_2022.nc``, 175 countries) is kept; set
  ``RADMAPS_GLOBOCAN=2022`` to select it.
- ``inputs/rt_utilisation_sets.csv``  — optimal RT utilisation rate
  per cancer type, one column per published set (``delaney_2005``,
  ``barton_2014`` …). ``rt_utilisation_sources.csv`` carries the provenance
  (source, year, which treatment intents are covered, whether the column is
  fully populated). Only complete sets are offered in the UI.
- ``inputs/optimal_fractions_per_course.csv``  — evidence-based optimal
  fractions per course by cancer type (Wong, Delaney & Barton 2016), used to
  convert a fractions-per-linac throughput into country-specific patients
  per linac.
- ``inputs/actual_data/{ISO3}.csv``  — observed per-country RT
  utilisation (optional; falls back to optimal if missing).

Cases are apportioned over H3 hexagons proportional to population, so that
the spatial distribution of cancer matches the population distribution.
"""

from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

_ROOT = Path(__file__).resolve().parent.parent
GLOBOCAN_FILES = {
    "2022": _ROOT / "inputs" / "globocan_xarray_2022.nc",
    "2024": _ROOT / "inputs" / "globocan_xarray_2024.nc",
}
GLOBOCAN_YEAR = os.environ.get("RADMAPS_GLOBOCAN", "2024")
XARRAY_PATH = GLOBOCAN_FILES[GLOBOCAN_YEAR]
RTU_SETS_CSV = _ROOT / "inputs" / "rt_utilisation_sets.csv"
RTU_SOURCES_CSV = _ROOT / "inputs" / "rt_utilisation_sources.csv"
FRACTIONS_CSV = _ROOT / "inputs" / "optimal_fractions_per_course.csv"
ACTUAL_DIR = _ROOT / "inputs" / "actual_data"
NEW_CASES_METRIC = "New_Cases_Number"

DEFAULT_RTU_SET = "barton_2014"
PUBLISHED_RTU_SET = "delaney_2005"  # the set used for the arXiv 2605.23456 results

# Wong, Delaney & Barton (2016): 19.4 fractions per course averaged over all
# sites. 450 patients × 19.4 ≈ 8,730 fractions per linac per year, so the
# fractions default reproduces the 450 patients/linac benchmark for a
# population with the Australian 2012 case mix the benchmark implies.
DEFAULT_FRACTIONS_PER_COURSE = 19.4
DEFAULT_FRACTIONS_PER_LINAC_PER_YEAR = 8730.0

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
    """Return True if *iso3* is present in the GLOBOCAN dataset (or is a region code)."""
    from data.regions import REGION_CODES
    if iso3 in REGION_CODES:
        return True
    da = xr.open_dataarray(XARRAY_PATH)
    return iso3 in [str(v) for v in da.coords["ISO3"].values]


@lru_cache(maxsize=1)
def _read_rtu_sets_table() -> pd.DataFrame:
    """RTU sets table indexed by original cancer-type name; empty if absent."""
    if not RTU_SETS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RTU_SETS_CSV, dtype={"cancer_type": str})
    df["cancer_type"] = df["cancer_type"].str.strip()
    return df.set_index("cancer_type")


@lru_cache(maxsize=1)
def _read_rtu_sources() -> List[Dict[str, str]]:
    if not RTU_SOURCES_CSV.exists():
        return []
    with open(RTU_SOURCES_CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def list_rtu_sets(include_incomplete: bool = False) -> List[Dict[str, str]]:
    """Return the published RT utilisation sets with their provenance.

    Each entry has ``set_id``, ``label``, ``source``, ``year``,
    ``intent_coverage``, ``url`` and ``complete`` ("true"/"false"). A set is
    *complete* when every non-aggregate cancer type has a value; incomplete
    sets are hidden from the UI unless *include_incomplete* is True.
    """
    table = _read_rtu_sets_table()
    out = []
    for row in _read_rtu_sources():
        sid = row["set_id"]
        declared = str(row.get("complete", "")).strip().lower() == "true"
        populated = False
        if not table.empty and sid in table.columns:
            sites = [c for c in table.index if c.strip().lower() not in _GLOBOCAN_AGGREGATE_KEYS]
            populated = bool(len(sites)) and table.loc[sites, sid].notna().all()
        complete = declared and populated
        if complete or include_incomplete:
            entry = dict(row)
            entry["complete"] = "true" if complete else "false"
            out.append(entry)
    if not out and not include_incomplete:
        # No sets file: expose the legacy file under the default id.
        out.append({"set_id": PUBLISHED_RTU_SET, "label": "Optimal RTU (Delaney et al. 2005)",
                    "source": "Delaney et al. 2005", "year": "2005",
                    "intent_coverage": "curative + palliative first-course indications",
                    "url": "https://doi.org/10.1002/cncr.21324", "complete": "true"})
    return out


def rtu_set_info(rtu_set: str) -> Dict[str, str]:
    """Provenance row for *rtu_set* (falls back to a minimal stub)."""
    for row in list_rtu_sets(include_incomplete=True):
        if row["set_id"] == rtu_set:
            return row
    return {"set_id": rtu_set, "label": rtu_set, "source": "", "year": "",
            "intent_coverage": "", "url": "", "complete": "false"}


def _load_optimal_fractions(rtu_set: str = DEFAULT_RTU_SET) -> Dict[str, float]:
    """Load optimal RT utilisation fractions keyed by normalised cancer name."""
    return {_norm_key(k): v for k, v in get_optimal_rt_fractions(rtu_set).items()}


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


def get_optimal_rt_fractions(rtu_set: str = DEFAULT_RTU_SET) -> Dict[str, float]:
    """Return optimal RT utilisation fractions keyed by original cancer type name.

    Reads the requested column of ``rt_utilisation_sets.csv``; types with no
    value in that set fall back to the default set.
    """
    table = _read_rtu_sets_table()
    if table.empty:
        raise FileNotFoundError(f"RTU sets table not found: {RTU_SETS_CSV}")
    col = rtu_set if rtu_set in table.columns else DEFAULT_RTU_SET
    base = table[DEFAULT_RTU_SET] if DEFAULT_RTU_SET in table.columns else None
    fracs: Dict[str, float] = {}
    for cancer, v in table[col].items():
        if pd.isna(v) and base is not None:
            v = base.get(cancer, np.nan)
        if pd.notna(v):
            fracs[str(cancer)] = float(v)
    return fracs


# ---------------------------------------------------------------------------
# Fractions-based capacity
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_fractions_per_course() -> Tuple[Dict[str, float], int, int]:
    """Optimal fractions per course by normalised cancer type.

    Returns ``(fractions, n_missing, n_total)``. Types without a published
    value are absent from the dict (callers substitute
    :data:`DEFAULT_FRACTIONS_PER_COURSE`); *n_missing* lets the UI flag an
    incomplete table.
    """
    if not FRACTIONS_CSV.exists():
        return {}, 0, 0
    df = pd.read_csv(FRACTIONS_CSV, comment="#", dtype={"cancer_type": str})
    df["cancer_type"] = df["cancer_type"].str.strip()
    vals = pd.to_numeric(df["fractions_per_course"], errors="coerce")
    fr = {_norm_key(c): float(v) for c, v in zip(df["cancer_type"], vals) if pd.notna(v) and v > 0}
    return fr, int(vals.isna().sum()), int(len(df))


def fractions_per_course_map(site_fractions: tuple = (), factor: float = 1.0) -> Dict[str, float]:
    """Fractions per course keyed by normalised cancer type, after user overrides.

    *site_fractions* is a hashable tuple of ``(cancer_type, fractions)`` pairs
    (e.g. a hypofractionated breast schedule of 5), applied on top of the Wong
    et al. 2016 table; *factor* then scales every course. Types absent from
    both fall back to :data:`DEFAULT_FRACTIONS_PER_COURSE` × factor in callers.
    """
    fr, _, _ = get_fractions_per_course()
    out = dict(fr)
    for name, v in site_fractions:
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out[_norm_key(name)] = v
    f = max(float(factor), 1e-6)
    return {k: v * f for k, v in out.items()}


def effective_capacity_per_linac(
    iso3: str,
    fractions_per_linac_per_year: float = DEFAULT_FRACTIONS_PER_LINAC_PER_YEAR,
    rtu_set: str = DEFAULT_RTU_SET,
    hypofractionation_factor: float = 1.0,
    site_fractions: tuple = (),
) -> Dict[str, float]:
    """Country-specific patients per linac per year from a fractions throughput.

    The mean fractions per course is weighted by each cancer type's share of
    national RT demand (GLOBOCAN incidence × RTU), so a country whose demand
    is dominated by short-course sites gets more patients per machine than
    one dominated by long-course sites. ``hypofractionation_factor`` scales
    every course length (0.7 = 30 % fewer fractions).

    Returns a dict with ``patients_per_linac``, ``mean_fractions_per_course``
    (after the hypofractionation factor), ``demand_share_with_site_data`` (how
    much of demand had a published per-site fraction count rather than the
    overall default) and ``total_rt_demand``.
    """
    factor = max(hypofractionation_factor, 1e-6)
    fr = fractions_per_course_map(site_fractions, factor)
    default_f = DEFAULT_FRACTIONS_PER_COURSE * factor
    all_types = get_cancer_types() + DERIVED_CANCER_TYPES
    cases = get_national_cases(iso3, all_types)
    rtu = _load_optimal_fractions(rtu_set)

    demand_total = 0.0
    fraction_total = 0.0
    with_data = 0.0
    for cancer, n in cases.items():
        if cancer.strip().lower() in _GLOBOCAN_AGGREGATE_KEYS:
            continue
        key = _norm_key(cancer)
        d = n * rtu.get(key, 0.0)
        if d <= 0:
            continue
        f = fr.get(key)
        if f is not None:
            with_data += d
        fraction_total += d * (f if f is not None else default_f)
        demand_total += d

    mean_f = (fraction_total / demand_total if demand_total > 0 else default_f)
    return {
        "patients_per_linac": fractions_per_linac_per_year / mean_f,
        "mean_fractions_per_course": mean_f,
        "demand_share_with_site_data": with_data / demand_total if demand_total > 0 else 0.0,
        "total_rt_demand": demand_total,
    }


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
    if iso3 not in da.coords["ISO3"].values:
        from data.regions import REGION_CODES
        if iso3 in REGION_CODES:  # region missing from the file: sum its members
            import pycountry
            total = dict.fromkeys(cancer_types, 0.0)
            for a2 in REGION_CODES[iso3].member_alpha2:
                c = pycountry.countries.get(alpha_2=a2)
                if c is not None and c.alpha_3 in da.coords["ISO3"].values:
                    for k, v in get_national_cases(c.alpha_3, cancer_types).items():
                        total[k] += v
            return total
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
    rtu_set: str = DEFAULT_RTU_SET,
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
    rtu_set : str
        Which published optimal-RTU set to use (see :func:`list_rtu_sets`).

    Returns
    -------
    GeoDataFrame with additional cancer columns.
    """
    national_cases = get_national_cases(iso3, cancer_types)
    optimal_fracs = _load_optimal_fractions(rtu_set)
    actual_fracs: Optional[Dict[str, float]] = (
        _load_actual_fractions(iso3) if use_actual_rt else None
    )

    pop = gdf["population"].to_numpy(dtype=np.float64)
    total_pop = pop.sum()
    pop_share = pop / total_pop if total_pop > 0 else np.zeros_like(pop)

    new_cols: Dict[str, np.ndarray] = {}
    for cancer in cancer_types:
        n_cases = national_cases.get(cancer, 0.0)
        incidence = pop_share * n_cases

        cancer_key = _norm_key(cancer)
        opt_frac = optimal_fracs.get(cancer_key, 0.0)
        act_frac = actual_fracs.get(cancer_key, opt_frac) if actual_fracs is not None else opt_frac

        new_cols[f"{cancer}_incidence"] = incidence.astype(np.float32)
        new_cols[f"{cancer}_optimal_rt"] = (incidence * opt_frac).astype(np.float32)
        new_cols[f"{cancer}_actual_rt"] = (incidence * act_frac).astype(np.float32)

    result = pd.concat([gdf, pd.DataFrame(new_cols, index=gdf.index)], axis=1)
    return gpd.GeoDataFrame(result, geometry=gdf.geometry.name, crs=gdf.crs)
