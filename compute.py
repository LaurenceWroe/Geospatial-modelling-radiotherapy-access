"""
Cached data-loading and radiotherapy-access computation layer for RadMaps.

Pure of UI state — every function takes explicit arguments and Streamlit caches
them across reruns. Extracted from app.py to keep the main file focused on the
sidebar, tabs, and map rendering.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pycountry
import streamlit as st
import xarray as xr

from data.population import load_population_at_resolution, load_region_population
from data.linacs import load_linacs_from_dirac_db, load_linacs_for_region, facility_quality_report
from data.regions import is_region, get_region, REGIONS, REGION_GLOBOCAN_CODES
from data.cancer import (
    get_cancer_types, apportion_cancer_to_h3, get_national_cases,
    get_optimal_rt_fractions, effective_capacity_per_linac, fractions_per_course_map,
    _load_optimal_fractions, _norm_key,
    DERIVED_CANCER_TYPES, XARRAY_PATH, DEFAULT_RTU_SET, DEFAULT_FRACTIONS_PER_COURSE,
)
from data.travel_time import CACHE_DIR as _TT_CACHE_DIR
import h3

from data.tt_store import load_store, store_matrix
from analysis.accessibility import (
    compute_accessibility, aggregate_access_gdf, attach_fraction_metrics, FRACTION_COLUMNS,
)
from analysis.fca import compute_fca

# Disk cache for apportioned regional cancer GeoDataFrames.  Mirrors the region
# *population* parquet cache (data.population.REGION_CACHE_DIR) so that the
# expensive per-country apportionment loop survives app restarts, redeploys, and
# LRU eviction of the in-memory st.cache_data — not just a single session.
_REGION_CANCER_CACHE_DIR = Path("H3_region_cache")

_AGGREGATE_CANCER_KEYS = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}


def _region_cancer_cache_path(globocan_code: str, resolution: int,
                              cancers: tuple, use_actual: bool) -> Path:
    """Stable on-disk path for one (region, resolution, cancers, actual) result."""
    key = json.dumps({"c": sorted(cancers), "a": bool(use_actual), "g": XARRAY_PATH.name}, sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    return _REGION_CANCER_CACHE_DIR / f"{globocan_code}_res{resolution}_cancer_{digest}.parquet"


@st.cache_data(show_spinner=False)
def _country_options() -> list[str]:
    """Sorted list of country display names available in GLOBOCAN (regions excluded)."""
    da = xr.open_dataarray(XARRAY_PATH)
    iso3_set = {str(v) for v in da.coords["ISO3"].values}
    names = []
    for iso3 in iso3_set:
        if iso3 in REGION_GLOBOCAN_CODES:
            continue
        try:
            names.append(pycountry.countries.get(alpha_3=iso3).name)
        except AttributeError:
            pass
    return sorted(names)


@st.cache_data(show_spinner=False)
def _region_options() -> list[str]:
    """List of region display names."""
    return [r.display_name for r in REGIONS]


@st.cache_data(show_spinner=False)
def _region_country_summary(region_name: str) -> pd.DataFrame:
    """Per-country GLOBOCAN + DIRAC summary table for the data tab (per-country mode)."""
    reg = get_region(region_name)
    rows = []
    for alpha2 in reg.member_alpha2:
        c_obj = pycountry.countries.get(alpha_2=alpha2)
        if c_obj is None:
            continue
        c_name = c_obj.name
        c_iso3 = c_obj.alpha_3
        c_cancer = 0
        try:
            c_cases = get_national_cases(c_iso3, ["All cancers excl. NMSC"])
            c_cancer = int(round(c_cases.get("All cancers excl. NMSC", 0.0)))
        except Exception:
            pass
        c_linacs = 0
        try:
            _, c_fac = load_linacs_from_dirac_db(c_name)
            if c_fac is not None and len(c_fac) > 0:
                c_linacs = int(c_fac["n_linacs"].sum())
        except Exception:
            pass
        rows.append({
            "Country": c_name,
            "ISO3": c_iso3,
            "Cancer incidence (excl. NMSC)": c_cancer,
            "LINACs (DIRAC)": c_linacs,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Cancer incidence (excl. NMSC)", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False, max_entries=8)
def _load_pop(country: str, h3_res: int = 8):
    return load_population_at_resolution(country, target_resolution=h3_res)


@st.cache_data(show_spinner=False, max_entries=3)
def _load_pop_region(region_name: str, h3_res: int = 3):
    return load_region_population(region_name, target_resolution=h3_res)


@st.cache_data(show_spinner=False, max_entries=8)
def _load_cancer(country: str, iso3: str, cancers: tuple, use_actual: bool,
                 h3_res: int = 8, region_flag: bool = False):
    gdf = _load_pop_region(country, h3_res) if region_flag else _load_pop(country, h3_res)
    return apportion_cancer_to_h3(gdf, iso3, list(cancers), use_actual_rt=use_actual)


@st.cache_data(show_spinner=False, max_entries=4)
def _load_cancer_region_percountry(region_name: str, cancers: tuple, use_actual: bool, h3_res: int = 3):
    """Build a cancer GeoDataFrame for a region using per-country GLOBOCAN data.

    Each country's cancer cases are apportioned to its own hexes using that
    country's iso3, then all country GDFs are concatenated.  Border hex
    duplicates are resolved by keeping the row with higher total cancer value.

    Result is persisted to a parquet disk cache (mirroring the region-population
    cache) so the ~200-country apportionment loop is not repeated on restart,
    redeploy, or after the in-memory cache evicts.
    """
    import geopandas as gpd
    from data.regions import get_region as _get_region

    reg = _get_region(region_name)

    # --- fast path: load from disk cache ---
    cache_path = _region_cancer_cache_path(reg.globocan_code, h3_res, cancers, use_actual)
    if cache_path.exists():
        try:
            return gpd.read_parquet(cache_path)
        except Exception:
            pass  # corrupt/partial file -> fall through and rebuild

    cancer_list = list(cancers)
    all_gdfs = []

    for alpha2 in reg.member_alpha2:
        country_obj = pycountry.countries.get(alpha_2=alpha2)
        if country_obj is None:
            continue
        c_name = country_obj.name
        c_iso3 = country_obj.alpha_3
        try:
            gdf = _load_pop(c_name, h3_res)
            c_gdf = apportion_cancer_to_h3(gdf, c_iso3, cancer_list, use_actual_rt=use_actual)
            all_gdfs.append(c_gdf)
        except Exception:
            continue

    if not all_gdfs:
        raise ValueError(f"No cancer data found for region {region_name!r}")

    cancer_cols = [c for c in all_gdfs[0].columns if c not in ("h3", "population", "geometry")]
    combined = pd.concat(
        [g[["h3", "population", "geometry"] + [c for c in cancer_cols if c in g.columns]]
         for g in all_gdfs],
        ignore_index=True,
    )
    # Deduplicate border hexes: keep row with highest total cancer burden
    _sum_cols = [c for c in cancer_cols if c in combined.columns]
    if _sum_cols:
        combined["_total"] = combined[_sum_cols].sum(axis=1)
        combined = combined.sort_values("_total", ascending=False).drop_duplicates("h3").drop(columns="_total")
    else:
        combined = combined.drop_duplicates("h3")
    combined = combined.reset_index(drop=True)
    result = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

    # --- persist to disk cache ---
    try:
        _REGION_CANCER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path)
    except Exception:
        pass  # non-fatal: disk cache is an optimisation, not correctness

    return result


# ---------------------------------------------------------------------------
# Demand construction (shared by every compute path)
# ---------------------------------------------------------------------------

def _national_mean_fractions(iso3: str, rtu_set: str, fmap: dict, default_f: float) -> float:
    """Demand-weighted mean fractions per course for a country's optimal RT mix."""
    cases = get_national_cases(iso3, get_cancer_types() + DERIVED_CANCER_TYPES)
    rtu = _load_optimal_fractions(rtu_set)
    num = den = 0.0
    for cancer, n in cases.items():
        if cancer.strip().lower() in _AGGREGATE_CANCER_KEYS:
            continue
        d = n * rtu.get(_norm_key(cancer), 0.0)
        if d > 0:
            num += d * fmap.get(_norm_key(cancer), default_f)
            den += d
    return num / den if den > 0 else default_f


def _build_demand(gdf, iso3: str, rt_method: str, rt_fraction: float,
                  custom_rtu: tuple, rtu_set: str = DEFAULT_RTU_SET,
                  site_fractions: tuple = (), hypofractionation: float = 1.0):
    """Return (demand, total cancer excl. NMSC, fractions demand); each None on failure.

    ``optimal``      per-type incidence × optimal RTU of *rtu_set*, summed over
                     individual (non-aggregate) cancer types.
    ``custom``       per-type incidence × user percentages from *custom_rtu*.
    ``proportional`` All cancers excl. NMSC × *rt_fraction*.

    *fractions demand* is Σ_i demand_{h,i} × fractions per course_i (Wong et al.
    2016, with *site_fractions* overrides and the *hypofractionation* factor),
    i.e. the annual treatment workload each hexagon's demand represents. For
    the proportional method, which has no per-type split, the national
    demand-weighted mean course length is used.
    Returns (None, None, None) if GLOBOCAN data cannot be loaded; the caller
    then falls back to raw population as the demand proxy.
    """
    fmap = fractions_per_course_map(site_fractions, hypofractionation)
    default_f = DEFAULT_FRACTIONS_PER_COURSE * max(hypofractionation, 1e-6)
    try:
        if rt_method in ("optimal", "custom"):
            all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES
            cancer_gdf = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False, rtu_set=rtu_set)
            excl_col = "All cancers excl. NMSC_incidence"
            total_excl = float(cancer_gdf[excl_col].clip(lower=0).sum()) if excl_col in cancer_gdf.columns else None
            demand = None
            fractions = None
            for cancer in all_cancers:
                if cancer.strip().lower() in _AGGREGATE_CANCER_KEYS:
                    continue
                if rt_method == "optimal":
                    col = f"{cancer}_optimal_rt"
                    if col not in cancer_gdf.columns:
                        continue
                    d_i = cancer_gdf[col].clip(lower=0).to_numpy(np.float64)
                else:
                    col = f"{cancer}_incidence"
                    if col not in cancer_gdf.columns:
                        continue
                    frac = dict((k.strip().lower(), v / 100.0) for k, v in custom_rtu).get(cancer.strip().lower(), 0.0)
                    d_i = cancer_gdf[col].clip(lower=0).to_numpy(np.float64) * frac
                f_i = fmap.get(_norm_key(cancer), default_f)
                demand = d_i if demand is None else demand + d_i
                fractions = d_i * f_i if fractions is None else fractions + d_i * f_i
            return demand, total_excl, fractions
        cancer_gdf = apportion_cancer_to_h3(gdf, iso3, ["All cancers excl. NMSC"], use_actual_rt=False)
        col = "All cancers excl. NMSC_incidence"
        if col not in cancer_gdf.columns:
            return None, None, None
        total_excl = float(cancer_gdf[col].clip(lower=0).sum())
        demand = (cancer_gdf[col].clip(lower=0) * rt_fraction).to_numpy(np.float64)
        mean_f = _national_mean_fractions(iso3, rtu_set, fmap, default_f)
        return demand, total_excl, demand * mean_f
    except Exception:
        return None, None, None


def _finish_result(gdf_out, stats, demand, fractions_demand, capacity_per_machine_per_year,
                   fractions_per_linac):
    """Attach workload metrics. With no fractions demand (population fallback) nothing is added."""
    if fractions_demand is None or demand is None:
        return gdf_out, stats
    if fractions_per_linac is None:
        # Patients-per-linac mode: express the same throughput in fractions at
        # this country's mean course length so the two units stay consistent.
        d_tot = float(np.sum(demand))
        mean_f = float(np.sum(fractions_demand)) / d_tot if d_tot > 0 else DEFAULT_FRACTIONS_PER_COURSE
        fractions_per_linac = capacity_per_machine_per_year * mean_f
    return attach_fraction_metrics(gdf_out, stats, fractions_demand, fractions_per_linac)


@st.cache_data(show_spinner=False, max_entries=64)
def _effective_capacity(iso3: str, fractions_per_linac: float, rtu_set: str,
                        hypofractionation: float, site_fractions: tuple = ()) -> dict:
    """Country-specific patients per linac from a fractions throughput (cached)."""
    return effective_capacity_per_linac(iso3, fractions_per_linac, rtu_set, hypofractionation, site_fractions)


# ---------------------------------------------------------------------------
# Access computation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=4)
def _compute_access(
    country: str,
    iso3: str,
    linac_locs: tuple,
    lambda_km: float,
    model: str,
    max_distance_km: float,
    capacity_per_machine_per_year: float,
    rt_method: str = "optimal",
    rt_fraction: float = 0.25,
    h3_res: int = 8,
    region_flag: bool = False,
    snap_linacs_to_hex: bool = False,
    weibull_k: float = 2.0,
    custom_rtu: tuple = (),
    rtu_set: str = DEFAULT_RTU_SET,
    allocation: str = "nearest_first",
    size_weighting: bool = True,
    cutoff_km: Optional[float] = None,
    site_fractions: tuple = (),
    hypofractionation: float = 1.0,
    fractions_per_linac: Optional[float] = None,
):
    # At coarse display resolutions the hex centroid is a poor proxy for where
    # people live (res 3 ≈ 12,400 km²). Compute at ≥ res 5 and aggregate down.
    compute_res = h3_res if (region_flag or h3_res >= 5) else 5
    gdf = _load_pop_region(country, compute_res) if region_flag else _load_pop(country, compute_res)
    demand, total_cancer_excl_nmsc, fractions_demand = _build_demand(
        gdf, iso3, rt_method, rt_fraction, custom_rtu, rtu_set, site_fractions, hypofractionation)

    gdf_out, stats = compute_accessibility(
        gdf,
        list(linac_locs),
        lambda_km=lambda_km,
        cutoff_km=cutoff_km,
        model=model,
        max_distance_km=max_distance_km,
        weibull_k=weibull_k,
        capacity_per_machine_per_year=capacity_per_machine_per_year,
        demand=demand,
        snap_linacs_to_hex=snap_linacs_to_hex,
        h3_resolution=compute_res,
        allocation=allocation,
        size_weighting=size_weighting,
    )
    gdf_out, stats = _finish_result(gdf_out, stats, demand, fractions_demand,
                                    capacity_per_machine_per_year, fractions_per_linac)
    if compute_res != h3_res:
        gdf_out = aggregate_access_gdf(gdf_out, h3_res)
        stats["n_hexagons"] = len(gdf_out)
        stats["compute_resolution"] = compute_res
    stats["total_cancer_excl_nmsc"] = total_cancer_excl_nmsc
    stats["rtu_set"] = rtu_set
    return gdf_out, stats


def _load_tt_matrix(tt_cache_key: str, tt_mode: str) -> np.ndarray:
    cache_file = _TT_CACHE_DIR / f"{tt_cache_key}_{tt_mode}.npz"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Travel time cache not found: {cache_file.name}. "
            "Press 'Fetch TravelTime Data' before computing accessibility."
        )
    return np.load(cache_file)["matrix"]


@st.cache_data(show_spinner=False, max_entries=4)
def _compute_access_travel_time(
    country: str,
    iso3: str,
    linac_locs: tuple,
    lambda_km: float,
    model: str,
    max_distance_km: float,
    capacity_per_machine_per_year: float,
    tt_mode: str,           # "driving" or "public_transport"
    tt_cache_key: str,      # pre-computed cache key
    rt_method: str = "optimal",
    rt_fraction: float = 0.25,
    h3_res: int = 8,
    region_flag: bool = False,
    snap_linacs_to_hex: bool = False,
    weibull_k: float = 2.0,
    custom_rtu: tuple = (),
    rtu_set: str = DEFAULT_RTU_SET,
    allocation: str = "nearest_first",
    size_weighting: bool = True,
    cutoff_km: Optional[float] = None,
    site_fractions: tuple = (),
    hypofractionation: float = 1.0,
    fractions_per_linac: Optional[float] = None,
):
    """Like _compute_access but loads a pre-computed travel time matrix from disk."""
    gdf = _load_pop_region(country, h3_res) if region_flag else _load_pop(country, h3_res)
    demand, total_cancer_excl_nmsc, fractions_demand = _build_demand(
        gdf, iso3, rt_method, rt_fraction, custom_rtu, rtu_set, site_fractions, hypofractionation)
    tt_matrix = _load_tt_matrix(tt_cache_key, tt_mode)

    gdf_out, stats = compute_accessibility(
        gdf,
        list(linac_locs),
        lambda_km=lambda_km,
        cutoff_km=cutoff_km,
        model=model,
        max_distance_km=max_distance_km,
        weibull_k=weibull_k,
        capacity_per_machine_per_year=capacity_per_machine_per_year,
        demand=demand,
        snap_linacs_to_hex=snap_linacs_to_hex,
        h3_resolution=h3_res,
        travel_time_matrix=tt_matrix,
        allocation=allocation,
        size_weighting=size_weighting,
    )
    gdf_out, stats = _finish_result(gdf_out, stats, demand, fractions_demand,
                                    capacity_per_machine_per_year, fractions_per_linac)
    stats["total_cancer_excl_nmsc"] = total_cancer_excl_nmsc
    stats["rtu_set"] = rtu_set
    return gdf_out, stats


# Regional driving time uses the per-country 120-min res-6 store
# (data/tt_store.py), aggregated to the compute resolution. Countries TravelTime
# does not serve fall back to great-circle distance, converted with the
# calibrated km equivalent of each driving-time budget (scripts/calibrate_fit.py).
REGION_TT_RES = 5
REGION_TT_MAX_MIN = 120.0
_PROXY_MIN = [0.0, 30.0, 60.0, 90.0, 120.0]
# straight-line km giving the same population coverage as each driving-time budget,
# fitted on the 152 countries in the local-08:00 travel-time store (scripts/calibrate_fit.py)
_PROXY_KM = [0.0, 12.6, 28.4, 41.6, 64.4]


def minutes_to_proxy_km(minutes: Optional[float]) -> Optional[float]:
    """Great-circle km equivalent of a driving-time budget (linear beyond 120 min)."""
    if minutes is None:
        return None
    if minutes <= _PROXY_MIN[-1]:
        return float(np.interp(minutes, _PROXY_MIN, _PROXY_KM))
    return _PROXY_KM[-1] * minutes / _PROXY_MIN[-1]


def proxy_km_to_minutes(km):
    """Inverse of minutes_to_proxy_km: the driving time whose calibrated
    straight-line equivalent is *km* (linear beyond 120 min)."""
    km = np.asarray(km, dtype=np.float64)
    return np.where(km <= _PROXY_KM[-1], np.interp(km, _PROXY_KM, _PROXY_MIN),
                    km * _PROXY_MIN[-1] / _PROXY_KM[-1])


def _great_circle_km(lat_lon: np.ndarray, lat: float, lon: float) -> np.ndarray:
    la1, lo1 = np.radians(lat_lon[:, 0]), np.radians(lat_lon[:, 1])
    la2, lo2 = np.radians(lat), np.radians(lon)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def proxy_minutes_column(hex_ids, lat: float, lon: float) -> np.ndarray:
    """Driving minutes from each hexagon centroid to (lat, lon), from the calibrated
    straight-line equivalent; inf beyond the store's 120-min cut-off."""
    ll = np.array([h3.cell_to_latlng(c) for c in hex_ids])
    t = proxy_km_to_minutes(_great_circle_km(ll, lat, lon)).astype(np.float32)
    t[t > REGION_TT_MAX_MIN] = np.inf
    return t


def _regional_tt_matrix(iso3: str, locs, hex_ids, tt_mode: str):
    """2 h matrix for this country's facilities from the res-6 store, or None when
    the country has no store. Facilities missing from the store (new or planned
    centres) get the calibrated straight-line equivalent of driving time."""
    store = load_store(iso3, tt_mode)
    if store is None or not locs:
        return None
    m, found = store.matrix(hex_ids, locs, return_found=True)
    for j in np.flatnonzero(~found):
        m[:, j] = proxy_minutes_column(hex_ids, locs[j][0], locs[j][1])
    return m if np.isfinite(m).any() else None


@st.cache_data(show_spinner=False, max_entries=1000)
def _compute_access_one_country_for_region(
    alpha2: str,
    lambda_km: float,
    model: str,
    max_distance_km: float,
    capacity_per_machine_per_year: float,
    rt_method: str,
    rt_fraction: float,
    h3_res: int,
    snap_linacs_to_hex: bool,
    weibull_k: float,
    custom_rtu: tuple,
    rtu_set: str = DEFAULT_RTU_SET,
    allocation: str = "nearest_first",
    size_weighting: bool = True,
    cutoff_km: Optional[float] = None,
    fractions_per_linac: Optional[float] = None,
    hypofractionation: float = 1.0,
    site_fractions: tuple = (),
    tt_mode: Optional[str] = None,
    linac_locs: Optional[tuple] = None,
):
    """Cached per-country computation for use in per-country regional aggregation.

    Returns (gdf_subset, c_stats) or None if the country should be skipped
    (no population data or no GLOBOCAN data).  Countries with no LINACs return
    a gdf with zero access so their unmet demand is still counted.

    When *fractions_per_linac* is given, the per-machine capacity is derived
    for each country from its own cancer-type mix (see
    ``data.cancer.effective_capacity_per_linac``) instead of the flat value.

    With *tt_mode* set, *max_distance_km*, *lambda_km* and *cutoff_km* are in
    minutes: the country's cached res-5 matrix is used if one exists, otherwise
    the minutes are converted to calibrated great-circle km
    (``c_stats["time_source"]`` is "traveltime", "proxy" or "none").

    *linac_locs* ((lat, lon, n_linacs), ...) replaces the DIRAC facilities,
    e.g. with edited machine counts at the same centres.
    """
    country_obj = pycountry.countries.get(alpha_2=alpha2)
    if country_obj is None:
        return None

    c_name = country_obj.name
    c_iso3 = country_obj.alpha_3

    # Compute at ≥ res 4 so centroid-to-facility distances are meaningful,
    # then aggregate down to the display resolution.
    compute_res = max(h3_res, REGION_TT_RES if tt_mode else 4)

    try:
        gdf = _load_pop(c_name, compute_res)
    except Exception:
        return None

    demand, c_cancer_excl_nmsc, fractions_demand = _build_demand(
        gdf, c_iso3, rt_method, rt_fraction, custom_rtu, rtu_set, site_fractions, hypofractionation)
    if demand is None:
        return None  # no GLOBOCAN data

    cap_per_machine = capacity_per_machine_per_year
    cap_basis = None
    if fractions_per_linac is not None:
        try:
            cap_basis = effective_capacity_per_linac(c_iso3, fractions_per_linac, rtu_set,
                                                     hypofractionation, site_fractions)
            cap_per_machine = float(cap_basis["patients_per_linac"])
        except Exception:
            cap_basis = None

    if linac_locs is not None:
        c_locs = [loc for loc in linac_locs if loc[2] > 0]
        has_linacs = bool(c_locs)
    else:
        try:
            c_locs, _ = load_linacs_from_dirac_db(c_name)
            has_linacs = bool(c_locs)
        except (ValueError, FileNotFoundError):
            c_locs = []
            has_linacs = False

    time_source = None
    tt = None
    if tt_mode and has_linacs:
        tt = _regional_tt_matrix(c_iso3, c_locs, list(gdf["h3"]), tt_mode)
        time_source = "traveltime" if tt is not None else "proxy"
        if tt is None:
            lambda_km = minutes_to_proxy_km(lambda_km)  # scale or power-law plateau, in minutes
            max_distance_km = minutes_to_proxy_km(max_distance_km)
            cutoff_km = minutes_to_proxy_km(cutoff_km)
    elif tt_mode:
        time_source = "none"

    if has_linacs:
        gdf_out, c_stats = compute_accessibility(
            gdf, c_locs,
            lambda_km=lambda_km, cutoff_km=cutoff_km, model=model,
            max_distance_km=max_distance_km, weibull_k=weibull_k,
            capacity_per_machine_per_year=cap_per_machine,
            demand=demand, snap_linacs_to_hex=snap_linacs_to_hex,
            h3_resolution=compute_res, travel_time_matrix=tt,
            allocation=allocation, size_weighting=size_weighting,
        )
    else:
        gdf_out = gdf.copy()
        gdf_out["nearest_linac_km"] = np.float32(np.nan)
        gdf_out["access_probability"] = np.float32(0.0)
        gdf_out["capacity_limited_probability"] = np.float32(0.0)
        gdf_out["rt_demand"] = demand.astype(np.float32)
        gdf_out["rt_treated"] = np.float32(0.0)
        gdf_out["rt_untreated"] = demand.astype(np.float32)
        gdf_out["pop_with_access"] = np.float32(0.0)
        gdf_out["sfca3_ratio"] = np.float32(0.0)
        gdf_out["sfca3_supply"] = np.float32(0.0)
        if tt_mode:
            gdf_out["nearest_linac_min"] = np.float32(np.nan)
        c_stats = {"n_facilities": 0, "total_machines": 0, "total_national_capacity": 0.0,
                   "total_rt_demand": float(demand.sum()), "total_rt_treated": 0.0,
                   "geo_accessible_demand": 0.0}
    gdf_out, c_stats = _finish_result(gdf_out, c_stats, demand, fractions_demand,
                                      cap_per_machine, fractions_per_linac)
    if tt_mode and "nearest_linac_min" not in gdf_out.columns:  # proxy country
        gdf_out["nearest_linac_min"] = np.float32(np.nan)

    gdf_out["country"] = c_name
    if compute_res != h3_res:
        gdf_out = aggregate_access_gdf(gdf_out, h3_res)
        gdf_out["country"] = c_name

    c_stats["time_source"] = time_source
    c_stats["cancer_excl_nmsc"] = c_cancer_excl_nmsc or 0.0
    c_stats["capacity_per_machine_per_year"] = cap_per_machine
    if cap_basis is not None:
        c_stats["capacity_basis"] = cap_basis
    return gdf_out, c_stats


def _compute_access_region_percountry(
    region_name: str,
    lambda_km: float,
    model: str,
    max_distance_km: float,
    capacity_per_machine_per_year: float,
    rt_method: str,
    rt_fraction: float,
    h3_res: int,
    snap_linacs_to_hex: bool,
    weibull_k: float,
    custom_rtu: tuple,
    progress_callback=None,
    rtu_set: str = DEFAULT_RTU_SET,
    allocation: str = "nearest_first",
    size_weighting: bool = True,
    cutoff_km: Optional[float] = None,
    fractions_per_linac: Optional[float] = None,
    hypofractionation: float = 1.0,
    site_fractions: tuple = (),
    tt_mode: Optional[str] = None,
):
    """Aggregate per-country RT access results into a single regional GeoDataFrame.

    progress_callback(done, total) is called after each country completes.
    With *tt_mode* set, countries are computed on driving time (see
    _compute_access_one_country_for_region); stats["proxy_countries"] lists
    those that fell back to the distance proxy.
    Per-country computations are individually cached via
    _compute_access_one_country_for_region.
    """
    import geopandas as gpd
    from data.regions import get_region as _get_region

    reg = _get_region(region_name)
    alpha2_list = reg.member_alpha2
    n_total = len(alpha2_list)

    all_gdfs = []
    skipped_countries: list[str] = []
    total_rt_demand = 0.0
    total_rt_treated = 0.0
    total_n_facilities = 0
    total_machines = 0
    total_capacity = 0.0
    total_geo_accessible = 0.0
    total_cancer_excl_nmsc = 0.0
    total_fd = total_ft = total_fcap = 0.0
    any_fractions = False
    proxy_countries: list[str] = []

    for i, alpha2 in enumerate(alpha2_list):
        result = _compute_access_one_country_for_region(
            alpha2, lambda_km, model, max_distance_km,
            capacity_per_machine_per_year, rt_method, rt_fraction,
            h3_res, snap_linacs_to_hex, weibull_k, custom_rtu,
            rtu_set=rtu_set, allocation=allocation, size_weighting=size_weighting,
            cutoff_km=cutoff_km, fractions_per_linac=fractions_per_linac,
            hypofractionation=hypofractionation, site_fractions=site_fractions,
            tt_mode=tt_mode,
        )
        if progress_callback is not None:
            progress_callback(i + 1, n_total)
        if result is None:
            _c_obj = pycountry.countries.get(alpha_2=alpha2)
            skipped_countries.append(_c_obj.name if _c_obj else alpha2)
            continue
        gdf_out, c_stats = result
        if c_stats.get("time_source") == "proxy" and c_stats["total_rt_demand"] > 0:
            proxy_countries.append(gdf_out["country"].iloc[0])
        total_rt_demand += c_stats["total_rt_demand"]
        total_rt_treated += c_stats.get("total_rt_treated", 0.0)
        total_n_facilities += c_stats.get("n_facilities", 0)
        total_machines += c_stats.get("total_machines", 0)
        total_capacity += c_stats.get("total_national_capacity",
                                      c_stats.get("total_machines", 0) * capacity_per_machine_per_year)
        total_geo_accessible += c_stats.get("geo_accessible_demand", 0.0)
        total_cancer_excl_nmsc += c_stats.get("cancer_excl_nmsc", 0.0)
        if c_stats.get("total_fractions_demand") is not None:
            any_fractions = True
            total_fd += c_stats["total_fractions_demand"]
            total_ft += c_stats.get("total_fractions_treated", 0.0)
            total_fcap += c_stats.get("fractions_capacity") or 0.0
        all_gdfs.append(gdf_out)

    if not all_gdfs:
        raise ValueError(f"No country data found for region {region_name!r}")

    _base_cols = ["h3", "population", "geometry", "country", "nearest_linac_km",
                  "access_probability", "capacity_limited_probability",
                  "rt_demand", "rt_treated", "rt_untreated", "pop_with_access", "sfca3_supply"]
    if tt_mode:
        _base_cols.append("nearest_linac_min")
    _frac_cols = [c for c in FRACTION_COLUMNS if all(c in g.columns for g in all_gdfs)]
    combined = pd.concat(
        [g[_base_cols + _frac_cols] for g in all_gdfs],
        ignore_index=True,
    )

    # Border hexes appear once per country, each carrying that country's
    # population/demand share (Kontur clips to national boundaries). Sum the
    # count columns and recompute the ratios so nothing is dropped.
    _w = combined["population"].clip(lower=1.0)
    combined["_w"] = _w
    combined["_wp"] = combined["access_probability"] * _w
    _km_valid = combined["nearest_linac_km"].notna()
    combined["_wd_km"] = (combined["nearest_linac_km"] * _w).where(_km_valid, 0.0)
    combined["_w_km"] = _w.where(_km_valid, 0.0)
    if tt_mode:
        _min_valid = combined["nearest_linac_min"].notna()
        combined["_wd_min"] = (combined["nearest_linac_min"] * _w).where(_min_valid, 0.0)
        combined["_w_min"] = _w.where(_min_valid, 0.0)

    grp = combined.groupby("h3", sort=False)
    _spec = dict(
        population=("population", "sum"),
        rt_demand=("rt_demand", "sum"),
        rt_treated=("rt_treated", "sum"),
        rt_untreated=("rt_untreated", "sum"),
        pop_with_access=("pop_with_access", "sum"),
        sfca3_supply=("sfca3_supply", "sum"),
        _w=("_w", "sum"),
        _wp=("_wp", "sum"),
        _wd_km=("_wd_km", "sum"),
        _w_km=("_w_km", "sum"),
        geometry=("geometry", "first"),
    )
    if tt_mode:
        _spec["_wd_min"] = ("_wd_min", "sum")
        _spec["_w_min"] = ("_w_min", "sum")
    for c in _frac_cols:
        _spec[c] = (c, "sum")
    merged = grp.agg(**_spec).reset_index()

    # Attribute each hex to the country holding most of its population
    _country_top = (
        combined.sort_values("population", ascending=False)
        .drop_duplicates("h3")[["h3", "country"]]
    )
    merged = merged.merge(_country_top, on="h3", how="left")

    merged["access_probability"] = (merged["_wp"] / merged["_w"]).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        merged["nearest_linac_km"] = (
            merged["_wd_km"] / merged["_w_km"].replace(0.0, np.nan)
        ).astype(np.float32)
        if tt_mode:
            merged["nearest_linac_min"] = (
                merged["_wd_min"] / merged["_w_min"].replace(0.0, np.nan)
            ).astype(np.float32)
            merged = merged.drop(columns=["_wd_min", "_w_min"])
    _dem = merged["rt_demand"].to_numpy(np.float64)
    merged["capacity_limited_probability"] = np.where(
        _dem > 0, merged["rt_treated"] / np.maximum(_dem, 1e-9), 0.0
    ).astype(np.float32)
    merged["sfca3_ratio"] = np.where(
        _dem > 0, merged["sfca3_supply"] / np.maximum(_dem, 1e-9), 0.0
    ).astype(np.float32)
    merged = merged.drop(columns=["_w", "_wp", "_wd_km", "_w_km"])
    gdf_merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")

    total_pop = float(merged["population"].sum())
    pop_with_access = float(merged["pop_with_access"].sum())
    stats = {
        "n_facilities": total_n_facilities,
        "total_machines": total_machines,
        "capacity_per_machine_per_year": capacity_per_machine_per_year,
        "total_national_capacity": total_capacity,
        "total_rt_demand": total_rt_demand,
        "total_rt_treated": total_rt_treated,
        "total_rt_untreated": total_rt_demand - total_rt_treated,
        "unused_capacity": max(total_capacity - total_rt_treated, 0.0),
        "pop_with_access": pop_with_access,
        "mean_access_probability": pop_with_access / total_pop if total_pop > 0 else 0.0,
        "geo_accessible_demand": total_geo_accessible,
        "demand_weighted_access": total_geo_accessible / total_rt_demand if total_rt_demand > 0 else 0.0,
        "total_cancer_excl_nmsc": total_cancer_excl_nmsc if total_cancer_excl_nmsc > 0 else None,
        "n_hexagons": len(merged),
        "skipped_countries": skipped_countries,
        "proxy_countries": proxy_countries,
        "tt_mode": tt_mode,
        "model": model,
        "allocation": allocation,
        "rtu_set": rtu_set,
        "capacity_basis": ("fractions" if fractions_per_linac is not None else "patients"),
    }
    if any_fractions:
        fpl = fractions_per_linac
        stats.update({
            "total_fractions_demand": total_fd,
            "total_fractions_treated": total_ft,
            "total_fractions_untreated": total_fd - total_ft,
            "mean_fractions_per_course": total_fd / total_rt_demand if total_rt_demand > 0 else 0.0,
            "fractions_per_linac": fpl,
            "fractions_capacity": total_fcap if total_fcap > 0 else None,
            # Region-level machine equivalents use the stated fractions throughput when
            # given, else the benchmark 450 × this region's mean course length.
            "linac_equivalents_deficit": (
                (total_fd - total_ft) / (fpl if fpl else capacity_per_machine_per_year * (total_fd / total_rt_demand))
                if total_rt_demand > 0 and total_fd > 0 else None
            ),
        })
    return gdf_merged, stats


def combine_allocations(gdf_nf, stats_nf, gdf_fca, stats_fca):
    """Attach the capped-3SFCA ("proportional") result to the nearest-first one.

    Adds ``rt_treated_fca``, ``rt_untreated_fca`` and
    ``capacity_limited_probability_fca`` columns (matched on h3) and
    ``stats["fca"]`` with the capped-3SFCA totals.
    """
    fca = gdf_fca.set_index("h3")
    out = gdf_nf.copy()
    for c in ("rt_treated", "rt_untreated", "capacity_limited_probability"):
        out[f"{c}_fca"] = out["h3"].map(fca[c]).astype(np.float32)
    stats = dict(stats_nf)
    stats["fca"] = {k: stats_fca.get(k) for k in
                    ("total_rt_treated", "total_rt_untreated", "unused_capacity", "allocation")}
    return out, stats


def allocation_view(gdf_out, stats, which: str):
    """The result as seen under one allocation rule: "nf" (nearest-first) or
    "fca" (capped 3SFCA), with that rule's treated/untreated columns and totals
    in the standard column names."""
    if which != "fca" or "rt_treated_fca" not in gdf_out.columns:
        return gdf_out, stats
    g = gdf_out.copy()
    for c in ("rt_treated", "rt_untreated", "capacity_limited_probability"):
        g[c] = g[f"{c}_fca"]
    s = dict(stats)
    s.update({k: v for k, v in stats["fca"].items() if v is not None})
    s["allocation"] = "proportional"
    return g, s


# ---------------------------------------------------------------------------
# Floating-catchment benchmarks
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=12)
def _compute_fca_cached(
    country: str,
    iso3: str,
    linac_locs: tuple,
    variant: str,
    model: str,
    lambda_km: float,
    weibull_k: float,
    max_distance_km: float,
    cutoff_km: Optional[float],
    capacity_per_machine_per_year: float,
    rt_method: str,
    rt_fraction: float,
    h3_res: int,
    snap_linacs_to_hex: bool,
    custom_rtu: tuple,
    rtu_set: str,
    tt_mode: Optional[str] = None,
    tt_cache_key: Optional[str] = None,
    gravity_beta: float = 1.5,
):
    """2SFCA / E2SFCA / 3SFCA / gravity ratios and implied deficits for a single country at the display resolution."""
    gdf = _load_pop(country, h3_res)
    demand, _, _ = _build_demand(gdf, iso3, rt_method, rt_fraction, custom_rtu, rtu_set)
    tt = _load_tt_matrix(tt_cache_key, tt_mode) if tt_cache_key and tt_mode else None
    return compute_fca(
        gdf, list(linac_locs), variant=variant, model=model, lambda_km=lambda_km,
        weibull_k=weibull_k, max_distance_km=max_distance_km, cutoff_km=cutoff_km,
        capacity_per_machine_per_year=capacity_per_machine_per_year, demand=demand,
        snap_linacs_to_hex=snap_linacs_to_hex, h3_resolution=h3_res,
        travel_time_matrix=tt, gravity_beta=gravity_beta,
    )


# ---------------------------------------------------------------------------
# Lookups for the Data / Capacity tabs
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=32)
def _load_dirac(country: str, merge_colocated: bool = False):
    try:
        if is_region(country):
            return load_linacs_for_region(country)
        return load_linacs_from_dirac_db(country, merge_colocated=merge_colocated)
    except (ValueError, FileNotFoundError) as e:
        return None, str(e)


@st.cache_data(show_spinner=False, max_entries=64)
def _facility_quality(country: str):
    try:
        return facility_quality_report(country)
    except (ValueError, FileNotFoundError):
        return None


@st.cache_data(show_spinner=False, max_entries=32)
def _data_tab_cancer(iso3: str) -> pd.DataFrame:
    """Cancer incidence table for the Data tab: one row per cancer type.

    Returns all types including aggregates. Callers should separate out aggregate
    rows (where Cancer type normalised is in _AGGREGATE_CANCER_KEYS) for display.
    """
    all_cancer_types = get_cancer_types()
    all_with_derived = all_cancer_types + DERIVED_CANCER_TYPES
    cases = get_national_cases(iso3, all_with_derived)
    # Use "All cancers" figure as denominator for % of total
    all_cancers_total = next(
        (v for k, v in cases.items() if k.strip().lower() == "all cancers"),
        sum(cases.values()),
    )
    rows = []
    for cancer in sorted(all_with_derived, key=lambda c: -cases.get(c, 0.0)):
        n = cases.get(cancer, 0.0)
        rows.append({
            "Cancer type": cancer,
            "New cases": int(round(n)),
            "% of All Cancers": round(100 * n / all_cancers_total, 1) if all_cancers_total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=6)
def _run_convergence_cached(country: str, iso3: str):
    """Resolution-convergence table for a country (cached; fetches TT on miss)."""
    from analysis.convergence import run_convergence
    _tt = st.secrets.get("traveltime", {}) if hasattr(st, "secrets") else {}
    return run_convergence(country, iso3, _tt.get("app_id", ""), _tt.get("api_key", ""))


@st.cache_data(show_spinner=False)
def _data_tab_rt_need(iso3: str, rtu_set: str = DEFAULT_RTU_SET) -> dict:
    """Compute country-level RT need from optimal utilisations × incidence."""
    all_cancer_types = get_cancer_types() + DERIVED_CANCER_TYPES
    cases = get_national_cases(iso3, all_cancer_types)
    opt = get_optimal_rt_fractions(rtu_set)
    opt_norm = {k.strip().lower(): v for k, v in opt.items()}
    total_rt = 0.0
    for cancer, n in cases.items():
        if cancer.strip().lower() in _AGGREGATE_CANCER_KEYS:
            continue
        frac = opt_norm.get(cancer.strip().lower(), 0.0)
        total_rt += n * frac
    total_cancer_excl_nmsc = cases.get("All cancers excl. NMSC", 0.0)
    return {"total_rt_cases": total_rt, "total_cancer_excl_nmsc": total_cancer_excl_nmsc}


# ---------------------------------------------------------------------------
# Simplified dashboard: the paper's main configuration, per member country
# ---------------------------------------------------------------------------

DASHBOARD_RES = 5                  # compute resolution of the paper; maps aggregate from it
DASHBOARD_EXP_SCALE_MIN = 60.0     # exponential decay scale (sensitivity analysis)
DASHBOARD_POWER_PLATEAU_MIN = 10.0  # power-law plateau x0 and exponent beta = 1
DASHBOARD_POWER_BETA = 1.0


def compute_members(alpha2_list, model: str = "gaussian", t_max: float = 120.0,
                    capacity: float = 450.0, rtu_set: str = DEFAULT_RTU_SET, linacs: Optional[dict] = None,
                    custom_rtu: tuple = (), allocation: str = "proportional", progress_callback=None):
    """Capped-3SFCA access for each member country at resolution 5, on stored
    local-08:00 driving times (calibrated straight-line proxy where TravelTime
    has no coverage) and the optimal RTU of *rtu_set* (CCORE 2012 by default).
    *linacs* maps alpha-2 codes to ((lat, lon, n_linacs), ...) replacing that
    country's DIRAC facilities. *custom_rtu* ((cancer, percent), ...) replaces
    the optimal RTUs. *allocation* is "proportional" (capped 3SFCA) or
    "nearest_first".

    *t_max* is the driving-time limit in minutes: the catchment of the step and
    Gaussian models and the cut-off of the exponential (scale 60 min) and power
    law (plateau 10 min, beta 1).

    Returns (hexes, countries):
      hexes      h3, country, population, D, A, A_G per resolution-5 hexagon
      countries  one row per country: country, iso3, D, capacity, A_C, A_G, A,
                 n_facilities, n_linacs, time_source
    """
    scale, shape, cutoff = None, 2.0, None
    if model == "exponential":
        scale, cutoff = DASHBOARD_EXP_SCALE_MIN, t_max
    elif model == "power":
        scale, shape, cutoff = DASHBOARD_POWER_PLATEAU_MIN, DASHBOARD_POWER_BETA, t_max
    hexes, rows = [], []
    for i, a2 in enumerate(alpha2_list):
        res = _compute_access_one_country_for_region(
            a2, scale if scale is not None else t_max, model, t_max, capacity,
            "custom" if custom_rtu else "optimal", 0.25,
            DASHBOARD_RES, False, shape, tuple(custom_rtu), rtu_set=rtu_set, allocation=allocation, cutoff_km=cutoff,
            tt_mode="driving", linac_locs=(linacs or {}).get(a2),
        )
        if progress_callback is not None:
            progress_callback(i + 1, len(alpha2_list))
        if res is None:
            continue
        g, s = res
        obj = pycountry.countries.get(alpha_2=a2)
        d = g["rt_demand"].to_numpy(np.float64)
        hexes.append(pd.DataFrame({
            "h3": g["h3"].to_numpy(), "country": obj.name,
            "population": g["population"].to_numpy(np.float64), "D": d,
            "A": g["rt_treated"].to_numpy(np.float64),
            "A_G": g["access_probability"].to_numpy(np.float64) * d,
        }))
        cap = float(s.get("total_national_capacity", 0.0))
        rows.append({
            "country": obj.name, "iso3": obj.alpha_3, "D": float(d.sum()), "capacity": cap,
            "A_C": min(float(d.sum()), cap), "A_G": float(hexes[-1]["A_G"].sum()),
            "A": float(s.get("total_rt_treated", 0.0)), "n_facilities": int(s.get("n_facilities", 0)),
            "n_linacs": float(s.get("total_machines", 0)), "time_source": s.get("time_source"),
        })
    if not hexes:
        raise ValueError("No member country has population and GLOBOCAN data")
    return pd.concat(hexes, ignore_index=True), pd.DataFrame(rows)


def access_profile(a_c: float, a_g: float, a: float, thr: float = 0.8) -> str:
    """Access profile of a country from its capacity-only, geography-only and RadMaps
    shares of demand (paper, Section 3.2), at a nominal threshold *thr*."""
    if a >= thr:
        return "Strong access"
    if a_c < thr and a_g < thr:
        return "Doubly constrained"
    return "Capacity constrained" if a_c < a_g else "Geography constrained"


def _country_inputs(alpha2: str, model: str, t_max: float, custom_rtu: tuple = (), base_locs=None):
    """Shared set-up for single-country calculations at resolution 5: (gdf, demand,
    facilities, driving-time matrix or None, km_mode, decay keyword arguments).
    *km_mode* is True where TravelTime has no coverage and distances are used."""
    obj = pycountry.countries.get(alpha_2=alpha2)
    gdf = _load_pop(obj.name, DASHBOARD_RES)
    dem, _, _ = _build_demand(gdf, obj.alpha_3, "custom" if custom_rtu else "optimal", 0.25, tuple(custom_rtu))
    if dem is None:
        raise ValueError(f"{obj.name} has no GLOBOCAN estimate")
    if base_locs is None:
        try:
            base_locs, _ = load_linacs_from_dirac_db(obj.name)
        except (ValueError, FileNotFoundError):
            base_locs = []
    locs = [tuple(x) for x in base_locs]
    tt = _regional_tt_matrix(obj.alpha_3, locs, list(gdf["h3"]), "driving") if locs else None
    km_mode = tt is None and obj.alpha_3 in {"CHN", "RUS", "PRK", "KOR"}
    scale, shape, cutoff = None, 2.0, None
    if model == "exponential":
        scale, cutoff = DASHBOARD_EXP_SCALE_MIN, t_max
    elif model == "power":
        scale, shape, cutoff = DASHBOARD_POWER_PLATEAU_MIN, DASHBOARD_POWER_BETA, t_max
    thr = minutes_to_proxy_km(t_max) if km_mode else t_max
    if km_mode:
        scale, cutoff = minutes_to_proxy_km(scale), minutes_to_proxy_km(cutoff)
    kw = dict(model=model, lambda_km=scale if scale is not None else thr, cutoff_km=cutoff, max_distance_km=thr,
              weibull_k=shape)
    return gdf, dem, locs, tt, km_mode, kw


def centre_loads(alpha2: str, model: str = "gaussian", t_max: float = 120.0, capacity: float = 450.0,
                 base_locs=None, custom_rtu: tuple = (), allocation: str = "proportional") -> pd.DataFrame:
    """Per centre (in the order of *base_locs*, or DIRAC): capacity, patients
    treated and, for capped 3SFCA, the demand Q_f directed to it."""
    gdf, dem, locs, tt, _, kw = _country_inputs(alpha2, model, t_max, custom_rtu, base_locs)
    if not locs:
        return pd.DataFrame(columns=["lat", "lon", "capacity", "treated", "demand"])
    _, st_ = compute_accessibility(gdf, locs, capacity_per_machine_per_year=capacity, demand=dem,
                                   h3_resolution=DASHBOARD_RES, travel_time_matrix=tt, allocation=allocation, **kw)
    q = st_["facility_demand"]
    return pd.DataFrame({"lat": [x[0] for x in locs], "lon": [x[1] for x in locs],
                         "capacity": st_["facility_capacity"], "treated": st_["facility_treated"],
                         "demand": q if q is not None else [np.nan] * len(locs)})


def plan_centres(alpha2: str, model: str = "gaussian", t_max: float = 120.0, capacity: float = 450.0,
                 base_locs=None, n_linacs: Optional[int] = 10, linacs_per_step: int = 1, strategy: str = "deficit",
                 snap_min: Optional[float] = None, custom_rtu: tuple = (), target_access: Optional[float] = None,
                 max_steps: int = 200, allocation: str = "proportional", progress_callback=None):
    """Greedy placement of new linacs in one country (resolution 5; *allocation*
    "proportional" for capped 3SFCA or "nearest_first").

    At each step the hexagon with the largest unmet demand (*strategy* "deficit")
    or the largest demand out of geographic reach ("isolation") receives
    *linacs_per_step* linacs: added to the existing centre nearest to it when one
    lies within *snap_min* minutes, otherwise as a new centre at its centroid.
    New centres use the calibrated straight-line equivalent of driving time.

    Stops after *n_linacs* linacs or, with *target_access* (fraction of demand),
    once national access reaches it; never after more than *max_steps* steps.

    Returns (before, steps, locs): national D, A_C, A_G, A before; one dict per
    step; and the final ((lat, lon, n_linacs), ...).
    """
    gdf, dem, locs, tt, km_mode, kw = _country_inputs(alpha2, model, t_max, custom_rtu, base_locs)
    hex_ids = list(gdf["h3"])
    if tt is None and not km_mode:  # no centres yet: every centre will be new
        tt = np.zeros((len(hex_ids), 0), dtype=np.float32)
    ll = np.array([h3.cell_to_latlng(c) for c in hex_ids])

    def run():
        if not locs:
            z = np.zeros(len(hex_ids))
            return pd.DataFrame({"rt_untreated": dem, "access_probability": z}), 0.0
        g, st_ = compute_accessibility(gdf, locs, capacity_per_machine_per_year=capacity, demand=dem,
                                       h3_resolution=DASHBOARD_RES, travel_time_matrix=tt, allocation=allocation,
                                       **kw)
        return g, float(st_["total_rt_treated"])

    def metrics(g, a):
        D = float(dem.sum())
        n = float(sum(x[2] for x in locs))
        return {"D": D, "A_C": min(D, n * capacity), "A_G": float((g["access_probability"] * dem).sum()), "A": a,
                "n_linacs": n}

    g, a = run()
    before = metrics(g, a)
    steps = []
    added, a_prev = 0, a
    for i in range(max_steps):
        if target_access is not None and a >= target_access * before["D"]:
            break
        if target_access is None and added >= n_linacs:
            break
        k = linacs_per_step if target_access is not None else min(linacs_per_step, n_linacs - added)
        score = (g["rt_untreated"].to_numpy(np.float64) if strategy == "deficit"
                 else dem * (1.0 - g["access_probability"].to_numpy(np.float64)))
        h = int(np.argmax(score))
        if score[h] <= 0:
            break
        j = None
        if snap_min and locs:
            if km_mode:
                d = np.array([_great_circle_km(ll[h:h + 1], la, lo)[0] for la, lo, _ in locs])
                j = int(np.argmin(d)) if d.min() <= minutes_to_proxy_km(snap_min) else None
            else:
                j = int(np.argmin(tt[h])) if tt.shape[1] and tt[h].min() <= snap_min else None
        if j is not None:
            la, lo, n = locs[j]
            locs[j] = (la, lo, n + k)
            where = "existing centre"
        else:
            la, lo = float(ll[h, 0]), float(ll[h, 1])
            locs.append((la, lo, float(k)))
            if not km_mode:
                tt = np.hstack([tt, proxy_minutes_column(hex_ids, la, lo)[:, None]])
            where = "new centre"
        g, a = run()
        added += k
        m = metrics(g, a)
        steps.append({"step": i + 1, "lat": round(la, 4), "lon": round(lo, 4), "where": where,
                      "linacs_added": k, "treated": a, "additional": a - a_prev,
                      "A/D (%)": 100 * a / m["D"], "deficit": m["D"] - a})
        a_prev = a
        if progress_callback is not None:
            progress_callback(i + 1, 100 * a / m["D"], added)
    return before, steps, tuple(locs)
