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
from data.linacs import load_linacs_from_dirac_db, load_linacs_for_region
from data.regions import is_region, get_region, REGIONS, REGION_GLOBOCAN_CODES
from data.cancer import (
    get_cancer_types, apportion_cancer_to_h3, get_national_cases,
    get_optimal_rt_fractions, DERIVED_CANCER_TYPES, XARRAY_PATH,
)
from data.travel_time import CACHE_DIR as _TT_CACHE_DIR
from analysis.accessibility import compute_accessibility, aggregate_access_gdf

# Disk cache for apportioned regional cancer GeoDataFrames.  Mirrors the region
# *population* parquet cache (data.population.REGION_CACHE_DIR) so that the
# expensive per-country apportionment loop survives app restarts, redeploys, and
# LRU eviction of the in-memory st.cache_data — not just a single session.
_REGION_CANCER_CACHE_DIR = Path("H3_region_cache")


def _region_cancer_cache_path(globocan_code: str, resolution: int,
                              cancers: tuple, use_actual: bool) -> Path:
    """Stable on-disk path for one (region, resolution, cancers, actual) result."""
    key = json.dumps({"c": sorted(cancers), "a": bool(use_actual)}, sort_keys=True)
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
):
    # At coarse display resolutions the hex centroid is a poor proxy for where
    # people live (res 3 ≈ 12,400 km²). Compute at ≥ res 5 and aggregate down.
    compute_res = h3_res if (region_flag or h3_res >= 5) else 5
    gdf = _load_pop_region(country, compute_res) if region_flag else _load_pop(country, compute_res)

    # Build RT demand per hex from cancer data
    demand = None
    total_cancer_excl_nmsc = None
    try:
        if rt_method in ("optimal", "custom"):
            all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES
            cancer_gdf = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False)
            excl_col = "All cancers excl. NMSC_incidence"
            if excl_col in cancer_gdf.columns:
                total_cancer_excl_nmsc = float(cancer_gdf[excl_col].clip(lower=0).sum())
            if rt_method == "optimal":
                rt_cols = [
                    c for c in cancer_gdf.columns
                    if c.endswith("_optimal_rt")
                    and c[:-len("_optimal_rt")].strip().lower() not in _AGGREGATE_CANCER_KEYS
                ]
                if rt_cols:
                    demand = cancer_gdf[rt_cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)
            else:  # custom — use incidence columns × user-supplied fractions
                _custom_fracs = {k.strip().lower(): v / 100.0 for k, v in custom_rtu}
                demand_arr = None
                for cancer in all_cancers:
                    if cancer.strip().lower() in _AGGREGATE_CANCER_KEYS:
                        continue
                    col = f"{cancer}_incidence"
                    if col not in cancer_gdf.columns:
                        continue
                    frac = _custom_fracs.get(cancer.strip().lower(), 0.0)
                    inc = cancer_gdf[col].clip(lower=0).to_numpy(np.float64)
                    demand_arr = inc * frac if demand_arr is None else demand_arr + inc * frac
                if demand_arr is not None:
                    demand = demand_arr
        else:  # proportional — use All cancers excl. NMSC × rt_fraction
            cancer_gdf = apportion_cancer_to_h3(
                gdf, iso3, ["All cancers excl. NMSC"], use_actual_rt=False
            )
            col = "All cancers excl. NMSC_incidence"
            if col in cancer_gdf.columns:
                total_cancer_excl_nmsc = float(cancer_gdf[col].clip(lower=0).sum())
                demand = (cancer_gdf[col].clip(lower=0) * rt_fraction).to_numpy(np.float64)
    except Exception:
        pass  # fallback: compute_accessibility uses raw population

    gdf_out, stats = compute_accessibility(
        gdf,
        list(linac_locs),
        lambda_km=lambda_km,
        model=model,
        max_distance_km=max_distance_km,
        weibull_k=weibull_k,
        capacity_per_machine_per_year=capacity_per_machine_per_year,
        demand=demand,
        snap_linacs_to_hex=snap_linacs_to_hex,
        h3_resolution=compute_res,
    )
    if compute_res != h3_res:
        gdf_out = aggregate_access_gdf(gdf_out, h3_res)
        stats["n_hexagons"] = len(gdf_out)
        stats["compute_resolution"] = compute_res
    stats["total_cancer_excl_nmsc"] = total_cancer_excl_nmsc
    return gdf_out, stats


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
):
    """Like _compute_access but loads a pre-computed travel time matrix from disk."""
    gdf = _load_pop_region(country, h3_res) if region_flag else _load_pop(country, h3_res)

    demand = None
    total_cancer_excl_nmsc = None
    try:
        if rt_method in ("optimal", "custom"):
            all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES
            cancer_gdf = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False)
            excl_col = "All cancers excl. NMSC_incidence"
            if excl_col in cancer_gdf.columns:
                total_cancer_excl_nmsc = float(cancer_gdf[excl_col].clip(lower=0).sum())
            if rt_method == "optimal":
                rt_cols = [
                    c for c in cancer_gdf.columns
                    if c.endswith("_optimal_rt")
                    and c[:-len("_optimal_rt")].strip().lower() not in _AGGREGATE_CANCER_KEYS
                ]
                if rt_cols:
                    demand = cancer_gdf[rt_cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)
            else:  # custom
                _custom_fracs = {k.strip().lower(): v / 100.0 for k, v in custom_rtu}
                demand_arr = None
                for cancer in all_cancers:
                    if cancer.strip().lower() in _AGGREGATE_CANCER_KEYS:
                        continue
                    col = f"{cancer}_incidence"
                    if col not in cancer_gdf.columns:
                        continue
                    frac = _custom_fracs.get(cancer.strip().lower(), 0.0)
                    inc = cancer_gdf[col].clip(lower=0).to_numpy(np.float64)
                    demand_arr = inc * frac if demand_arr is None else demand_arr + inc * frac
                if demand_arr is not None:
                    demand = demand_arr
        else:
            cancer_gdf = apportion_cancer_to_h3(
                gdf, iso3, ["All cancers excl. NMSC"], use_actual_rt=False
            )
            col = "All cancers excl. NMSC_incidence"
            if col in cancer_gdf.columns:
                total_cancer_excl_nmsc = float(cancer_gdf[col].clip(lower=0).sum())
                demand = (cancer_gdf[col].clip(lower=0) * rt_fraction).to_numpy(np.float64)
    except Exception:
        pass

    # Load travel time matrix from disk cache (must already exist)
    cache_file = _TT_CACHE_DIR / f"{tt_cache_key}_{tt_mode}.npz"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Travel time cache not found: {cache_file.name}. "
            "Press 'Fetch TravelTime Data' before computing accessibility."
        )
    tt_matrix = np.load(cache_file)["matrix"]

    gdf_out, stats = compute_accessibility(
        gdf,
        list(linac_locs),
        lambda_km=lambda_km,
        model=model,
        max_distance_km=max_distance_km,
        weibull_k=weibull_k,
        capacity_per_machine_per_year=capacity_per_machine_per_year,
        demand=demand,
        snap_linacs_to_hex=snap_linacs_to_hex,
        h3_resolution=h3_res,
        travel_time_matrix=tt_matrix,
    )
    stats["total_cancer_excl_nmsc"] = total_cancer_excl_nmsc
    return gdf_out, stats


@st.cache_data(show_spinner=False, max_entries=300)
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
):
    """Cached per-country computation for use in per-country regional aggregation.

    Returns (gdf_subset, c_stats) or None if the country should be skipped
    (no population data or no GLOBOCAN data).  Countries with no LINACs return
    a gdf with zero access so their unmet demand is still counted.
    """
    country_obj = pycountry.countries.get(alpha_2=alpha2)
    if country_obj is None:
        return None

    c_name = country_obj.name
    c_iso3 = country_obj.alpha_3
    all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES
    _agg_keys = _AGGREGATE_CANCER_KEYS

    # Compute at ≥ res 4 so centroid-to-facility distances are meaningful,
    # then aggregate down to the display resolution.
    compute_res = max(h3_res, 4)

    try:
        gdf = _load_pop(c_name, compute_res)
    except Exception:
        return None

    demand = None
    c_cancer_excl_nmsc = None
    try:
        if rt_method in ("optimal", "custom"):
            cancer_gdf = apportion_cancer_to_h3(gdf, c_iso3, all_cancers, use_actual_rt=False)
            excl_col = "All cancers excl. NMSC_incidence"
            if excl_col in cancer_gdf.columns:
                c_cancer_excl_nmsc = float(cancer_gdf[excl_col].clip(lower=0).sum())
            if rt_method == "optimal":
                rt_cols = [
                    c for c in cancer_gdf.columns
                    if c.endswith("_optimal_rt")
                    and c[:-len("_optimal_rt")].strip().lower() not in _agg_keys
                ]
                if rt_cols:
                    demand = cancer_gdf[rt_cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)
            else:
                _custom_fracs = {k.strip().lower(): v / 100.0 for k, v in custom_rtu}
                demand_arr = None
                for cancer in all_cancers:
                    if cancer.strip().lower() in _agg_keys:
                        continue
                    col = f"{cancer}_incidence"
                    if col not in cancer_gdf.columns:
                        continue
                    frac = _custom_fracs.get(cancer.strip().lower(), 0.0)
                    inc = cancer_gdf[col].clip(lower=0).to_numpy(np.float64)
                    demand_arr = inc * frac if demand_arr is None else demand_arr + inc * frac
                if demand_arr is not None:
                    demand = demand_arr
        else:
            cancer_gdf = apportion_cancer_to_h3(gdf, c_iso3, ["All cancers excl. NMSC"], use_actual_rt=False)
            col = "All cancers excl. NMSC_incidence"
            if col in cancer_gdf.columns:
                c_cancer_excl_nmsc = float(cancer_gdf[col].clip(lower=0).sum())
                demand = (cancer_gdf[col].clip(lower=0) * rt_fraction).to_numpy(np.float64)
    except Exception:
        return None  # no GLOBOCAN data

    if demand is None:
        return None

    try:
        c_locs, _ = load_linacs_from_dirac_db(c_name)
        has_linacs = bool(c_locs)
    except (ValueError, FileNotFoundError):
        c_locs = []
        has_linacs = False

    if has_linacs:
        gdf_out, c_stats = compute_accessibility(
            gdf, c_locs,
            lambda_km=lambda_km, model=model,
            max_distance_km=max_distance_km, weibull_k=weibull_k,
            capacity_per_machine_per_year=capacity_per_machine_per_year,
            demand=demand, snap_linacs_to_hex=snap_linacs_to_hex,
            h3_resolution=compute_res,
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
        c_stats = {"n_facilities": 0, "total_machines": 0,
                   "total_rt_demand": float(demand.sum()), "total_rt_treated": 0.0}

    gdf_out["country"] = c_name
    if compute_res != h3_res:
        gdf_out = aggregate_access_gdf(gdf_out, h3_res)
        gdf_out["country"] = c_name

    c_stats["cancer_excl_nmsc"] = c_cancer_excl_nmsc or 0.0
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
):
    """Aggregate per-country RT access results into a single regional GeoDataFrame.

    progress_callback(done, total) is called after each country completes.
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
    total_cancer_excl_nmsc = 0.0

    for i, alpha2 in enumerate(alpha2_list):
        result = _compute_access_one_country_for_region(
            alpha2, lambda_km, model, max_distance_km,
            capacity_per_machine_per_year, rt_method, rt_fraction,
            h3_res, snap_linacs_to_hex, weibull_k, custom_rtu,
        )
        if progress_callback is not None:
            progress_callback(i + 1, n_total)
        if result is None:
            _c_obj = pycountry.countries.get(alpha_2=alpha2)
            skipped_countries.append(_c_obj.name if _c_obj else alpha2)
            continue
        gdf_out, c_stats = result
        total_rt_demand += c_stats["total_rt_demand"]
        total_rt_treated += c_stats.get("total_rt_treated", 0.0)
        total_n_facilities += c_stats.get("n_facilities", 0)
        total_machines += c_stats.get("total_machines", 0)
        total_cancer_excl_nmsc += c_stats.get("cancer_excl_nmsc", 0.0)
        all_gdfs.append(gdf_out)

    if not all_gdfs:
        raise ValueError(f"No country data found for region {region_name!r}")

    combined = pd.concat(
        [g[["h3", "population", "geometry", "country", "nearest_linac_km",
            "access_probability", "capacity_limited_probability",
            "rt_demand", "rt_treated", "rt_untreated", "pop_with_access"]]
         for g in all_gdfs],
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

    grp = combined.groupby("h3", sort=False)
    merged = grp.agg(
        population=("population", "sum"),
        rt_demand=("rt_demand", "sum"),
        rt_treated=("rt_treated", "sum"),
        rt_untreated=("rt_untreated", "sum"),
        pop_with_access=("pop_with_access", "sum"),
        _w=("_w", "sum"),
        _wp=("_wp", "sum"),
        _wd_km=("_wd_km", "sum"),
        _w_km=("_w_km", "sum"),
        geometry=("geometry", "first"),
    ).reset_index()

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
    _dem = merged["rt_demand"].to_numpy(np.float64)
    merged["capacity_limited_probability"] = np.where(
        _dem > 0, merged["rt_treated"] / np.maximum(_dem, 1e-9), 0.0
    ).astype(np.float32)
    merged = merged.drop(columns=["_w", "_wp", "_wd_km", "_w_km"])
    gdf_merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")

    total_pop = float(merged["population"].sum())
    pop_with_access = float(merged["pop_with_access"].sum())
    stats = {
        "n_facilities": total_n_facilities,
        "total_machines": total_machines,
        "total_national_capacity": total_machines * capacity_per_machine_per_year,
        "total_rt_demand": total_rt_demand,
        "total_rt_treated": total_rt_treated,
        "total_rt_untreated": total_rt_demand - total_rt_treated,
        "pop_with_access": pop_with_access,
        "mean_access_probability": pop_with_access / total_pop if total_pop > 0 else 0.0,
        "total_cancer_excl_nmsc": total_cancer_excl_nmsc if total_cancer_excl_nmsc > 0 else None,
        "n_hexagons": len(merged),
        "skipped_countries": skipped_countries,
    }
    return gdf_merged, stats


@st.cache_data(show_spinner=False, max_entries=32)
def _load_dirac(country: str):
    try:
        if is_region(country):
            return load_linacs_for_region(country)
        return load_linacs_from_dirac_db(country)
    except (ValueError, FileNotFoundError) as e:
        return None, str(e)


_AGGREGATE_CANCER_KEYS = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}


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
def _data_tab_rt_need(iso3: str) -> dict:
    """Compute country-level RT need from optimal utilisations × incidence."""
    all_cancer_types = get_cancer_types() + DERIVED_CANCER_TYPES
    cases = get_national_cases(iso3, all_cancer_types)
    opt = get_optimal_rt_fractions()
    opt_norm = {k.strip().lower(): v for k, v in opt.items()}
    total_rt = 0.0
    for cancer, n in cases.items():
        if cancer.strip().lower() in _AGGREGATE_CANCER_KEYS:
            continue
        frac = opt_norm.get(cancer.strip().lower(), 0.0)
        total_rt += n * frac
    total_cancer_excl_nmsc = cases.get("All cancers excl. NMSC", 0.0)
    return {"total_rt_cases": total_rt, "total_cancer_excl_nmsc": total_cancer_excl_nmsc}
