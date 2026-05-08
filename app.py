"""
RadMaps — Interactive H3 Map (Streamlit)

Run with:
    streamlit run app.py

Map types
---------
Population Density    — Kontur H3 population per hexagon (log scale)
Cancer Incidence      — Estimated cases per hexagon (proportional to pop)
Radiotherapy Demand   — Cancer cases requiring RT per hexagon
Radiotherapy Access   — P(patient can access a LINAC) per hexagon
Nearest Linac         — Distance (km) to the closest LINAC facility
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
import pydeck as pdk
import pycountry
import streamlit as st

from data.population import load_population_at_resolution, load_region_population
from data.linacs import load_linacs_from_dirac_db, load_linacs_for_region
from data.regions import is_region, get_region, REGIONS, REGION_GLOBOCAN_CODES
from data.cancer import (
    get_cancer_types, apportion_cancer_to_h3, has_globocan_data,
    get_national_cases, get_optimal_rt_fractions, DERIVED_CANCER_TYPES,
    XARRAY_PATH,
)
from analysis.accessibility import compute_accessibility
from data.travel_time import compute_travel_time_matrix, CACHE_DIR as _TT_CACHE_DIR, MAX_TRAVEL_TIME_BY_RES as _TT_MAX_BY_RES, TT_SUPPORTED_RESOLUTIONS as _TT_SUPPORTED_RES


# ---------------------------------------------------------------------------
# Streamlit version capabilities
# ---------------------------------------------------------------------------

_ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2])
# on_select for st.pydeck_chart was added in Streamlit 1.35
_PYDECK_CLICK_SUPPORTED = _ST_VERSION >= (1, 35)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RadMaps",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] > div { font-size: 1.5rem !important; }
    [data-testid="stMetricLabel"] > div { font-size: 1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _apply_app_dark_mode(enabled: bool) -> None:
    """Inject CSS to switch the entire app to a dark background."""
    if not enabled:
        return
    st.markdown(
        """
        <style>
        /* Main content and sidebar backgrounds */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background-color: #1a1d23 !important;
            color: #fafafa !important;
        }
        /* Tabs */
        [data-testid="stTabs"] button, .stTabs [data-baseweb="tab"] {
            color: #fafafa !important;
        }
        /* Text, labels, captions */
        p, span, label, div, h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"], .stCaption {
            color: #fafafa !important;
        }
        /* Metric values and labels */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            color: #fafafa !important;
        }
        /* Dataframe */
        [data-testid="stDataFrame"] {
            background-color: #1a1d23 !important;
        }
        /* White boxes with black text — must beat broad span/div rule above */
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div *,
        [data-baseweb="input"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        /* Expander */
        [data-testid="stExpander"] {
            background-color: #1a1d23 !important;
            border-color: #3a3d47 !important;
        }
        /* Dividers */
        hr { border-color: #3a3d47 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_VIRIDIS = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
]


def _viridis_rgb(t: float) -> List[int]:
    t = float(np.clip(t, 0, 1))
    n = len(_VIRIDIS) - 1
    i = min(int(t * n), n - 1)
    lo, hi = _VIRIDIS[i], _VIRIDIS[i + 1]
    f = t * n - i
    return [int(lo[j] + f * (hi[j] - lo[j])) for j in range(3)]


def _rdylgn_rgb(t: float) -> List[int]:
    stops = [
        [215, 25, 28],
        [253, 174, 97],
        [255, 255, 191],
        [166, 217, 106],
        [26, 150, 65],
    ]
    t = float(np.clip(t, 0, 1))
    n = len(stops) - 1
    i = min(int(t * n), n - 1)
    lo, hi = stops[i], stops[i + 1]
    f = t * n - i
    return [int(lo[j] + f * (hi[j] - lo[j])) for j in range(3)]


def _rdylgn_reversed_rgb(t: float) -> List[int]:
    return _rdylgn_rgb(1.0 - t)


# Named colormaps available to users
COLORMAPS = {
    "Purple → Yellow (Viridis)": _viridis_rgb,
    "Red → Green": _rdylgn_rgb,
    "Green → Red": _rdylgn_reversed_rgb,
}

# Binary colourmap is handled separately (needs threshold) — sentinel value in dict
BINARY_CMAP_NAME = "Binary (Green / Red threshold)"

_DEFAULT_CMAP = {
    "Population Density": "Green → Red",
    "Cancer Incidence": "Green → Red",
    "Radiotherapy Demand": "Green → Red",
    "Radiotherapy Access": "Green → Red",
    "Nearest Linac": "Green → Red",
}


def _apply_colormap_fixed(
    values: np.ndarray,
    cmap_fn,
    vmin: float,
    vmax: float,
    alpha: int = 160,
) -> List[List[int]]:
    if vmax <= vmin:
        return [[128, 128, 128, alpha]] * len(values)
    normed = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    return [cmap_fn(v) + [alpha] for v in normed]


# ---------------------------------------------------------------------------
# Colorbar
# ---------------------------------------------------------------------------

def _colorbar_fig(
    cmap_fn,
    vmin: float,
    vmax: float,
    label: str,
    log_scale: bool = False,
    text_color: str = "black",
    clamp: bool = False,
) -> plt.Figure:
    n = 256
    colors_01 = [[c / 255.0 for c in cmap_fn(i / n)] for i in range(n + 1)]
    cmap = mcolors.LinearSegmentedColormap.from_list("_cb", colors_01)
    safe_vmin = max(float(vmin), 1e-6) if log_scale else float(vmin)
    safe_vmax = max(float(vmax), safe_vmin * 1.01 + 1e-6)
    norm = (
        mcolors.LogNorm(safe_vmin, safe_vmax)
        if log_scale
        else mcolors.Normalize(safe_vmin, safe_vmax)
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(0.9, 4.0))
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_label(label, fontsize=11, color=text_color)
    cbar.ax.tick_params(labelsize=10, labelcolor=text_color, color=text_color)
    if not log_scale and safe_vmax >= 10_000:
        import math as _math
        exp = _math.floor(_math.log10(safe_vmax))
        scale = 10 ** exp
        cbar.formatter = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / scale:.1f}")
        cbar.update_ticks()
        cbar.ax.set_title(f"×10$^{{{exp}}}$", fontsize=10, color=text_color, pad=4)
    if clamp:
        fig.canvas.draw()
        texts = [t.get_text() for t in cbar.ax.get_yticklabels()]
        if len(texts) >= 2:
            if texts[0] and float(vmin) != 0.0:
                texts[0] = "< " + texts[0]
            if texts[-1]:
                texts[-1] = "> " + texts[-1]
            cbar.ax.set_yticklabels(texts, color=text_color, fontsize=10)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor(text_color)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.tight_layout(pad=0.2)
    return fig


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _selection_options() -> list[str]:
    """Return UI options: region display names first, then GLOBOCAN country names."""
    da = xr.open_dataarray(XARRAY_PATH)
    iso3_set = {str(v) for v in da.coords["ISO3"].values}
    country_names = []
    for iso3 in iso3_set:
        if iso3 in REGION_GLOBOCAN_CODES:
            continue  # handled separately as region entries
        try:
            country_names.append(pycountry.countries.get(alpha_3=iso3).name)
        except AttributeError:
            pass
    region_names = [r.display_name for r in REGIONS]
    return region_names + sorted(country_names)


# Keep old name as alias so cached calls referencing it still work
_globocan_country_list = _selection_options


@st.cache_data(show_spinner=False)
def _load_pop(country: str, h3_res: int = 8):
    return load_population_at_resolution(country, target_resolution=h3_res)


@st.cache_data(show_spinner=False)
def _load_pop_region(region_name: str, h3_res: int = 3):
    return load_region_population(region_name, target_resolution=h3_res)


@st.cache_data(show_spinner=False)
def _load_cancer(country: str, iso3: str, cancers: tuple, use_actual: bool,
                 h3_res: int = 8, region_flag: bool = False):
    gdf = _load_pop_region(country, h3_res) if region_flag else _load_pop(country, h3_res)
    return apportion_cancer_to_h3(gdf, iso3, list(cancers), use_actual_rt=use_actual)


@st.cache_data(show_spinner=False)
def _load_cancer_region_percountry(region_name: str, cancers: tuple, use_actual: bool, h3_res: int = 3):
    """Build a cancer GeoDataFrame for a region using per-country GLOBOCAN data.

    Each country's cancer cases are apportioned to its own hexes using that
    country's iso3, then all country GDFs are concatenated.  Border hex
    duplicates are resolved by keeping the row with higher total cancer value.
    """
    import geopandas as gpd
    from data.regions import get_region as _get_region

    reg = _get_region(region_name)
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
    return gpd.GeoDataFrame(combined, geometry="geometry")


@st.cache_data(show_spinner=False)
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
    gdf = _load_pop_region(country, h3_res) if region_flag else _load_pop(country, h3_res)

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
        h3_resolution=h3_res,
    )
    stats["total_cancer_excl_nmsc"] = total_cancer_excl_nmsc
    return gdf_out, stats


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
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

    try:
        gdf = _load_pop(c_name, h3_res)
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
            h3_resolution=h3_res,
        )
    else:
        gdf_out = gdf.copy()
        gdf_out["nearest_linac_km"] = np.float32(np.inf)
        gdf_out["access_probability"] = np.float32(0.0)
        gdf_out["capacity_limited_probability"] = np.float32(0.0)
        gdf_out["rt_demand"] = demand.astype(np.float32)
        gdf_out["rt_treated"] = np.float32(0.0)
        gdf_out["rt_untreated"] = demand.astype(np.float32)
        gdf_out["pop_with_access"] = np.float32(0.0)
        c_stats = {"n_facilities": 0, "total_machines": 0,
                   "total_rt_demand": float(demand.sum()), "total_rt_treated": 0.0}

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
        [g[["h3", "population", "geometry", "nearest_linac_km",
            "access_probability", "capacity_limited_probability",
            "rt_demand", "rt_treated", "rt_untreated", "pop_with_access"]]
         for g in all_gdfs],
        ignore_index=True,
    )
    combined = (
        combined
        .sort_values("access_probability", ascending=False)
        .drop_duplicates("h3")
        .reset_index(drop=True)
    )
    gdf_merged = gpd.GeoDataFrame(combined, geometry="geometry")

    total_pop = float(combined["population"].sum())
    pop_with_access = float(combined["pop_with_access"].sum())
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
        "n_hexagons": len(combined),
    }
    return gdf_merged, stats


@st.cache_data(show_spinner=False)
def _load_dirac(country: str):
    try:
        if is_region(country):
            return load_linacs_for_region(country)
        return load_linacs_from_dirac_db(country)
    except (ValueError, FileNotFoundError) as e:
        return None, str(e)


_AGGREGATE_CANCER_KEYS = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def _data_tab_optimal_rt() -> pd.DataFrame:
    """Optimal RT utilisations table for the Data tab."""
    opt = get_optimal_rt_fractions()
    return pd.DataFrame([
        {"Cancer type": k, "Optimal RT fraction": v, "Optimal RT %": f"{v:.0%}"}
        for k, v in sorted(opt.items(), key=lambda x: -x[1])
    ])


# ---------------------------------------------------------------------------
# Map layer builders
# ---------------------------------------------------------------------------

_HEX_LAYER_ID = "hex-layer"


def _build_hex_layer(df: pd.DataFrame) -> pdk.Layer:
    return pdk.Layer(
        "H3HexagonLayer",
        id=_HEX_LAYER_ID,
        data=df,
        get_hexagon="h3",
        get_fill_color="color",
        auto_highlight=True,
        pickable=True,
        opacity=0.7,
    )


_LINAC_BLUE = [30, 120, 220, 220]

# Discrete colour scale: red → orange → yellow → green → dark green (up to 5 bands)
_DISCRETE_PALETTE = [
    [220,  50,  50, 220],  # red
    [230, 120,  30, 220],  # orange
    [240, 200,  30, 220],  # yellow
    [ 60, 180,  60, 220],  # green
    [ 30, 120,  30, 220],  # dark green
]
_LINAC_COLORS = [
    [30,  120, 220, 220],  # blue
    [220, 60,  60,  220],  # red
    [60,  180, 60,  220],  # green
    [220, 140, 30,  220],  # orange
    [140, 60,  180, 220],  # purple
    [30,  180, 180, 220],  # teal
    [180, 60,  120, 220],  # pink
    [180, 180, 60,  220],  # yellow
]


def _build_linac_columns(
    facilities_df: pd.DataFrame,
    h3_res: int = 6,
    country_span_km: float = 1000.0,
    height_scale: float = 1.0,
    radius_scale: float = 1.0,
    style: str = "stacked",
    color: Optional[List[int]] = None,
) -> List[pdk.Layer]:
    """Return ColumnLayers for LINAC towers.

    style="stacked"    — co-located facilities (same H3 cell) merged into one
                         tower with proportional segments per centre.
    style="individual" — one column per facility at its own lat/lon.
    """
    if facilities_df.empty:
        return []
    hex_area_km2 = h3.average_hexagon_area(h3_res, unit="km^2")
    hex_radius_km = math.sqrt(hex_area_km2 / math.pi)
    col_radius_m = int(hex_radius_km * 1000 * 0.45 * radius_scale)
    elevation_per_linac = (
        # max(hex_radius_km * 1000 * 0.6, country_span_km * 1000 * 0.0008) * height_scale
        hex_radius_km * 1000 * 0.6* height_scale
    )

    _has_row_color = "color" in facilities_df.columns

    if style == "individual":
        rows = []
        for i, row_data in facilities_df.reset_index(drop=True).iterrows():
            _city = row_data.get("city", "") if hasattr(row_data, "get") else ""
            _row_color = (
                row_data["color"] if _has_row_color
                else color if color is not None
                else _LINAC_COLORS[i % len(_LINAC_COLORS)]
            )
            rows.append({
                "lat": float(row_data["lat"]),
                "lon": float(row_data["lon"]),
                "elevation": float(row_data["capacity"]/450) * elevation_per_linac,
                "color": _row_color,
                "tip": (
                    f"<b>{row_data['name']}</b><br/>"
                    + (f"{_city}<br/>" if _city else "")
                    + f"{int(row_data['n_linacs'])} LINAC{'s' if row_data['n_linacs'] != 1 else ''}"
                    + (f"<br/>Capacity: {int(row_data['capacity']):,} pts/yr" if "capacity" in row_data else "")
                ),
            })
        ind_df = pd.DataFrame(rows)
        return [pdk.Layer(
            "ColumnLayer",
            id="linac-columns-individual",
            data=ind_df,
            get_position="[lon, lat]",
            get_elevation="elevation",
            elevation_scale=1,
            radius=col_radius_m,
            get_fill_color="color",
            get_line_color=[0, 0, 0, 60],
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )]

    # --- stacked (segmented) mode ---
    df = facilities_df.copy()
    df["hex_id"] = df.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lon"], h3_res), axis=1)

    # tiers[i] = rows for the i-th facility slot across all hexes.
    # Within each hex, sort by _stack_order (ascending) then n_linacs (descending) so
    # that facilities with a lower _stack_order value are placed lower in the stack.
    _has_stack_order = "_stack_order" in df.columns
    tiers: dict = {}
    for hex_id, group in df.groupby("hex_id"):
        if _has_stack_order:
            group = group.sort_values(["_stack_order", "n_linacs"], ascending=[True, False]).reset_index(drop=True)
        else:
            group = group.sort_values("n_linacs", ascending=False).reset_index(drop=True)
        hc = h3.cell_to_latlng(hex_id)  # (lat, lon)
        cum = 0.0
        for i, row_data in group.iterrows():
            cum += float(row_data["n_linacs"]) * elevation_per_linac
            _city = row_data.get("city", "") if hasattr(row_data, "get") else ""
            _row_color = (
                row_data["color"] if _has_row_color
                else color if color is not None
                else _LINAC_COLORS[i % len(_LINAC_COLORS)]
            )
            tiers.setdefault(i, []).append({
                "lat": hc[0],
                "lon": hc[1],
                "cum_height": cum,
                "color": _row_color,
                "tip": (
                    f"<b>{row_data['name']}</b><br/>"
                    + (f"{_city}<br/>" if _city else "")
                    + f"{int(row_data['n_linacs'])} LINAC{'s' if row_data['n_linacs'] != 1 else ''}"
                    + (f"<br/>Capacity: {int(row_data['capacity']):,} pts/yr" if "capacity" in row_data else "")
                ),
            })

    # Render from HIGHEST tier index to 0. depthMask=False so shorter layers
    # always paint over the bottom of taller ones (stacked-bar effect).
    layers = []
    for tier_idx in sorted(tiers.keys(), reverse=True):
        tier_df = pd.DataFrame(tiers[tier_idx])
        layers.append(pdk.Layer(
            "ColumnLayer",
            id=f"linac-columns-tier-{tier_idx}",
            data=tier_df,
            get_position="[lon, lat]",
            get_elevation="cum_height",
            elevation_scale=1,
            radius=col_radius_m,
            get_fill_color="color",
            get_line_color=[0, 0, 0, 60],
            pickable=True,
            auto_highlight=True,
            extruded=True,
            parameters={"depthMask": False},
        ))
    return layers


def _fmt_sigfig(value: float, sig: int = 3) -> str:
    """Format a number to 1 decimal place with k/M suffix."""
    if value is None:
        return "N/A"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f} M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f} k"
    return f"{value:.1f}"


def _hex_areas_km2(gdf) -> np.ndarray:
    """Return exact H3 cell area in km² for each row (varies with latitude)."""
    return np.array([h3.cell_area(str(cell), unit="km^2") for cell in gdf["h3"]])


def _scale_caption(gdf) -> str:
    """Return a human-readable scale string for the initial map view."""
    geom = gdf.geometry
    lat_mid = float((geom.bounds["maxy"].max() + geom.bounds["miny"].min()) / 2)
    lon_span = float(geom.bounds["maxx"].max() - geom.bounds["minx"].min())
    width_km = lon_span * 111.32 * math.cos(math.radians(lat_mid))
    res = h3.get_resolution(str(gdf["h3"].iloc[0]))
    try:
        hex_area = h3.average_hexagon_area(res, unit="km^2")
        hex_diam = math.sqrt(hex_area / math.pi) * 2
    except Exception:
        hex_diam = None
    parts = []
    if hex_diam:
        parts.append(f" | Diameter per hexagon ≈ {hex_diam:.1f} km")
    return " ".join(parts)


def _make_view(gdf, pitch: float = 0.0) -> pdk.ViewState:
    geom = gdf.geometry
    span_lat = float(geom.bounds["maxy"].max() - geom.bounds["miny"].min())
    if span_lat > 130:  # world scale — fixed view centred on land masses
        return pdk.ViewState(latitude=20, longitude=10, zoom=0.7, pitch=pitch, bearing=0)
    cx = float(geom.centroid.x.mean())
    cy = float(geom.centroid.y.mean())
    zoom = max(3, min(8, int(8 - np.log2(max(span_lat, 0.5)))))
    return pdk.ViewState(latitude=cy, longitude=cx, zoom=zoom, pitch=pitch, bearing=0)


_CARTO_LIGHT_LABELS   = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
_CARTO_LIGHT_NOLABELS = "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json"
_CARTO_DARK_LABELS    = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
_CARTO_DARK_NOLABELS  = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json"
# Defaults — overwritten after sidebar renders with show_map_labels value
CARTO_LIGHT = _CARTO_LIGHT_NOLABELS
CARTO_DARK  = _CARTO_DARK_NOLABELS


def _linac_legend_fig(dark: bool = False) -> plt.Figure:
    """Small cylinder icon with 'LINAC' label for map legend."""
    text_color = "white" if dark else "black"
    fig, ax = plt.subplots(figsize=(0.9, 0.9))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    from matplotlib.patches import Ellipse, FancyBboxPatch
    # cylinder body
    ax.add_patch(FancyBboxPatch((0.25, 0.25), 0.5, 0.45, boxstyle="round,pad=0.0",
                                facecolor="#e05a2b", edgecolor="white", linewidth=0.8))
    # top ellipse
    ax.add_patch(Ellipse((0.5, 0.70), 0.5, 0.18, facecolor="#f08060", edgecolor="white", linewidth=0.8))
    ax.text(0.5, 0.08, "LINAC", ha="center", va="bottom", fontsize=7,
            color=text_color, fontweight="bold")
    fig.tight_layout(pad=0.1)
    return fig


def _render_discrete_legend(bounds: list, palette: list, unit: str, text_color: str = "black") -> None:
    """Render a discrete colour legend in the current column."""
    n = len(palette)
    def _hex_color(rgba):
        return "#{:02x}{:02x}{:02x}".format(rgba[0], rgba[1], rgba[2])
    band_h = max(40, int(300 / n))
    items = []
    for i in range(n):
        color = _hex_color(palette[i])
        if i == 0:
            label = f"< {bounds[0]:.4g}" + (f" {unit}" if unit else "")
        elif i == n - 1:
            label = f"≥ {bounds[i - 1]:.4g}" + (f" {unit}" if unit else "")
        else:
            label = f"{bounds[i - 1]:.4g}–{bounds[i]:.4g}" + (f" {unit}" if unit else "")
        items.append(
            f"<div style='display:flex;align-items:center;height:{band_h}px'>"
            f"<div style='background:{color};width:16px;height:{band_h - 2}px;flex-shrink:0;border-radius:2px'></div>"
            f"<span style='font-size:18px;color:{text_color};margin-left:4px;line-height:1.2'>{label}</span></div>"
        )
    _cr_color = "#aaa" if text_color == "white" else "#888"
    st.markdown(
        f"<p style='font-size:10px;font-family:monospace;color:{_cr_color};margin:0;text-align:center;'>© RadMaps 2025</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-top:8px'>" + "".join(items) + "</div>", unsafe_allow_html=True)


def _render_map_no_cb(layers, view: pdk.ViewState, dark: bool, on_select=None):
    """Render a pydeck map with an empty right column (matching _render_with_colorbar layout)."""
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=CARTO_DARK if dark else CARTO_LIGHT,
        tooltip={"html": "{tip}"},
    )
    col_map, col_cb = st.columns([7, 1])
    chart_state = None
    with col_map:
        if on_select and _PYDECK_CLICK_SUPPORTED:
            chart_state = st.pydeck_chart(deck, use_container_width=True,
                                          on_select=on_select, selection_mode="single-object")
        else:
            st.pydeck_chart(deck, use_container_width=True)
    with col_cb:
        st.markdown(
            "<p style='font-size:10px;font-family:monospace;color:#888;margin:0;text-align:center;'>© RadMaps 2025</p>",
            unsafe_allow_html=True,
        )
    return chart_state


def _render_with_colorbar(
    layers,
    view: pdk.ViewState,
    cmap_fn,
    vmin: float,
    vmax: float,
    cb_label: str,
    log_scale: bool = False,
    dark: bool = False,
    dark_text: bool = False,
    clamp: bool = False,
    show_linac_legend: bool = False,
    on_select=None,
):
    if not isinstance(layers, list):
        layers = [layers]
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=CARTO_DARK if dark else CARTO_LIGHT,
        tooltip={"html": "{tip}"},
    )
    col_map, col_cb = st.columns([7, 1])
    chart_state = None
    with col_map:
        if on_select and _PYDECK_CLICK_SUPPORTED:
            chart_state = st.pydeck_chart(deck, use_container_width=True,
                                          on_select=on_select, selection_mode="single-object")
        else:
            st.pydeck_chart(deck, use_container_width=True)
    with col_cb:
        _cr_color = "#aaa" if (dark or dark_text) else "#888"
        st.markdown(
            f"<p style='font-size:10px;font-family:monospace;color:{_cr_color};margin:0;text-align:center;'>© RadMaps 2025</p>",
            unsafe_allow_html=True,
        )
        fig = _colorbar_fig(cmap_fn, vmin, vmax, cb_label, log_scale=log_scale, text_color="white" if (dark or dark_text) else "black", clamp=clamp)
        st.pyplot(fig, use_container_width=True)
    return chart_state


def _process_click_event(chart_state) -> bool:
    """Check a pydeck chart state for a hex click; add LINAC to session state.

    Returns True if a new LINAC was added (caller should call st.rerun()).
    """
    if chart_state is None:
        return False
    sel = getattr(chart_state, "selection", None)
    if sel is None:
        return False
    objects = getattr(sel, "objects", None) or {}
    # Try the known hex layer id first, then fall back to any layer
    clicked_rows = objects.get(_HEX_LAYER_ID, [])
    if not clicked_rows:
        for v in objects.values():
            if v:
                clicked_rows = v
                break
    if not clicked_rows:
        return False
    clicked_h3 = clicked_rows[0].get("h3")
    if not clicked_h3:
        return False
    centroid = h3.cell_to_latlng(str(clicked_h3))
    _lat, _lon = round(centroid[0], 5), round(centroid[1], 5)
    custom = st.session_state.setdefault("custom_linacs", [])
    if any(abs(c["lat"] - _lat) < 1e-4 and abs(c["lon"] - _lon) < 1e-4 for c in custom):
        return False  # duplicate
    custom.append({
        "name": f"Custom LINAC {len(custom) + 1}",
        "lat": _lat,
        "lon": _lon,
        "capacity": 450,
    })
    return True


def _h3_caption(gdf) -> str:
    res = h3.get_resolution(str(gdf["h3"].iloc[0]))
    try:
        area = h3.average_hexagon_area(res, unit="km^2")
        return (
            # f"(Empty hexagons = no population in Kontur data)  \n" 
            f"**H3 Setup:** Resolution = {res} | Total hexagons = {len(gdf):,} | Area per hexagon ≈ {area:.2f} km² "
            
        )
    except Exception:
        return f"**H3 Setup:** H3 Resolution = {res} | Total hexagons = {len(gdf):,} hexagons"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

MAP_TYPES = [
    "Population Data",
    "Radiotherapy Access",
    "Nearest Linac",
]

_POP_DATA_METRICS = ["Population Density", "Cancer Incidence", "Radiotherapy Demand"]

# ---------------------------------------------------------------------------
# World default cache — pre-computed result loaded on first session visit
# ---------------------------------------------------------------------------

_WORLD_DEFAULT_CACHE = Path(__file__).resolve().parent / "cache" / "world_default.pkl"
_WORLD_DEFAULT_PARAMS = {
    "country": "World",
    "iso3": "WLD",
    "model": "step",
    "max_distance_km": 200.0,
    "h3_res": 3,
    "rt_method": "optimal",
    "capacity": 450.0,
    "region_percountry": True,
}


def _save_world_default(gdf_out, stats) -> None:
    import pickle
    _WORLD_DEFAULT_CACHE.parent.mkdir(exist_ok=True)
    with open(_WORLD_DEFAULT_CACHE, "wb") as f:
        pickle.dump({"gdf_out": gdf_out, "stats": stats, "params": _WORLD_DEFAULT_PARAMS}, f)


def _load_world_default():
    import pickle
    if not _WORLD_DEFAULT_CACHE.exists():
        return None
    try:
        with open(_WORLD_DEFAULT_CACHE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# On first visit this session, pre-populate the map result from disk cache
if "_session_init" not in st.session_state:
    st.session_state["_session_init"] = True
    _cached = _load_world_default()
    if _cached is not None:
        st.session_state.setdefault("_map_result", {
            "gdf_out": _cached["gdf_out"],
            "stats": _cached["stats"],
            "region_percountry": True,
        })
        st.session_state.setdefault("_map_generated", True)


with st.sidebar:
    st.title("🏥 RadMaps")

    # ── Region and Resolution ─────────────────────────────────────────────
    st.subheader("Region and Resolution")

    _all_options = _selection_options()
    country = st.selectbox(
        "Country / Region",
        options=_all_options,
        index=_all_options.index("World") if "World" in _all_options else 0,
    )
    _is_region = is_region(country)

    # Regional mode selector — only shown when a region is selected
    _region_percountry = False
    if _is_region:
        _reg_mode = st.radio(
            "Regional mode",
            ["Uniform", "Per-country"],
            index=1,
            horizontal=True,
            key="region_mode_radio",
            help=(
                "**Uniform**: single regional cancer profile (GLOBOCAN aggregate), "
                "all LINACs pooled across the region.\n\n"
                "**Per-country**: each country's own cancer data and LINACs; "
                "countries with no LINACs contribute full unmet demand."
            ),
        )
        _region_percountry = _reg_mode == "Per-country"

    if _is_region:
        _reg_def = get_region(country)
        _res_opts = [r for r in [1, 2, 3] if r <= _reg_def.max_resolution]
        _res_labels = {1: "H1 (~2.5M km²)", 2: "H2 (~87k km²)", 3: "H3 (~12,400 km²)"}
        h3_resolution = st.selectbox(
            "H3 resolution", options=_res_opts,
            index=min(2, len(_res_opts) - 1),
            format_func=lambda r: _res_labels.get(r, str(r)),
            key="h3_res_region",
        )
    else:
        h3_resolution = st.selectbox(
            "H3 resolution", options=[7, 6, 5, 4, 3], index=2,
            format_func=lambda r: {
                7: "H7 (~5 km²)", 6: "H6 (~36 km²)",
                5: "H5 (~253 km²)", 4: "H4 (~1,770 km²)", 3: "H3 (~12,400 km²)",
            }[r],
            key="h3_res_country",
        )

    st.divider()

    # ── Radiotherapy Demand Calculation ──────────────────────────────────
    rt_method: str = "optimal"
    rt_fraction: float = 0.25

    st.subheader("Radiotherapy Demand Calculation")
    _rt_label = st.radio(
        "RT demand method",
        ["Optimal RTU", "Custom RTU", "Proportional RTU"],
        horizontal=False, key="rt_demand_method_radio", label_visibility="collapsed",
    )
    if "Custom" in _rt_label:
        rt_method = "custom"
    elif "Optimal" in _rt_label:
        rt_method = "optimal"
    else:
        rt_method = "proportional"
    if rt_method == "proportional":
        rt_fraction = st.slider(
            "Fraction of cancer cases needing RT",
            min_value=0.01, max_value=1.0, value=0.25, step=0.01, format="%.2f",
        )

    # Proportional uses all cancers excl. NMSC; optimal/custom uses per-cancer utilisation rates
    selected_cancers: List[str] = ["All cancers excl. NMSC"] if rt_method == "proportional" else ["All cancers"]
    access_rt_method: str = rt_method
    access_rt_fraction: float = rt_fraction
    # For custom: read current RTU rates from session state (populated by Data tab editor)
    # Use country as key — iso3 is not yet resolved in the sidebar
    _custom_rtu_ss_key = f"custom_rtu_{country}"
    if rt_method == "custom" and _custom_rtu_ss_key in st.session_state:
        access_custom_rtu: tuple = tuple(sorted(st.session_state[_custom_rtu_ss_key].items()))
    else:
        access_custom_rtu = ()

    st.divider()

    # ── RT Access Calculations ────────────────────────────────────────────
    lambda_km: float = 30.0
    max_distance_km: float = 100.0
    weibull_k: float = 2.0
    access_model: str = "weibull"
    capacity_per_machine_per_year: float = 450.0
    use_travel_time: bool = False
    tt_mode: str = "driving"
    tt_app_id: str = ""
    tt_api_key: str = ""
    tt_max_travel_time_sec: int = 18000
    snap_linacs_to_hex: bool = False

    st.subheader("RT Access Calculations")

    st.markdown("**Capacity**")
    capacity_per_machine_per_year = float(st.slider(
        "Capacity per LINAC (patients/yr)", min_value=50, max_value=1000, value=450, step=50,
    ))
    
    st.caption('Individual facilities capacities can be modified in the Data tab.')

    st.markdown("**Geographic**")

    tt_method = st.radio(
        "Decay metric",
        ["Straight-line distance", "Driving time", "Public transport time"],
        index=0, horizontal=True,
    )
    use_travel_time = tt_method != "Straight-line distance"
    tt_mode = "driving" if tt_method == "Driving time" else "public_transport"

    if use_travel_time:
        if h3_resolution not in _TT_SUPPORTED_RES:
            st.warning(
                f"TravelTime H3 API supports resolutions 5–12. "
                f"Resolution {h3_resolution} is not supported."
            )
        st.markdown("**TravelTime API credentials** [(get a key)](https://traveltime.com/)")
        _tt_default_app_id = st.secrets.get("traveltime", {}).get("app_id", "")
        _tt_default_api_key = st.secrets.get("traveltime", {}).get("api_key", "")
        tt_app_id = st.text_input("App ID", value=_tt_default_app_id, key="tt_app_id")
        tt_api_key = st.text_input("API Key", value=_tt_default_api_key, key="tt_api_key", type="password")
        _tt_res_cap_sec = _TT_MAX_BY_RES.get(h3_resolution, 36000)
        _tt_res_cap_h = _tt_res_cap_sec // 3600
        tt_max_travel_time_hours = st.slider(
            "Travel time cut-off (hours)",
            min_value=1, max_value=_tt_res_cap_h, value=min(5, _tt_res_cap_h), step=1,
            help=(
                f"Shorter cut-offs fetch fewer hexagons and use fewer API credits. "
                f"Resolution {h3_resolution} hard cap: {_tt_res_cap_h}h "
                f"({'documented' if h3_resolution >= 6 else 'empirical'})."
            ),
        )
        tt_max_travel_time_sec = tt_max_travel_time_hours * 3600

    model_label = st.radio(
        "Access model",
        ["Weibull", "Step function", "Uniform (no decay)"],
        index=1, horizontal=True,
    )
    access_model = {"Weibull": "weibull", "Step function": "step", "Uniform (no decay)": "uniform"}[model_label]
    _unit = "min" if use_travel_time else "km"
    if access_model == "weibull":
        lambda_km = float(st.slider(f"Scale λ ({_unit})  —  P(λ) = 37%", 5, 200, 60 if use_travel_time else 150, step=5))
        weibull_k = float(st.slider("Shape k  —  steeper at higher k (k=1 → exponential)", 1.0, 6.0, 4.0, step=0.5))
    elif access_model == "step":
        max_distance_km = float(st.slider(
            f"Max treatment {'time' if use_travel_time else 'distance'} ({_unit})",
            10, 500, 60 if use_travel_time else 200, step=10,
        ))

    _use_latlng = st.checkbox(
        "Use lat/long coords", value=True,
        help="When enabled, the exact lat/long facility coordinates from DIRAC are used. "
             "When disabled, each facility is projected to the H3 hex centroid.",
    )
    snap_linacs_to_hex = not _use_latlng

    st.divider()

    # ── Map to View ───────────────────────────────────────────────────────
    st.subheader("Map to View")
    map_type = st.selectbox(
        "Map type", MAP_TYPES, index=MAP_TYPES.index("Radiotherapy Access"), key="_map_type_select",
    )
    _is_pop_data = map_type == "Population Data"
    if _is_pop_data:
        map_type = st.selectbox(
            "Display metric", _POP_DATA_METRICS, index=0, key="_pop_metric_select",
        )

    is_rt_demand_map = map_type == "Radiotherapy Demand"
    is_cancer = map_type in ("Cancer Incidence", "Radiotherapy Demand")
    is_access = map_type == "Radiotherapy Access"
    is_nearest = map_type == "Nearest Linac"
    needs_linac = is_access or is_nearest

    access_display_metric: str = "Modelled Access Deficit"
    if is_access:
        access_display_metric = st.selectbox(
            "RT Access display metric",
            ["Modelled Access Deficit", "Modelled Accessed", "Modelled Access Ratio", "Geographic Access Probability"],
            index=0,
        )

    _fac_cap_ss_key = f"facility_cap_{country}"
    _calc_fingerprint = (
        country, h3_resolution, access_model,
        round(lambda_km, 2), round(max_distance_km, 2), round(weibull_k, 2),
        round(capacity_per_machine_per_year, 1),
        rt_method, round(rt_fraction, 3),
        snap_linacs_to_hex, use_travel_time, tt_mode, tt_max_travel_time_sec,
        access_custom_rtu,
        tuple(sorted(st.session_state.get(_fac_cap_ss_key, {}).items())),
    )
    _calc_stale = _calc_fingerprint != st.session_state.get("_last_calc_fingerprint")
    if st.button(
        "Calculate RT Access",
        type="primary" if _calc_stale else "secondary",
        use_container_width=True,
        help="Recompute accessibility with current parameters." if _calc_stale else "Parameters unchanged since last run.",
    ):
        st.session_state["_map_generated"] = True
        st.session_state["_map_result"] = None
        st.session_state["_opt_result"] = None
        st.session_state["_last_calc_fingerprint"] = _calc_fingerprint
        st.session_state["_switch_to_map_tab"] = True

    st.divider()

    # Read click-mode toggle from session state before the map renders.
    # The toggle widget itself is shown later (inside the Add Additional LINACs section),
    # but its value must be known when building the map layers with on_select.
    click_mode: bool = bool(st.session_state.get("click_mode_toggle", False)) if is_access else False

    st.subheader("Colourbar")
    

    _default_cmap_name = _DEFAULT_CMAP.get(map_type, "Purple → Yellow (Viridis)")
    _cmap_options = list(COLORMAPS.keys())
    cb_cmap_name = st.selectbox(
        "Colour bar", options=_cmap_options,
        index=_cmap_options.index(_default_cmap_name) if _default_cmap_name in _cmap_options else 0,
    )
    cb_cmap_fn = COLORMAPS.get(cb_cmap_name, _viridis_rgb)

    _default_log = map_type in ("Population Density", "Cancer Incidence")
    _discrete_scale = False
    _no_hex = False
    _discrete_base = 60.0
    _discrete_steps = 4
    _scale_opts = ["Linear", "Log", "Discrete", "No hex"]
    _scale_default_idx = 1 if _default_log else 0
    _scale_type = st.radio("Scale", _scale_opts, index=_scale_default_idx, horizontal=True, key="scale_type_radio")
    cb_log = _scale_type == "Log"
    _discrete_scale = _scale_type == "Discrete"
    _no_hex = _scale_type == "No hex"
    if _discrete_scale:
        if is_access and access_display_metric in ("Modelled Access Ratio", "Geographic Access Probability"):
            _disc_default, _disc_step = 0.2, 0.05
        elif is_access:
            _disc_default, _disc_step = 450.0, 50.0
        elif is_nearest:
            _disc_default, _disc_step = (30.0, 5.0) if use_travel_time else (50.0, 10.0)
        else:
            _disc_default, _disc_step = 60.0, 5.0
        _disc_key = f"disc_base_{map_type}_{access_display_metric}_{'tt' if (is_nearest and use_travel_time) else 'km'}"
        _discrete_base = st.number_input(
            "Base threshold (X)",
            min_value=0.1, value=_disc_default, step=_disc_step,
            key=_disc_key,
            help="Bands: < X, X–2X, 2X–3X, … up to N bands. Colours: red → orange → yellow → green.",
        )
        _discrete_steps = int(st.number_input("Number of bands", min_value=2, max_value=5, value=5, step=1))
    # back-compat aliases used in _color_values and rendering paths
    _binary_scale = _discrete_scale
    _binary_cmap = _discrete_scale
    _binary_threshold = _discrete_base

    _count_maps = {"Population Density", "Cancer Incidence", "Radiotherapy Demand"}
    _count_access_metrics = {"Modelled Access", "Modelled Access Deficit"}
    _supports_per_km2 = map_type in _count_maps or (is_access and access_display_metric in _count_access_metrics)
    density_per_km2: bool = False
    _show_cb_controls = not _no_hex
    if _supports_per_km2 and _show_cb_controls:
        _density_radio = st.radio(
            "Colour scale normalisation", ["Per hexagon", "Per 10 km²"], index=0, horizontal=True,
        )
        density_per_km2 = _density_radio == "Per 10 km²"

    cb_auto = True
    cb_vmin_user: Optional[float] = None
    cb_vmax_user: Optional[float] = None
    if _show_cb_controls:
        cb_auto = st.checkbox("Auto range", value=True)
        if not cb_auto:
            cb_vmin_user = st.number_input("Min value", value=0.0, format="%.4g")
            cb_vmax_user = st.number_input("Max value", value=1.0, format="%.4g")

    if st.button("Update Map", use_container_width=True,
                 help="Re-render with current display settings — no recomputation."):
        st.session_state["_map_generated"] = True
        st.session_state["_switch_to_map_tab"] = True

    generate = st.session_state.get("_map_generated", False)

    # ── Plot Settings ─────────────────────────────────────────────────────
    st.subheader("Linac View Settings")

    show_linac_markers: bool = True
    tower_height_scale: float = 1.0
    tower_radius_scale: float = 1.0
    linac_tower_style: str = "stacked"
    linac_multi_color: bool = False

    show_linac_markers = st.checkbox("Show LINAC locations", value=needs_linac)
    map_pitch_on: bool = False
    if show_linac_markers:
        tower_height_scale = float(st.slider("Tower height scale", 0.05, 5.0, 1.0, step=0.05))
        tower_radius_scale = float(st.slider("Tower radius scale", 0.1, 5.0, 1.0, step=0.1))
        map_pitch_on = st.toggle("3D view (map pitch)", value=True)
        _tower_style_label = st.radio(
            "Tower style",
            ["Individual (tower per centre)", "Stacked (tower per hex)"],
            horizontal=False,
        )
        linac_tower_style = "individual" if "Individual" in _tower_style_label else "stacked"
        linac_multi_color = st.checkbox("Multiple colours", value=False,
                                        help="Assign a distinct colour to each facility; otherwise all shown in blue.")

    st.divider()

    st.subheader("App View")

    app_dark_mode: bool = False
    dark_mode: bool = False

    app_dark_mode = st.toggle("Dark background", value=False)
    _apply_app_dark_mode(app_dark_mode)
    dark_mode = st.checkbox("Dark map", value=False)
    show_map_labels = st.checkbox("Show place names", value=False)

CARTO_LIGHT = _CARTO_LIGHT_LABELS if show_map_labels else _CARTO_LIGHT_NOLABELS
CARTO_DARK  = _CARTO_DARK_LABELS  if show_map_labels else _CARTO_DARK_NOLABELS


# ---------------------------------------------------------------------------
# Helpers used in map sections
# ---------------------------------------------------------------------------

def _color_values(values: np.ndarray, cmap_fn, auto_vmin: float, auto_vmax: float,
                  invert_binary: bool = False):
    """Apply colormap using user or auto range, optionally in log/binary space.

    invert_binary: when True, above-threshold = red (for metrics where high = bad).
    """
    vmin = cb_vmin_user if not cb_auto and cb_vmin_user is not None else auto_vmin
    vmax = cb_vmax_user if not cb_auto and cb_vmax_user is not None else auto_vmax
    if _discrete_scale:
        _bounds = [n * _discrete_base for n in range(1, _discrete_steps)]
        if invert_binary:
            _palette = list(reversed(_DISCRETE_PALETTE[:_discrete_steps]))
        else:
            _palette = _DISCRETE_PALETTE[:_discrete_steps]
        def _disc_color(v):
            if not np.isfinite(v):
                return [80, 80, 80, 100]
            for i, b in enumerate(_bounds):
                if v < b:
                    return _palette[i]
            return _palette[-1]
        colors = [_disc_color(v) for v in values.tolist()]
    elif cb_log:
        colors = _apply_colormap_fixed(np.log1p(values), cmap_fn, np.log1p(max(vmin, 0)), np.log1p(vmax))
    else:
        colors = _apply_colormap_fixed(values, cmap_fn, vmin, vmax)
    return colors, vmin, vmax


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

if _is_region:
    iso3 = get_region(country).globocan_code
else:
    try:
        iso3 = pycountry.countries.lookup(country).alpha_3
    except LookupError:
        st.error(f"Could not resolve country: {country!r}")
        st.stop()

tab_map, tab_data, tab_geo, tab_cap, _tab_sep1, tab_intro, tab_method, tab_assumptions, _tab_sep2, tab_toy, tab_model = st.tabs([
    "🗺️ RT Access", "📊 Data", "🌍 Geography-Only", "⚡ Capacity-Only", "│", "💡 Introduction", "📖 Method", "⚠️ Assumptions", "│", "🧪 Toy Example", "📐 Probability Models",
])

if st.session_state.pop("_switch_to_map_tab", False):
    import streamlit.components.v1 as _components
    _components.html(
        "<script>setTimeout(function(){"
        "var t=window.parent.document.querySelectorAll('[data-baseweb=\"tab\"]');"
        "if(t.length>0)t[0].click();"
        "},120);</script>",
        height=0,
    )

# ---------------------------------------------------------------------------
# Data tab — always available, no Generate button required
# ---------------------------------------------------------------------------

with tab_data:
    st.header(f"Data — {country}")

    # ---- Population --------------------------------------------------------
    st.subheader(f"Population — {country}")
    with st.spinner("Loading population data…"):
        _pop_gdf = _load_pop_region(country, 3) if _is_region else _load_pop(country, 5)
    _total_pop = int(_pop_gdf["population"].sum())
    st.metric("Total population", f"{_total_pop:,}")
    st.caption(
        "Population Source: [Kontur Population Dataset 2023](https://data.humdata.org/dataset/kontur-population-dataset) "
        "— population modelled at 400 m H3 resolution from GHSL, OSM, and census data."
    )

    st.divider()

    # ---- Annual cancer incidence -------------------------------------------
    st.subheader(f"Cancer Incidence and Radiotherapy Demand — {country}")
    if not has_globocan_data(iso3):
        st.warning(f"**{country}** (ISO3: {iso3}) is not present in the GLOBOCAN dataset.")
    else:
        with st.spinner("Loading cancer data…"):
            _cancer_df = _data_tab_cancer(iso3)

        # Separate aggregate rows from site rows
        _agg_mask = _cancer_df["Cancer type"].str.strip().str.lower().isin(_AGGREGATE_CANCER_KEYS)
        _agg_rows = _cancer_df[_agg_mask].set_index(_cancer_df[_agg_mask]["Cancer type"].str.strip().str.lower())
        _site_df = _cancer_df[~_agg_mask].copy()

        _all_cancers_n = int(_agg_rows.loc["all cancers", "New cases"]) if "all cancers" in _agg_rows.index else int(_site_df["New cases"].sum())
        _excl_nmsc_n = int(_agg_rows.loc["all cancers excl. nmsc", "New cases"]) if "all cancers excl. nmsc" in _agg_rows.index else _all_cancers_n

        # --- Build RTU% column based on rt_method chosen in sidebar ---
        _opt_fracs = get_optimal_rt_fractions()
        _opt_fracs_norm = {k.strip().lower(): v for k, v in _opt_fracs.items()}
        _rtu_col_label = {
            "optimal": "RTU (optimal) %",
            "custom": "RTU (custom) %",
            "proportional": "RTU (proportional) %",
        }[rt_method]
        _dt_custom_ss_key = f"custom_rtu_{country}"

        if rt_method == "proportional":
            _site_df["_rtu_pct"] = round(rt_fraction * 100, 1)
        elif rt_method == "custom":
            if _dt_custom_ss_key not in st.session_state:
                st.session_state[_dt_custom_ss_key] = {
                    row["Cancer type"]: round(_opt_fracs_norm.get(row["Cancer type"].strip().lower(), 0.0) * 100, 1)
                    for _, row in _site_df.iterrows()
                }
            _site_df["_rtu_pct"] = _site_df["Cancer type"].map(
                lambda c: st.session_state[_dt_custom_ss_key].get(c, 0.0)
            )
        else:  # optimal
            _site_df["_rtu_pct"] = _site_df["Cancer type"].apply(
                lambda c: round(_opt_fracs_norm.get(c.strip().lower(), 0.0) * 100, 1)
            )

        _site_df["_cases_rt"] = (_site_df["New cases"] * _site_df["_rtu_pct"] / 100).round(0).astype(int)
        _total_rt_demand_n = int(_site_df["_cases_rt"].sum())

        # --- Three headline metrics ---
        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.metric(
            "Total Cancer Incidence", f"{_all_cancers_n:,}",
            delta=f"{100 * _all_cancers_n / _total_pop:.2f}% of population" if _total_pop > 0 else None,
            delta_color="off",
        )
        _mc2.metric(
            "Total Cancer Incidence (excl. NMSC)", f"{_excl_nmsc_n:,}",
            delta=f"{100 * _excl_nmsc_n / _total_pop:.2f}% of population" if _total_pop > 0 else None,
            delta_color="off",
        )
        _mc3.metric(
            "Total Radiotherapy Demand", f"{_total_rt_demand_n:,}",
            delta=f"{100 * _total_rt_demand_n / _total_pop:.2f}% of population" if _total_pop > 0 else None,
            delta_color="off",
        )

        # --- Cancer site table ---
        st.markdown("**Cancer site breakdown**")
        if rt_method == "custom":
            # No key: state managed entirely via session_state so editor diffs don't conflict
            # across reruns when multiple cells are edited sequentially.
            _editor_in = _site_df[["Cancer type", "New cases", "% of All Cancers", "_rtu_pct", "_cases_rt"]].rename(
                columns={"_rtu_pct": _rtu_col_label, "_cases_rt": "Cases needing RT"}
            )
            _edited_df = st.data_editor(
                _editor_in,
                column_config={
                    "Cancer type": st.column_config.TextColumn(disabled=True),
                    "New cases": st.column_config.NumberColumn(disabled=True),
                    "% of All Cancers": st.column_config.NumberColumn(disabled=True),
                    _rtu_col_label: st.column_config.NumberColumn(
                        min_value=0.0, max_value=100.0, step=0.5, format="%.1f",
                        help="Edit radiotherapy utilisation rate (%)",
                    ),
                    "Cases needing RT": st.column_config.NumberColumn(disabled=True),
                },
                use_container_width=True,
                hide_index=True,
            )
            st.session_state[_dt_custom_ss_key] = {
                row["Cancer type"]: float(row[_rtu_col_label])
                for _, row in _edited_df.iterrows()
            }
            st.caption("Press **Calculate RT Access** in the sidebar to update the map and refresh these numbers.")
        else:
            
            _display_df = _site_df[["Cancer type", "New cases", "% of All Cancers", "_rtu_pct", "_cases_rt"]].rename(
                columns={"_rtu_pct": _rtu_col_label, "_cases_rt": "Cases needing RT"}
            ).copy()
            _display_df["New cases"] = _display_df["New cases"].apply(lambda x: f"{x:,}")
            _display_df["Cases needing RT"] = _display_df["Cases needing RT"].apply(lambda x: f"{x:,}")
            st.dataframe(_display_df, use_container_width=True, hide_index=True)

            if rt_method == "optimal":
                st.caption(
                    "Optimal RTU Source: Delaney et al. (2005). " 
                    "Cancer. 2005;104(6):1129–37. "
                    "[doi:10.1002/cncr.21324](https://doi.org/10.1002/cncr.21324)"
                )
        
        st.caption("Cancer Incidence Source: [GLOBOCAN 2022](https://gco.iarc.who.int/today/), International Agency for Research on Cancer (IARC).")

    st.divider()

    # ---- LINAC facilities --------------------------------------------------
    st.subheader(f"Radiotherapy Facilities with Linacs — {country}")
    _linac_result = _load_dirac(country)
    if _linac_result[0] is None:
        st.info(f"No Linacs data found for **{country}** in the DIRAC database.")
    else:
        _, _linac_df = _linac_result
        _fac_cap_key = f"facility_cap_{country}"
        _fac_cap_store = st.session_state.get(_fac_cap_key, {})
        _linac_editor_df = _linac_df[["name", "city", "lat", "lon", "n_linacs"]].copy()
        _linac_editor_df["Capacity (pts/yr)"] = _linac_editor_df.apply(
            lambda r: _fac_cap_store.get(r["name"], r["n_linacs"] * capacity_per_machine_per_year),
            axis=1,
        ).astype(int)
        _linac_editor_df = _linac_editor_df.rename(columns={
            "name": "Facility", "city": "City",
            "lat": "Lat", "lon": "Lon", "n_linacs": "LINACs",
        })
        _linac_edited = st.data_editor(
            _linac_editor_df,
            column_config={
                "Facility": st.column_config.TextColumn(disabled=True, width="medium"),
                "City": st.column_config.TextColumn(disabled=True, width="small"),
                "Lat": st.column_config.NumberColumn(disabled=True, format="%.3f", width="small"),
                "Lon": st.column_config.NumberColumn(disabled=True, format="%.3f", width="small"),
                "LINACs": st.column_config.NumberColumn(disabled=True, width="small"),
                "Capacity (pts/yr)": st.column_config.NumberColumn(
                    min_value=0, step=50, format="%d",
                    help="Total patients treated per year at this facility. Default = LINACs × capacity per machine.",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )
        # Persist edits to session state
        st.session_state[_fac_cap_key] = {
            row["Facility"]: int(row["Capacity (pts/yr)"])
            for _, row in _linac_edited.iterrows()
        }
        st.caption(
            f"**{int(_linac_df['n_linacs'].sum())} LINACs** across **{len(_linac_df)} facilities**. "
            "Edit Capacity to override per-facility throughput. "
            "Press **Calculate RT Access** to apply changes.  \n"
            "Source: [IAEA DIRAC Database](https://dirac.iaea.org/). "
            "Coordinates corrected via OpenStreetMap geocoding where missing or erroneous."
        )

    
# ---------------------------------------------------------------------------
# Probability Model tab
# ---------------------------------------------------------------------------

with tab_model:
    st.header("Probability Model")
    st.markdown(
        "This tab illustrates how the selected probability model translates "
        "distance from a LINAC facility into a probability of treatment."
    )

    _pm_model = st.selectbox(
        "Probability model",
        ["Weibull", "Step function", "Uniform (no decay)"],
        key="pm_model",
    )

    st.divider()

    if _pm_model == "Weibull":
        _pm_wlambda = st.slider("Scale λ (km)  —  P(λ) = 37%", 5, 500, 150, step=5, key="pm_wlambda")
        _pm_wk = st.slider("Shape k  —  higher = steeper", 1.0, 6.0, 4.0, step=0.5, key="pm_wk")

        st.markdown("### Formula")
        st.latex(r"P(\text{treatment} \mid d) = \exp\!\left(-\left(\frac{d}{\lambda}\right)^k\right)")
        st.markdown(
            r"$\lambda$ is the scale (km) at which $P = e^{-1} \approx 37\%$ for any $k$. "
            r"$k$ controls shape: $k = 1$ is identical to exponential decay; "
            r"$k > 1$ gives an S-curve with a flat plateau near the facility then a steeper drop-off. "
            "When multiple facilities exist, contributions are combined as:"
        )
        st.latex(r"P_{\text{total}} = 1 - \prod_{i}\!\left(1 - e^{-(d_i/\lambda)^k}\right)^{w_i}")
        _p_at_lambda = float(np.exp(-1))
        _p_at_half = float(np.exp(-(0.5 ** _pm_wk)))
        st.markdown(
            f"At the current settings (λ = {_pm_wlambda} km, k = {_pm_wk}): "
            f"P({_pm_wlambda} km) = **{_p_at_lambda:.1%}**, "
            f"P({_pm_wlambda//2} km) = **{_p_at_half:.1%}**."
        )

        _pm_dist = np.linspace(0, 1000, 500)
        _pm_prob = np.exp(-np.power(_pm_dist / _pm_wlambda, _pm_wk))

    elif _pm_model == "Step function":
        _pm_cutoff = st.slider("Max treatment distance (km)", 10, 1000, 100, step=10, key="pm_cutoff")

        st.markdown("### Formula")
        st.latex(
            r"P(\text{treatment} \mid d) = \begin{cases} 1 & d \leq d_{\max} \\ 0 & d > d_{\max} \end{cases}"
        )
        st.markdown(
            f"Patients within **{_pm_cutoff} km** of a LINAC are assumed to have "
            "100% probability of treatment; those beyond have 0%."
        )

        _pm_dist = np.linspace(0, 1000, 500)
        _pm_prob = (_pm_dist <= _pm_cutoff).astype(float)

    else:  # Uniform
        st.markdown("### Formula")
        st.latex(r"P(\text{treatment} \mid d) = 1 \quad \forall\, d")
        st.markdown(
            "Distance has no effect on access. All patients are assumed to have "
            "equal probability of treatment regardless of their distance from a LINAC. "
            "Capacity constraints still apply."
        )

        _pm_dist = np.linspace(0, 1000, 500)
        _pm_prob = np.ones(500)

    # ---- Plot ---------------------------------------------------------------
    _fig_pm = go.Figure()
    _fig_pm.add_trace(
        go.Scatter(
            x=_pm_dist,
            y=_pm_prob * 100,
            mode="lines",
            line=dict(color="#1f77b4", width=2.5),
            name="P(treatment)",
        )
    )

    if _pm_model == "Step function":
        _fig_pm.add_vline(
            x=_pm_cutoff,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Cut-off: {_pm_cutoff} km",
            annotation_position="top right",
        )
    elif _pm_model == "Exponential decay":
        _fig_pm.add_vline(
            x=_pm_lambda,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"λ = {_pm_lambda} km  (P ≈ 37%)",
            annotation_position="top right",
        )
    elif _pm_model == "Weibull":
        _fig_pm.add_vline(
            x=_pm_wlambda,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"λ = {_pm_wlambda} km  (P ≈ 37%)",
            annotation_position="top right",
        )

    _fig_pm.update_layout(
        xaxis_title="Distance from nearest LINAC (km)",
        xaxis=dict(range=[0, 1000]),
        yaxis_title="Probability of treatment (%)",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        height=420,
        margin=dict(l=60, r=30, t=30, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(_fig_pm, use_container_width=True)

# ---------------------------------------------------------------------------
# Map tab
# ---------------------------------------------------------------------------

def _render_pop_data_fom(gdf, iso3: str, country: str, rt_method: str, rt_fraction: float) -> None:
    """Render the four Population Data figures-of-merit below any sub-metric map."""
    st.divider()
    _fom_c1, _fom_c2, _fom_c3, _fom_c4 = st.columns(4)
    _fom_c1.metric("Population", f"{int(gdf['population'].sum()):,}")
    _fom_c2.metric("H3 hexagons", f"{len(gdf):,}")
    if not has_globocan_data(iso3):
        _fom_c3.metric("Cancer Incidence excl. NMSC", "N/A")
        _fom_c4.metric("Cases Requiring RT", "N/A")
        return
    _fom_rt = _data_tab_rt_need(iso3)
    _fom_c3.metric("Cancer Incidence excl. NMSC", _fmt_sigfig(_fom_rt['total_cancer_excl_nmsc']))
    if rt_method == "optimal":
        _fom_rt_val = _fom_rt["total_rt_cases"]
        _fom_rt_note = "Optimal RTU (Delaney et al. 2005)"
    elif rt_method == "proportional":
        _fom_rt_val = _fom_rt["total_cancer_excl_nmsc"] * rt_fraction
        _fom_rt_note = f"Proportional ({rt_fraction:.0%} of cases excl. NMSC)"
    else:  # custom
        _cstm = st.session_state.get(f"custom_rtu_{country}", {})
        _nat = get_national_cases(iso3, get_cancer_types() + DERIVED_CANCER_TYPES)
        _fom_rt_val = sum(
            _nat.get(c, 0.0) * (_cstm.get(c, 0.0) / 100.0)
            for c in _nat if c.strip().lower() not in _AGGREGATE_CANCER_KEYS
        )
        _fom_rt_note = "Custom RTU rates"
    _fom_c4.metric("Cases Requiring RT", _fmt_sigfig(_fom_rt_val), help=f"Method: {_fom_rt_note}")


with tab_map:
    _map_header = f"Population Data › {map_type} — {country}" if _is_pop_data else f"{map_type} — {country}"
    st.header(_map_header)

    if not generate:
        st.info("Configure options in the sidebar and click **Generate Map**.")
    else:

        # Load LINAC data (needed for access/nearest maps or when markers are requested)
        locs: Optional[List[Tuple[float, float, float]]] = None
        facilities_df: Optional[pd.DataFrame] = None
        country_span_km: float = 1000.0  # default; refined when gdf is available
        if needs_linac or show_linac_markers:
            with st.spinner("Loading LINAC data from DIRAC database…"):
                result = _load_dirac(country)
            if result[0] is None:
                locs = []
                facilities_df = pd.DataFrame()
            else:
                locs, facilities_df = result

        # Apply per-facility custom capacities (set via Data tab editor)
        _fac_cap_overrides = st.session_state.get(f"facility_cap_{country}", {})
        if locs is not None and facilities_df is not None and not facilities_df.empty:
            facilities_df = facilities_df.copy()
            facilities_df["capacity"] = facilities_df.apply(
                lambda r: _fac_cap_overrides.get(r["name"], r["n_linacs"] * capacity_per_machine_per_year),
                axis=1,
            )
            # Rebuild locs weights from effective capacity
            locs = [
                (row["lat"], row["lon"], row["capacity"] / capacity_per_machine_per_year)
                for _, row in facilities_df.iterrows()
            ]

        # Merge any custom LINACs added via click-to-add
        _custom_linacs = st.session_state.get("custom_linacs", [])
        if _custom_linacs and locs is not None:
            def _custom_cap(c):
                return c.get("capacity") or c.get("n_linacs", 1.0) * capacity_per_machine_per_year
            _custom_rows = pd.DataFrame([
                {"name": c["name"], "city": "Custom", "lat": c["lat"], "lon": c["lon"],
                 "n_linacs": _custom_cap(c) / capacity_per_machine_per_year,
                 "capacity": _custom_cap(c)}
                for c in _custom_linacs
            ])
            if facilities_df is not None and not facilities_df.empty:
                facilities_df = pd.concat([facilities_df, _custom_rows], ignore_index=True)
            elif facilities_df is not None:
                facilities_df = _custom_rows
            locs = list(locs) + [(c["lat"], c["lon"], _custom_cap(c) / capacity_per_machine_per_year) for c in _custom_linacs]

        # ---- Population Density -----------------------------------------------
        if map_type == "Population Density":
            with st.spinner("Loading population data…"):
                gdf = _load_pop_region(country, h3_resolution) if _is_region else _load_pop(country, h3_resolution)

            pop = gdf["population"].to_numpy(dtype=np.float64)
            _areas_pop = _hex_areas_km2(gdf)
            if density_per_km2:
                plot_vals = pop / (_areas_pop / 10)
                pop_label = "Population per 10 km²"
            else:
                plot_vals = pop
                pop_label = "Population per hexagon"
            auto_vmin = float(max(plot_vals.min(), 1e-3))
            auto_vmax = float(plot_vals.max())
            colors, vmin, vmax = _color_values(plot_vals, cb_cmap_fn, auto_vmin, auto_vmax)

            gdf = gdf.copy()
            gdf["color"] = colors
            _s_area_pop = pd.Series(_areas_pop, index=gdf.index).apply(_fmt_sigfig)
            _s_pop_raw = gdf["population"].apply(_fmt_sigfig)
            _pop_tip = (
                "<b>" + gdf["h3"].astype(str) + "</b><br/>"
                + "Population: " + _s_pop_raw + "<br/>"
                + "Hex area: " + _s_area_pop + " km²"
            )
            gdf["tip"] = _pop_tip

            _geom_pop = gdf.geometry
            _lat_span_pop = float(_geom_pop.bounds["maxy"].max() - _geom_pop.bounds["miny"].min())
            _lon_span_pop = float(_geom_pop.bounds["maxx"].max() - _geom_pop.bounds["minx"].min())
            _lat_mid_pop = float((_geom_pop.bounds["maxy"].max() + _geom_pop.bounds["miny"].min()) / 2)
            country_span_km = max(_lat_span_pop * 111.32, _lon_span_pop * 111.32 * math.cos(math.radians(_lat_mid_pop)))

            df = pd.DataFrame({"h3": gdf["h3"], "color": gdf["color"], "tip": gdf["tip"]})
            _pop_pitch = 30.0 if (show_linac_markers and map_pitch_on and facilities_df is not None and not facilities_df.empty) else 0.0
            _pop_layers = [] if _no_hex else [_build_hex_layer(df)]
            if show_linac_markers and facilities_df is not None and not facilities_df.empty:
                _pop_layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style, color=None if linac_multi_color else _LINAC_BLUE))
            if _no_hex:
                _render_map_no_cb(_pop_layers, _make_view(gdf, pitch=_pop_pitch), dark_mode)
            else:
                _render_with_colorbar(
                    _pop_layers,
                    _make_view(gdf, pitch=_pop_pitch),
                    cb_cmap_fn, vmin, vmax, pop_label, log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                    show_linac_legend=show_linac_markers and facilities_df is not None and not facilities_df.empty,
                )
            st.caption(_h3_caption(gdf) + _scale_caption(gdf))
            _render_pop_data_fom(gdf, iso3, country, rt_method, rt_fraction)

        # ---- Cancer maps -------------------------------------------------------
        elif is_cancer:
            if not selected_cancers:
                st.warning("Please select at least one cancer type.")
            else:
                if not has_globocan_data(iso3):
                    st.warning(
                        f"**{country}** (ISO3: {iso3}) is not present in this GLOBOCAN dataset. "
                        "Cancer case counts will be zero. The population map is still available."
                    )

                # For optimal RT calculation in aggregate views, expand to individual
                # sites so each cancer type is weighted by its own RT fraction rather
                # than a flat aggregate rate.
                _all_individual = [
                    c for c in get_cancer_types() + DERIVED_CANCER_TYPES
                    if c.strip().lower() not in _AGGREGATE_CANCER_KEYS
                ]
                # For optimal RT demand, expand to all individual sites so each
                # cancer type is weighted by its own utilisation rate.
                if map_type == "Radiotherapy Demand" and rt_method == "optimal":
                    _load_cancers = _all_individual
                else:
                    _load_cancers = selected_cancers  # "All cancers excl. NMSC" for proportional

                with st.spinner("Apportioning cancer incidence to H3 grid…"):
                    if _is_region and _region_percountry:
                        gdf = _load_cancer_region_percountry(country, tuple(_load_cancers), False, h3_resolution)
                    else:
                        gdf = _load_cancer(country, iso3, tuple(_load_cancers), False, h3_resolution, region_flag=_is_region)

                if map_type == "Radiotherapy Demand":
                    suffix = "_optimal_rt" if rt_method == "optimal" else "_incidence"
                else:
                    suffix = "_incidence"

                cols_of_interest = [c + suffix for c in _load_cancers if (c + suffix) in gdf.columns]
                if not cols_of_interest:
                    st.error("No matching columns found in data.")
                else:
                    combined = gdf[cols_of_interest].sum(axis=1).to_numpy(dtype=np.float64)

                    if map_type == "Radiotherapy Demand" and rt_method == "proportional":
                        combined = combined * rt_fraction

                    _areas_cancer = _hex_areas_km2(gdf)
                    if density_per_km2:
                        plot_vals_c = combined / (_areas_cancer / 10)
                        _per_suffix = " per 10 km²"
                    else:
                        plot_vals_c = combined
                        _per_suffix = " per hexagon"

                    auto_vmin = float(max(plot_vals_c.min(), 0.001))
                    auto_vmax = float(plot_vals_c.max())
                    colors, vmin, vmax = _color_values(plot_vals_c, cb_cmap_fn, auto_vmin, auto_vmax)

                    gdf = gdf.copy()
                    gdf["color"] = colors

                    if map_type == "Radiotherapy Demand":
                        label = f"RT demand{_per_suffix}"
                    else:
                        label = f"Cancer incidence{_per_suffix}"

                    _s_combined_raw = pd.Series(combined, index=gdf.index).round(2).astype(str)
                    _s_area_c = pd.Series(_areas_cancer, index=gdf.index).apply(_fmt_sigfig)
                    _s_pop_c = gdf["population"].apply(_fmt_sigfig)
                    _tip_raw_label = "Radiotherapy Demand" if map_type == "Radiotherapy Demand" else "Cancer Incidence"
                    gdf["tip"] = (
                        "<b>" + gdf["h3"].astype(str) + "</b><br/>"
                        + _tip_raw_label + ": " + _s_combined_raw + "<br/>"
                        + "<hr style='margin:3px 0'/>"
                        + "Population: " + _s_pop_c + "<br/>"
                        + "Hex area: " + _s_area_c + " km²"
                    )

                    _geom_c = gdf.geometry
                    _lat_span_c = float(_geom_c.bounds["maxy"].max() - _geom_c.bounds["miny"].min())
                    _lon_span_c = float(_geom_c.bounds["maxx"].max() - _geom_c.bounds["minx"].min())
                    _lat_mid_c = float((_geom_c.bounds["maxy"].max() + _geom_c.bounds["miny"].min()) / 2)
                    country_span_km = max(_lat_span_c * 111.32, _lon_span_c * 111.32 * math.cos(math.radians(_lat_mid_c)))

                    df = pd.DataFrame({"h3": gdf["h3"], "color": gdf["color"], "tip": gdf["tip"]})
                    _cancer_pitch = 30.0 if (show_linac_markers and map_pitch_on and facilities_df is not None and not facilities_df.empty) else 0.0
                    _cancer_layers = [] if _no_hex else [_build_hex_layer(df)]
                    if show_linac_markers and facilities_df is not None and not facilities_df.empty:
                        _cancer_layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style, color=None if linac_multi_color else _LINAC_BLUE))
                    if _no_hex:
                        _render_map_no_cb(_cancer_layers, _make_view(gdf, pitch=_cancer_pitch), dark_mode)
                    else:
                        _render_with_colorbar(
                            _cancer_layers,
                            _make_view(gdf, pitch=_cancer_pitch),
                            cb_cmap_fn, vmin, vmax, label, log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                            show_linac_legend=show_linac_markers and facilities_df is not None and not facilities_df.empty,
                        )
                    st.caption(_h3_caption(gdf) + _scale_caption(gdf))
                    if map_type == "Radiotherapy Demand":
                        if rt_method == "optimal":
                            _rt_method_text = (
                                "Based on optimal RT utilisation rates (Delaney et al. 2005): "
                                "each cancer site weighted by its evidence-based RT fraction."
                            )
                        else:
                            _rt_method_text = (
                                f"Based on proportional scaling: {rt_fraction:.0%} of all cancer cases assumed to require RT."
                            )
                        st.caption(_rt_method_text)
                    if map_type == "Cancer Incidence":
                        _scope_text = (
                            "Showing: all cancers excl. NMSC (proportional)"
                            if rt_method == "proportional"
                            else "Showing: all cancer sites (GLOBOCAN)"
                        )
                        st.caption(_scope_text)
                    _cases_label = (
                        "Cancer Incidence: All Cancers excl. NMSC"
                        if rt_method == "proportional"
                        else "Cancer Incidence: All Sites"
                    )

                    if map_type == "Radiotherapy Demand":
                        incidence_cols = [c + "_incidence" for c in _load_cancers if (c + "_incidence") in gdf.columns]
                        total_incidence = float(gdf[incidence_cols].sum(axis=1).sum()) if incidence_cols else 0.0
                        col1, col2, col3 = st.columns(3)
                        col1.metric(_cases_label, _fmt_sigfig(total_incidence))
                        col2.metric("Corresponding Cases Requiring RT", _fmt_sigfig(combined.sum()))
                        col3.metric("H3 hexagons", f"{len(gdf):,}")
                    else:
                        total_pop = float(gdf["population"].sum())
                        col1, col2, col3 = st.columns(3)
                        col1.metric(_cases_label, _fmt_sigfig(combined.sum()))
                        col2.metric("Country population", f"{int(total_pop):,}")
                        col3.metric("H3 hexagons", f"{len(gdf):,}")

        # ---- Radiotherapy Access / Nearest Linac ----------------------
        elif is_access or is_nearest:
            linac_locs_tuple = tuple(locs)

            _map_result = st.session_state.get("_map_result")
            # Discard cached result if TT mode or regional mode has changed.
            if _map_result is not None:
                _result_had_tt = "nearest_linac_min" in _map_result["gdf_out"].columns
                _result_was_percountry = _map_result.get("region_percountry", False)
                if use_travel_time != _result_had_tt or _region_percountry != _result_was_percountry:
                    _map_result = None
                    st.session_state["_map_result"] = None
            if _map_result is not None:
                gdf_out = _map_result["gdf_out"]
                stats = _map_result["stats"]
            else:
                if use_travel_time and _region_percountry:
                    st.warning(
                        "Travel time mode is not supported with **Per-country** regional analysis. "
                        "Switch to **Uniform** regional mode or a single country to use travel times."
                    )
                    st.stop()
                if use_travel_time:
                    if not tt_app_id or not tt_api_key:
                        st.warning(
                            "Enter your TravelTime App ID and API Key in the sidebar to use "
                            "driving or public transport times."
                        )
                        st.stop()

                    # Build structured cache key: {iso3}_res{R}_{Hh}h_{linac_hash8}
                    # File on disk: {key}_{mode}.npz  (mode appended by compute_travel_time_matrix)
                    import hashlib as _hl, json as _json
                    _gdf_tmp = _load_pop_region(country, h3_resolution) if _is_region else _load_pop(country, h3_resolution)
                    _hex_ids = list(_gdf_tmp["h3"])
                    _linac_ll = [(lat, lon) for lat, lon, _ in locs]
                    _linac_hash = _hl.md5(_json.dumps(sorted(_linac_ll)).encode()).hexdigest()[:8]
                    _tt_max_h = tt_max_travel_time_sec // 3600
                    _tt_cache_key = f"{iso3}_res{h3_resolution}_{_tt_max_h}h_{_linac_hash}"
                    _tt_cache_file = _TT_CACHE_DIR / f"{_tt_cache_key}_{tt_mode}.npz"

                    if not _tt_cache_file.exists():
                        _tt_progress = st.progress(0, text="Fetching travel times from TravelTime API…")
                        def _tt_cb(done, total):
                            _tt_progress.progress(
                                min(done / max(total, 1), 1.0),
                                text=f"Travel time API: request {done}/{total}…",
                            )
                        try:
                            _, _tt_errors = compute_travel_time_matrix(
                                _hex_ids, _linac_ll, h3_resolution, tt_mode,
                                tt_app_id, tt_api_key,
                                cache_key=_tt_cache_key,
                                progress_callback=_tt_cb,
                                max_travel_time_sec=tt_max_travel_time_sec,
                            )
                            _tt_progress.empty()
                            if _tt_errors:
                                st.warning("Some TravelTime batches failed: " + "; ".join(_tt_errors))
                        except Exception as _e:
                            st.error(f"TravelTime API error: {_e}")
                            st.stop()

                    with st.spinner("Computing accessibility…"):
                        gdf_out, stats = _compute_access_travel_time(
                            country, iso3, linac_locs_tuple,
                            float(lambda_km), access_model, float(max_distance_km),
                            capacity_per_machine_per_year, tt_mode, _tt_cache_key,
                            access_rt_method, access_rt_fraction,
                            h3_resolution, _is_region, snap_linacs_to_hex,
                            weibull_k=float(weibull_k),
                            custom_rtu=access_custom_rtu,
                        )
                elif _region_percountry:
                    _n_countries = len(get_region(country).member_alpha2)
                    _pc_bar = st.progress(0, text=f"Computing per-country accessibility: 0 / {_n_countries}")
                    def _pc_progress(done, total):
                        _pc_bar.progress(
                            done / total,
                            text=f"Computing per-country accessibility: {done} / {total}",
                        )
                    gdf_out, stats = _compute_access_region_percountry(
                        country,
                        float(lambda_km), access_model, float(max_distance_km),
                        capacity_per_machine_per_year, access_rt_method, access_rt_fraction,
                        h3_resolution, snap_linacs_to_hex,
                        weibull_k=float(weibull_k),
                        custom_rtu=access_custom_rtu,
                        progress_callback=_pc_progress,
                    )
                    _pc_bar.empty()
                else:
                    with st.spinner("Computing accessibility…"):
                        gdf_out, stats = _compute_access(
                            country, iso3, linac_locs_tuple,
                            float(lambda_km), access_model, float(max_distance_km),
                            capacity_per_machine_per_year, access_rt_method, access_rt_fraction,
                            h3_resolution, _is_region, snap_linacs_to_hex,
                            weibull_k=float(weibull_k),
                            custom_rtu=access_custom_rtu,
                        )
                st.session_state["_map_result"] = {"gdf_out": gdf_out, "stats": stats, "region_percountry": _region_percountry}

                # Persist world default to disk so future sessions load instantly
                _wp = _WORLD_DEFAULT_PARAMS
                if (country == _wp["country"] and access_model == _wp["model"]
                        and abs(float(max_distance_km) - _wp["max_distance_km"]) < 1
                        and h3_resolution == _wp["h3_res"]
                        and access_rt_method == _wp["rt_method"]
                        and abs(capacity_per_machine_per_year - _wp["capacity"]) < 1
                        and _region_percountry == _wp["region_percountry"]):
                    _save_world_default(gdf_out, stats)

            pitch = 30.0 if (show_linac_markers and map_pitch_on) else 0.0

            _geom = gdf_out.geometry
            _lat_span = float(_geom.bounds["maxy"].max() - _geom.bounds["miny"].min())
            _lon_span = float(_geom.bounds["maxx"].max() - _geom.bounds["minx"].min())
            _lat_mid = float((_geom.bounds["maxy"].max() + _geom.bounds["miny"].min()) / 2)
            country_span_km = max(
                _lat_span * 111.32,
                _lon_span * 111.32 * math.cos(math.radians(_lat_mid)),
            )

            if is_nearest:
                _has_tt = use_travel_time and "nearest_linac_min" in gdf_out.columns
                if _has_tt:
                    dist_vals = gdf_out["nearest_linac_min"].to_numpy(dtype=np.float64)
                    _near_label = "Travel time (min)"
                    _near_tip_label = "Nearest Linac"
                    _near_tip_unit = "min"
                else:
                    dist_vals = gdf_out["nearest_linac_km"].to_numpy(dtype=np.float64)
                    _near_label = "Distance (km)"
                    _near_tip_label = "Nearest Linac"
                    _near_tip_unit = "km"

                valid = np.isfinite(dist_vals)
                # Cap unreachable (inf) at the user-selected max travel time for stats/histograms/tooltips
                _TT_MAX = tt_max_travel_time_sec / 60.0 if _has_tt else _TT_MAX_BY_RES.get(h3_resolution, 14400) / 60.0
                _has_capped = _has_tt and not valid.all()
                _dist_for_stats = np.where(np.isfinite(dist_vals), dist_vals, _TT_MAX) if _has_tt else dist_vals
                auto_vmin = 0.0
                auto_vmax = float(np.percentile(_dist_for_stats, 95)) if len(_dist_for_stats) > 0 else 500.0

                # Tooltip: show "> 240 min" for unreachable hexes when using travel time
                _tip_dist_str = pd.Series(dist_vals, index=gdf_out.index)
                if _has_tt:
                    _tip_dist_str = _tip_dist_str.apply(
                        lambda v: f"> {int(_TT_MAX)} {_near_tip_unit}" if not np.isfinite(v) else f"{v:.1f} {_near_tip_unit}"
                    )
                else:
                    _tip_dist_str = _tip_dist_str.round(1).astype(str) + f" {_near_tip_unit}"

                # Colour assignment
                if _discrete_scale:
                    dist_vals_plot = np.where(valid, dist_vals, np.inf)
                    colors, vmin, vmax = _color_values(dist_vals_plot, cb_cmap_fn, auto_vmin, auto_vmax)
                    vmin, vmax = 0.0, _discrete_base * _discrete_steps
                else:
                    dist_vals_plot = np.where(valid, dist_vals, auto_vmax)
                    colors, vmin, vmax = _color_values(dist_vals_plot, cb_cmap_fn, auto_vmin, auto_vmax)

                _areas_near = _hex_areas_km2(gdf_out)
                _s_area_near = pd.Series(_areas_near, index=gdf_out.index).apply(_fmt_sigfig)
                _s_pop_near = gdf_out["population"].apply(_fmt_sigfig)
                gdf_out = gdf_out.copy()
                gdf_out["color"] = colors
                gdf_out["tip"] = (
                    "<b>" + gdf_out["h3"].astype(str) + "</b><br/>"
                    + _near_tip_label + ": "
                    + _tip_dist_str
                    + "<hr style='margin:3px 0'/>"
                    + "Population: " + _s_pop_near + "<br/>"
                    + "Hex area: " + _s_area_near + " km²"
                )

                _near_df = pd.DataFrame({"h3": gdf_out["h3"], "color": gdf_out["color"], "tip": gdf_out["tip"]})
                layers = [] if _no_hex else [_build_hex_layer(_near_df)]
                if show_linac_markers:
                    layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style, color=None if linac_multi_color else _LINAC_BLUE))

                if click_mode and _PYDECK_CLICK_SUPPORTED:
                    st.info("Click a hexagon on the map to place a LINAC there, then click **Generate Map** to recompute.")

                _near_chart_state = None
                if _no_hex:
                    _near_chart_state = _render_map_no_cb(layers, _make_view(gdf_out, pitch=pitch), dark_mode,
                                                          on_select="rerun" if click_mode else None)
                elif _discrete_scale:
                    _map_col, _leg_col = st.columns([7, 1])
                    with _map_col:
                        _bin_deck = pdk.Deck(layers=layers, initial_view_state=_make_view(gdf_out, pitch=pitch),
                                             map_style=CARTO_DARK if dark_mode else CARTO_LIGHT, tooltip={"html": "{tip}"})
                        if click_mode and _PYDECK_CLICK_SUPPORTED:
                            _near_chart_state = st.pydeck_chart(_bin_deck, use_container_width=True,
                                                                on_select="rerun", selection_mode="single-object")
                        else:
                            st.pydeck_chart(_bin_deck, use_container_width=True)
                    with _leg_col:
                        _disc_bounds_near = [n * _discrete_base for n in range(1, _discrete_steps)]
                        _render_discrete_legend(_disc_bounds_near, _DISCRETE_PALETTE[:_discrete_steps], _near_tip_unit,
                                                "white" if (dark_mode or app_dark_mode) else "black")
                else:
                    _near_chart_state = _render_with_colorbar(
                        layers, _make_view(gdf_out, pitch=pitch),
                        cb_cmap_fn, vmin, vmax, _near_label,
                        log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                        show_linac_legend=show_linac_markers,
                        on_select="rerun" if click_mode else None,
                    )
                if click_mode and _process_click_event(_near_chart_state):
                    st.rerun()
                st.caption(_h3_caption(gdf_out) + _scale_caption(gdf_out))
                _near_pop_all = gdf_out["population"].to_numpy(dtype=np.float64)
                _pop_total_nn = _near_pop_all.sum()
                _mean_geo_prob_nn = stats.get("mean_access_probability", 0.0)
                _median_val = float(np.median(_dist_for_stats))
                _gt = "> " if _has_capped and _median_val >= _TT_MAX - 0.1 else ""
                # Population-weighted median via cumulative population sort
                _sort_idx = np.argsort(_dist_for_stats)
                _cum_pop = np.cumsum(_near_pop_all[_sort_idx])
                _pw_median_idx = np.searchsorted(_cum_pop, _pop_total_nn * 0.5)
                _pw_median_val = float(_dist_for_stats[_sort_idx[min(_pw_median_idx, len(_sort_idx) - 1)]])
                _gt_pw = "> " if _has_capped and _pw_median_val >= _TT_MAX - 0.1 else ""
                col0, col1, col2, col3, col4 = st.columns(5)
                col0.metric("Facilities", int(stats["n_facilities"]))
                col1.metric("LINACs", int(stats["total_machines"]))
                col2.metric(f"Median {_near_tip_label}", f"{_gt}{_median_val:.1f} {_near_tip_unit}")
                col3.metric(f"Pop-Weighted Median {_near_tip_label}", f"{_gt_pw}{_pw_median_val:.1f} {_near_tip_unit}")
                col4.metric("Average Geographic Access Probability", f"{_mean_geo_prob_nn:.1%}")
                if _has_tt and not valid.all():
                    _unreachable_pop = float(_near_pop_all[~valid].sum())
                    _unreachable_pct = _unreachable_pop / _pop_total_nn * 100 if _pop_total_nn > 0 else 0.0
                    st.caption(
                        f"**Unreachable hexes:** {int((~valid).sum()):,} hexes · "
                        f"population {_fmt_sigfig(_unreachable_pop)} ({_unreachable_pct:.1f}% of total)."
                        f"(Note: TravelTime returns no route when a hex centroid cannot be snapped to the road network, "
                        f"e.g. isolated area, lake, or river)."
                    )

                # ---- Geography Only Calculations (Nearest Linac) -----------
                st.divider()
                _pop_wtd_dist_nn = float((_dist_for_stats * _near_pop_all).sum() / _pop_total_nn) if _pop_total_nn > 0 else 0.0
                if len(_dist_for_stats) > 0:
                    _fig_nn1, _fig_nn2 = st.columns(2)
                    with _fig_nn1:
                        _fig_h1 = go.Figure()
                        _fig_h1.add_trace(go.Histogram(
                            x=_dist_for_stats,
                            nbinsx=40,
                            marker_color="#4C9BE8",
                        ))
                        _fig_h1.update_layout(
                            xaxis_title=f"{_near_tip_label} ({_near_tip_unit})"
                                + (f" — bar at {int(_TT_MAX)} includes all travel time > {int(_TT_MAX)}" if _has_capped else ""),
                            yaxis_title="Number of hexagons",
                            height=240,
                            margin=dict(l=40, r=20, t=30, b=40),
                            title_text="Hexagon count by distance",
                            showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(_fig_h1, use_container_width=True)
                    with _fig_nn2:
                        _fig_h2 = go.Figure()
                        _fig_h2.add_trace(go.Histogram(
                            x=_dist_for_stats,
                            y=_near_pop_all,
                            histfunc="sum",
                            nbinsx=40,
                            marker_color="#F97316",
                        ))
                        _fig_h2.update_layout(
                            xaxis_title=f"{_near_tip_label} ({_near_tip_unit})"
                                + (f" — bar at {int(_TT_MAX)} includes all travel time > {int(_TT_MAX)}" if _has_capped else ""),
                            yaxis_title="Population",
                            height=240,
                            margin=dict(l=40, r=20, t=30, b=40),
                            title_text="Population by distance",
                            showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(_fig_h2, use_container_width=True)

            else:  # Radiotherapy Access
                prob = gdf_out["access_probability"].to_numpy(dtype=np.float64)
                cap_prob = gdf_out["capacity_limited_probability"].to_numpy(dtype=np.float64)
                raw_pop = gdf_out["population"].to_numpy(dtype=np.float64)

                s_h3 = gdf_out["h3"].astype(str)
                s_prob = (gdf_out["access_probability"] * 100).round(1).astype(str)
                s_cap = (gdf_out["capacity_limited_probability"] * 100).round(1).astype(str)

                # Common tooltip fields
                _areas_acc = _hex_areas_km2(gdf_out)
                _s_area_acc = pd.Series(_areas_acc, index=gdf_out.index).apply(_fmt_sigfig)
                s_pop_fmt = gdf_out["population"].apply(_fmt_sigfig)
                s_treated = gdf_out["rt_treated"].round(1).astype(str)
                s_untreated = gdf_out["rt_untreated"].round(1).astype(str)
                _rt_demand_arr = gdf_out["rt_demand"].to_numpy(dtype=np.float64)
                _pct_arr = np.where(
                    _rt_demand_arr > 0,
                    gdf_out["rt_treated"].to_numpy(dtype=np.float64) / _rt_demand_arr * 100,
                    0.0,
                )
                s_pct = pd.Series(_pct_arr, index=gdf_out.index).round(1).astype(str)

                if access_display_metric == "Modelled Access Deficit":
                    display_vals = gdf_out["rt_untreated"].to_numpy(dtype=np.float64)
                    cb_label_access = "RT access deficit"
                    auto_vmin_a = 0.0
                    auto_vmax_a = float(np.nanmax(display_vals))
                    _tip_prefix = "RT access deficit"
                    _tip_extra = "RT accessed"
                    _tip_extra_s = s_treated
                    metric_cmap_fn = _rdylgn_reversed_rgb

                elif access_display_metric == "Modelled Accessed":
                    display_vals = gdf_out["rt_treated"].to_numpy(dtype=np.float64)
                    cb_label_access = "RT accessed"
                    auto_vmin_a = 0.0
                    auto_vmax_a = float(np.nanmax(display_vals))
                    _tip_prefix = "RT accessed"
                    _tip_extra = "RT access deficit"
                    _tip_extra_s = s_untreated
                    metric_cmap_fn = _rdylgn_rgb

                elif access_display_metric == "Modelled Access Ratio":
                    display_vals = cap_prob
                    cb_label_access = "Modelled Access Ratio"
                    auto_vmin_a, auto_vmax_a = 0.0, 1.0
                    tip_series = (
                        "<b>" + s_h3 + "</b><br/>"
                        + "Modelled access ratio: " + s_cap + "%<br/>"
                        + "Geographic access probability: " + s_prob + "%<br/>"
                        + "<hr style='margin:3px 0'/>"
                        + "Population: " + s_pop_fmt + "<br/>"
                        + "Hex area: " + _s_area_acc + " km²"
                    )
                    metric_cmap_fn = _rdylgn_rgb

                else:  # Geographic Access Probability
                    display_vals = prob
                    cb_label_access = "Geographic Access Probability"
                    auto_vmin_a, auto_vmax_a = 0.0, 1.0
                    tip_series = (
                        "<b>" + s_h3 + "</b><br/>"
                        + "Geographic access probability: " + s_prob + "%<br/>"
                        + "Modelled access probability: " + s_cap + "%<br/>"
                        + "<hr style='margin:3px 0'/>"
                        + "Population: " + s_pop_fmt + "<br/>"
                        + "Hex area: " + _s_area_acc + " km²"
                    )
                    metric_cmap_fn = _rdylgn_rgb

                # Apply per-km² normalisation for count-based access metrics
                if access_display_metric in _count_access_metrics:
                    # Tooltip always shows raw per-hex counts regardless of density_per_km2
                    _s_rt_demand_raw = gdf_out["rt_demand"].round(1).astype(str)
                    if density_per_km2:
                        display_vals = display_vals / (_areas_acc / 10)
                        auto_vmax_a = float(np.nanmax(display_vals))
                        cb_label_access += " per 10 km²"
                    else:
                        cb_label_access += " per hexagon"
                    tip_series = (
                        "<b>" + s_h3 + "</b><br/>"
                        + "Demand Accessed: " + s_treated + "<br/>"
                        + "Total Demand: " + _s_rt_demand_raw + "<br/>"
                        + "Percent Accessed: " + s_pct + "%<br/>"
                        + "<hr style='margin:3px 0'/>"
                        + "Population: " + s_pop_fmt + "<br/>"
                        + "Hex area: " + _s_area_acc + " km²"
                    )

                _map_default_cmap = _DEFAULT_CMAP.get(map_type, "Green → Red")
                active_cmap_fn = metric_cmap_fn if cb_cmap_name == _map_default_cmap else cb_cmap_fn
                # For "Modelled Access Deficit", higher = worse → invert binary colours (above = red)
                _acc_invert_binary = access_display_metric == "Modelled Access Deficit"
                colors, vmin, vmax = _color_values(display_vals, active_cmap_fn, auto_vmin_a, auto_vmax_a,
                                                   invert_binary=_acc_invert_binary)

                gdf_out = gdf_out.copy()
                gdf_out["color"] = colors
                gdf_out["tip"] = tip_series.values

                _acc_df = pd.DataFrame({"h3": gdf_out["h3"], "color": gdf_out["color"], "tip": gdf_out["tip"]})
                layers = [] if _no_hex else [_build_hex_layer(_acc_df)]
                if show_linac_markers:
                    layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style, color=None if linac_multi_color else _LINAC_BLUE))

                if stats["total_rt_demand"] == 0:
                    st.warning(
                        f"RT demand is zero for **{country}** — this country may not be in the GLOBOCAN dataset. "
                        "Capacity allocation cannot be computed; geographic access probability is still valid."
                    )

                if access_display_metric in ("Modelled Access Ratio", "Geographic Access Probability"):
                    st.subheader(access_display_metric)

                if click_mode and _PYDECK_CLICK_SUPPORTED:
                    st.info("Click a hexagon on the map to place a LINAC there, then click **Generate Map** to recompute.")

                if _no_hex:
                    _acc_chart_state = _render_map_no_cb(layers, _make_view(gdf_out, pitch=pitch), dark_mode,
                                                         on_select="rerun" if click_mode else None)
                elif _discrete_scale:
                    _acc_map_col, _acc_leg_col = st.columns([7, 1])
                    with _acc_map_col:
                        _acc_deck = pdk.Deck(layers=layers, initial_view_state=_make_view(gdf_out, pitch=pitch),
                                             map_style=CARTO_DARK if dark_mode else CARTO_LIGHT, tooltip={"html": "{tip}"})
                        if click_mode and _PYDECK_CLICK_SUPPORTED:
                            _acc_chart_state = st.pydeck_chart(_acc_deck, use_container_width=True,
                                                               on_select="rerun", selection_mode="single-object")
                        else:
                            st.pydeck_chart(_acc_deck, use_container_width=True)
                            _acc_chart_state = None
                    with _acc_leg_col:
                        _disc_palette_acc = list(reversed(_DISCRETE_PALETTE[:_discrete_steps])) if _acc_invert_binary else _DISCRETE_PALETTE[:_discrete_steps]
                        _disc_bounds_acc = [n * _discrete_base for n in range(1, _discrete_steps)]
                        _render_discrete_legend(_disc_bounds_acc, _disc_palette_acc, "",
                                                "white" if (dark_mode or app_dark_mode) else "black")
                else:
                    _acc_chart_state = _render_with_colorbar(
                        layers, _make_view(gdf_out, pitch=pitch),
                        active_cmap_fn, vmin, vmax, cb_label_access,
                        log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                        show_linac_legend=show_linac_markers,
                        on_select="rerun" if click_mode else None,
                    )
                if click_mode and _process_click_event(_acc_chart_state):
                    st.rerun()
                st.caption(_h3_caption(gdf_out) + _scale_caption(gdf_out))
                if use_travel_time and "nearest_linac_min" in gdf_out.columns:
                    _acc_unreach_mask = ~np.isfinite(gdf_out["nearest_linac_min"].to_numpy(dtype=np.float64))
                    if _acc_unreach_mask.any():
                        _acc_unreach_pop = float(gdf_out["population"].to_numpy(dtype=np.float64)[_acc_unreach_mask].sum())
                        _acc_total_pop = float(gdf_out["population"].to_numpy(dtype=np.float64).sum())
                        _acc_unreach_pct = _acc_unreach_pop / _acc_total_pop * 100 if _acc_total_pop > 0 else 0.0
                        st.caption(
                            f"**Unreachable hexes:** {int(_acc_unreach_mask.sum()):,} hexes · "
                            f"population {_fmt_sigfig(_acc_unreach_pop)} ({_acc_unreach_pct:.1f}% of total). "
                            f"(Note: TravelTime returns no route when a hex centroid cannot be snapped to the road network, "
                            f"e.g. isolated area, lake, or river)."
                        )
                if access_display_metric == "Modelled Access Ratio":
                    st.caption(
                        "Modelled Access Ratio gives the ratio of patients accessing RT to total RT demand per hex."
                    )
                elif access_display_metric == "Geographic Access Probability":
                    st.caption(
                        "Geographic Access Probability shows the probability a patient in a given hex will "
                        "have treatment, given there are no linac capacity constraints."
                    )

                if access_model == "weibull":
                    model_info = f"Weibull | λ = {lambda_km} km | k = {weibull_k} | cut-off = {stats['cutoff_km']:.0f} km"
                elif access_model == "step":
                    model_info = f"Step function | max distance = {max_distance_km:.0f} km"
                else:
                    model_info = "Uniform (no distance decay)"

                # pre-compute all values
                _globocan = stats.get("total_cancer_excl_nmsc")
                _demand = stats['total_rt_demand']
                _treated = stats['total_rt_treated']
                _total_pop_acc = float(gdf_out["population"].sum())
                _modelled_ratio = _treated / _demand if _demand > 0 else 0.0
                _modelled_deficit = _demand - _treated
                _geo_access = stats.get("mean_access_probability", 0.0)
                _cap_ratio = min(stats['total_national_capacity'] / _demand, 1.0) if _demand > 0 else None
                _cap_accessed = min(stats['total_national_capacity'], _demand) if _demand > 0 else None
                _cap_deficit = max(_demand - stats['total_national_capacity'], 0.0) if _demand > 0 else None

                def _fmt_k(v) -> str:
                    if v is None: return "N/A"
                    return f"{float(v) / 1000:.1f} k"

                def _pct_num(number, pct):
                    return f"{_fmt_k(number)} ({pct:.1%})"

                # ── Statistics ───────────────────────────────────────────────
                _cancer_pct_of_pop = _globocan / _total_pop_acc if (_globocan and _total_pop_acc > 0) else None
                _rt_pct_of_cancer = _demand / _globocan if (_globocan and _globocan > 0) else None

                st.markdown("**Statistics**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Population", _fmt_sigfig(_total_pop_acc))
                col2.metric("Facilities", int(stats["n_facilities"]))
                col3.metric("LINACs", int(stats["total_machines"]))
                col4.metric("Cancer Incidence",
                            f"{_fmt_k(_globocan)} ({_cancer_pct_of_pop:.2%})" if _cancer_pct_of_pop is not None else (_fmt_k(_globocan) if _globocan else "N/A"),
                            help="Annual cancer cases (excl. NMSC) · % of population")

                st.divider()

                # ── RT Demand ────────────────────────────────────────────────
                st.markdown("**RT Demand**")
                _rtu_label = "optimal" if access_rt_method == "optimal" else f"proportional ({access_rt_fraction:.0%})"
                st.caption(f"**Parameters:** RTU = {_rtu_label}")
                st.metric("RT Demand",
                          _pct_num(_demand, _rt_pct_of_cancer) if _rt_pct_of_cancer is not None else _fmt_k(_demand),
                          help="Annual patients requiring RT · % of cancer incidence")

                st.divider()

                # ── Calculations ─────────────────────────────────────────────
                st.markdown("**Calculations**")
                if access_model == "step":
                    _params_str = (
                        f"**Parameters:** H3 Resolution = {h3_resolution}, Access Model = Step function, "
                        f"Cut-off = {int(max_distance_km)} km, Capacity per machine = {int(capacity_per_machine_per_year)}"
                    )
                elif access_model == "weibull":
                    _params_str = (
                        f"**Parameters:** H3 Resolution = {h3_resolution}, Access Model = Weibull, "
                        f"λ = {lambda_km} km, k = {weibull_k}, Capacity per machine = {int(capacity_per_machine_per_year)}"
                    )
                elif access_model == "uniform":
                    _params_str = (
                        f"**Parameters:** H3 Resolution = {h3_resolution}, Access Model = Uniform (no decay), "
                        f"Capacity per machine = {int(capacity_per_machine_per_year)}"
                    )
                else:
                    _params_str = (
                        f"**Parameters:** H3 Resolution = {h3_resolution}, Access Model = Exponential, "
                        f"λ = {lambda_km} km, Capacity per machine = {int(capacity_per_machine_per_year)}"
                    )
                st.caption(_params_str)

                st.markdown("*RadMaps*")
                col1b, col2b = st.columns(2)
                col1b.metric("Accessed",
                             _pct_num(_treated, _modelled_ratio),
                             help="Patients accessing RT per year · % of RT demand (capacity + geography combined)")
                col2b.metric("Deficit ∆",
                             _pct_num(_modelled_deficit, 1.0 - _modelled_ratio),
                             help="Patients not accessing RT per year · % of RT demand")

                st.markdown("*Capacity-only*")
                col1c, col2c = st.columns(2)
                col1c.metric("Accessed",
                             _pct_num(_cap_accessed, _cap_ratio) if _cap_ratio is not None else "N/A",
                             help="Patients serviceable by machine capacity alone · % of RT demand (no geographic barrier assumed)")
                col2c.metric("Deficit ∆",
                             _pct_num(_cap_deficit, 1.0 - _cap_ratio) if _cap_ratio is not None else "N/A",
                             help="Patients demand exceeds machine capacity · % of RT demand")

                st.markdown("*Geography-only*")
                col1d, col2d = st.columns(2)
                col1d.metric("Accessed",
                             _pct_num(_geo_access * _demand, _geo_access),
                             help="Patients within geographic reach of a LINAC · % of RT demand (unlimited capacity assumed)")
                col2d.metric("Deficit ∆",
                             _pct_num((1.0 - _geo_access) * _demand, 1.0 - _geo_access),
                             help="Patients too far from any LINAC · % of RT demand")
                # ---- Add Additional LINACs ------------------------------------
                if is_access:
                    st.divider()
                    st.subheader("Add Additional LINACs")
                    _add_tab_custom, _add_tab_opt = st.tabs(["Add Custom LINACs", "Suggest Optimal LINAC Placements"])

                    # ── Custom LINACs tab ───────────────────────────────────────
                    with _add_tab_custom:
                        _click_mode_custom = st.toggle(
                            "Click map to add LINACs", key="click_mode_toggle",
                            help=(
                                "Click a hexagon on the map to add a LINAC there, "
                                "then press Calculate RT Access to recompute."
                            ) if _PYDECK_CLICK_SUPPORTED else (
                                "Enter coordinates below. "
                                "Map-click support requires Streamlit ≥ 1.35 (current: " + st.__version__ + ")."
                            ),
                        )
                        if _click_mode_custom and not _PYDECK_CLICK_SUPPORTED:
                            _add_name = st.text_input("Name", value="Custom Centre", key="add_linac_name")
                            _add_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0,
                                                       value=0.0, format="%.5f", key="add_linac_lat")
                            _add_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                                       value=0.0, format="%.5f", key="add_linac_lon")
                            _add_cap = st.number_input("Capacity (pts/yr)", min_value=1, value=450, step=50,
                                                       key="add_linac_cap")
                            if st.button("Add Centre", use_container_width=True, key="add_linac_btn"):
                                _ex = st.session_state.setdefault("custom_linacs", [])
                                _ex.append({
                                    "name": _add_name or f"Custom Centre {len(_ex) + 1}",
                                    "lat": round(_add_lat, 5),
                                    "lon": round(_add_lon, 5),
                                    "capacity": int(_add_cap),
                                })
                                st.rerun()
                        _custom_now_tab = st.session_state.get("custom_linacs", [])
                        if _custom_now_tab:
                            st.caption(f"{len(_custom_now_tab)} custom centre(s) added — press **Calculate RT Access** to recompute.")
                            _edited_custom = st.data_editor(
                                pd.DataFrame(_custom_now_tab),
                                column_config={
                                    "name": st.column_config.TextColumn("Name"),
                                    "lat": st.column_config.NumberColumn("Lat", format="%.5f"),
                                    "lon": st.column_config.NumberColumn("Lon", format="%.5f"),
                                    "capacity": st.column_config.NumberColumn("Capacity (pts/yr)", min_value=1, step=50, format="%d"),
                                },
                                num_rows="dynamic",
                                use_container_width=True,
                                key="custom_linacs_editor",
                            )
                            _cc1, _cc2 = st.columns(2)
                            if _cc1.button("Apply edits", key="apply_custom_linacs"):
                                st.session_state["custom_linacs"] = _edited_custom.dropna(subset=["lat", "lon"]).to_dict("records")
                                st.rerun()
                            if _cc2.button("Clear all", key="clear_custom_linacs"):
                                st.session_state["custom_linacs"] = []
                                st.rerun()

                    # ── Suggest Optimal tab ─────────────────────────────────────
                    with _add_tab_opt:
                        _oc1, _oc2 = st.columns(2)
                        _n_suggest = int(_oc1.number_input(
                            "New facilities to suggest", min_value=1, max_value=999, value=3, step=1,
                            key="opt_n",
                        ))
                        _machines_per_new = float(_oc2.number_input(
                            "Machines per new facility", min_value=1, max_value=10, value=1, step=1,
                            key="opt_machines",
                        ))
                        _opt_metric_label = st.radio(
                            "Optimisation metric",
                            [
                                "Unmet RT demand (most patients without access)",
                                "Geographic isolation (most distant population)",
                            ],
                            horizontal=True,
                            key="opt_metric",
                        )
                        _opt_metric = "rt_access" if "Unmet" in _opt_metric_label else "geographic"

                        _snap_to_existing = st.checkbox(
                            "Add machines to nearest existing facility if it is within the distance threshold "
                            "(rather than creating a new site)",
                            key="opt_snap_enabled",
                            help="When the optimal placement location is within the threshold distance of any "
                                 "existing or already-suggested facility, the new machines are added there instead.",
                        )
                        if _snap_to_existing:
                            _sc1, _sc2 = st.columns(2)
                            _snap_km = float(_sc1.number_input(
                                "Threshold — distance (km)", min_value=1, max_value=500, value=50, step=5,
                                key="opt_snap_km",
                            ))
                            _snap_min = float(_sc2.number_input(
                                "Threshold — travel time (min)", min_value=1, max_value=300, value=60, step=5,
                                key="opt_snap_min",
                                help="Travel-time threshold is only applied when a TravelTime matrix is available.",
                            ))
                        else:
                            _snap_km = None
                            _snap_min = None

                        if st.button("Run optimisation", key="run_opt", type="primary"):
                            from pyproj import Geod as _OptGeod
                            _geod_opt = _OptGeod(ellps="WGS84")

                            # Load TT matrix from cache if available (for snap threshold in minutes)
                            _opt_tt_matrix = None
                            if use_travel_time:
                                _opt_tt_cache_file = _TT_CACHE_DIR / f"{_tt_cache_key}_{tt_mode}.npz"
                                if _opt_tt_cache_file.exists():
                                    _opt_tt_matrix = np.load(_opt_tt_cache_file)["matrix"]

                            _opt_locs = list(linac_locs_tuple)
                            _opt_gdf = gdf_out
                            _opt_stats = stats
                            _opt_suggested: list = []
                            _opt_steps: list = []
                            _opt_progress = st.progress(0, text="Running optimisation…")

                            for _opt_step in range(_n_suggest):
                                _olats = _opt_gdf["centroid_lat"].to_numpy()
                                _olons = _opt_gdf["centroid_lon"].to_numpy()
                                _opop = _opt_gdf["population"].to_numpy()

                                if _opt_metric == "rt_access":
                                    _oscores = _opt_gdf["rt_untreated"].to_numpy(dtype=np.float64)
                                else:
                                    _odist = _opt_gdf["nearest_linac_km"].fillna(0).to_numpy(dtype=np.float64)
                                    _oscores = _odist * np.where(_opop > 0, _opop, 0.0)

                                _obest = int(np.argmax(_oscores))
                                _onew_lat = float(_olats[_obest])
                                _onew_lon = float(_olons[_obest])

                                # Snap to nearest existing/suggested facility if within threshold
                                _snap_idx = None
                                if _snap_km is not None:
                                    _best_snap_dist_km = np.inf
                                    for _si, (_sfl, _sfo, _sfw) in enumerate(_opt_locs):
                                        _, _, _sdm = _geod_opt.inv(_sfo, _sfl, _onew_lon, _onew_lat)
                                        _sdk = _sdm * 1e-3
                                        _within = _sdk <= _snap_km
                                        if not _within and _snap_min is not None and _opt_tt_matrix is not None:
                                            if _si < _opt_tt_matrix.shape[1]:
                                                _tt_val = float(_opt_tt_matrix[_obest, _si])
                                                _within = np.isfinite(_tt_val) and _tt_val <= _snap_min
                                        if _within and _sdk < _best_snap_dist_km:
                                            _best_snap_dist_km = _sdk
                                            _snap_idx = _si

                                if _snap_idx is not None:
                                    # Merge into existing facility — add machines there
                                    _sfl, _sfo, _sfw = _opt_locs[_snap_idx]
                                    _opt_locs[_snap_idx] = (_sfl, _sfo, _sfw + _machines_per_new)
                                    _onew_lat, _onew_lon = _sfl, _sfo
                                    _placement_note = f"Merged into existing facility ({_sfl:.3f}, {_sfo:.3f})"
                                else:
                                    _opt_locs.append((_onew_lat, _onew_lon, _machines_per_new))
                                    _placement_note = "New facility"

                                _onew_gdf, _onew_stats = _compute_access(
                                    country, iso3, tuple(_opt_locs),
                                    float(lambda_km), access_model, float(max_distance_km),
                                    capacity_per_machine_per_year, access_rt_method, access_rt_fraction,
                                    h3_resolution, _is_region, snap_linacs_to_hex,
                                    weibull_k=float(weibull_k),
                                    custom_rtu=access_custom_rtu,
                                )
                                _oimprove = _onew_stats["total_rt_treated"] - _opt_stats["total_rt_treated"]
                                _onew_ratio = (
                                    _onew_stats["total_rt_treated"] / _onew_stats["total_rt_demand"]
                                    if _onew_stats["total_rt_demand"] > 0 else 0.0
                                )
                                _opt_suggested.append({
                                    "name": f"Suggested {_opt_step + 1}",
                                    "lat": round(_onew_lat, 4),
                                    "lon": round(_onew_lon, 4),
                                    "n_linacs": _machines_per_new,
                                })
                                _opt_steps.append({
                                    "Step": _opt_step + 1,
                                    "Placement": _placement_note,
                                    "Latitude": round(_onew_lat, 4),
                                    "Longitude": round(_onew_lon, 4),
                                    "RT treated improvement": f"+{_fmt_sigfig(_oimprove)}",
                                    "Cumulative access ratio": f"{_onew_ratio:.1%}",
                                })
                                _opt_gdf = _onew_gdf
                                _opt_stats = _onew_stats
                                _opt_progress.progress(
                                    (_opt_step + 1) / _n_suggest,
                                    text=f"Step {_opt_step + 1}/{_n_suggest} — +{_fmt_sigfig(_oimprove)} patients treated",
                                )

                            _opt_progress.empty()
                            st.session_state["_opt_result"] = {
                                "suggested": _opt_suggested,
                                "steps": _opt_steps,
                                "final_gdf": _opt_gdf,
                                "final_stats": _opt_stats,
                                "final_locs": _opt_locs,
                            }

                        _opt_res = st.session_state.get("_opt_result")
                        if _opt_res:
                            _opt_sug = _opt_res["suggested"]
                            _opt_final_gdf = _opt_res["final_gdf"]
                            _opt_final_stats = _opt_res["final_stats"]
                            _opt_final_locs = _opt_res["final_locs"]

                            # Map showing final state — use same metric as the main display
                            if access_display_metric == "Modelled Access Deficit":
                                _opt_disp_col = "rt_untreated"
                                _opt_cmap_fn = _rdylgn_reversed_rgb
                                _opt_cb_label = "RT access deficit"
                                _opt_tip_label = "RT access deficit"
                                _opt_is_ratio = False
                            elif access_display_metric == "Modelled Accessed":
                                _opt_disp_col = "rt_treated"
                                _opt_cmap_fn = _rdylgn_rgb
                                _opt_cb_label = "RT access"
                                _opt_tip_label = "RT access"
                                _opt_is_ratio = False
                            elif access_display_metric == "Modelled Access Ratio":
                                _opt_disp_col = "capacity_limited_probability"
                                _opt_cmap_fn = _rdylgn_rgb
                                _opt_cb_label = "Modelled Access Ratio"
                                _opt_tip_label = "Access ratio"
                                _opt_is_ratio = True
                            else:  # Geographic Access Probability
                                _opt_disp_col = "access_probability"
                                _opt_cmap_fn = _rdylgn_rgb
                                _opt_cb_label = "Geographic Access Probability"
                                _opt_tip_label = "Geo. access prob"
                                _opt_is_ratio = True

                            _opt_raw = _opt_final_gdf[_opt_disp_col].to_numpy(dtype=np.float64)
                            if _opt_is_ratio:
                                _opt_vmin, _opt_vmax = 0.0, 1.0
                            else:
                                _opt_vmin = float(np.nanmin(_opt_raw[_opt_raw > 0])) if np.any(_opt_raw > 0) else 0.0
                                _opt_vmax = float(np.nanmax(_opt_raw)) if _opt_raw.size > 0 else 1.0
                            _opt_colors, _, _ = _color_values(_opt_raw, _opt_cmap_fn, _opt_vmin, _opt_vmax)
                            _opt_final_gdf = _opt_final_gdf.copy()
                            _opt_final_gdf["color"] = _opt_colors
                            if _opt_is_ratio:
                                _opt_tip_val = (_opt_final_gdf[_opt_disp_col] * 100).round(1).astype(str) + "%"
                            else:
                                _opt_tip_val = _opt_final_gdf[_opt_disp_col].round(1).astype(str)
                            _opt_final_gdf["tip"] = (
                                "<b>" + _opt_final_gdf["h3"].astype(str) + "</b><br/>"
                                + _opt_tip_label + " (optimised): " + _opt_tip_val
                            )
                            _opt_hex_layer = _build_hex_layer(
                                pd.DataFrame({
                                    "h3": _opt_final_gdf["h3"],
                                    "color": _opt_final_gdf["color"],
                                    "tip": _opt_final_gdf["tip"],
                                })
                            )

                            _opt_existing_df = facilities_df.copy() if facilities_df is not None and not facilities_df.empty else pd.DataFrame()
                            _opt_new_df = pd.DataFrame(_opt_sug)

                            # Merge existing + new into one dataframe so stacked mode places
                            # new machines physically on top of existing cylinders at the same site.
                            _opt_combined_parts = []
                            if show_linac_markers and not _opt_existing_df.empty:
                                _ex = _opt_existing_df.copy()
                                _ex["color"] = [[255, 180, 0, 240]] * len(_ex)   # gold
                                _ex["_stack_order"] = 0  # existing → bottom of stack
                                _opt_combined_parts.append(_ex)
                            if not _opt_new_df.empty:
                                _nw = _opt_new_df.copy()
                                _nw["color"] = [[0, 180, 255, 240]] * len(_nw)   # bright blue
                                _nw["_stack_order"] = 1  # new → top of stack
                                _opt_combined_parts.append(_nw)

                            _opt_layers = [_opt_hex_layer]
                            if _opt_combined_parts:
                                _opt_combined_df = pd.concat(_opt_combined_parts, ignore_index=True)
                                _opt_layers.extend(_build_linac_columns(
                                    _opt_combined_df, h3_res=h3_resolution,
                                    country_span_km=country_span_km,
                                    height_scale=tower_height_scale,
                                    radius_scale=tower_radius_scale,
                                    style=linac_tower_style,
                                ))

                            st.caption(f"Map shows **{access_display_metric}** after all suggested LINACs are added. Gold = existing facilities, blue = suggested placements.")
                            _render_with_colorbar(
                                _opt_layers, _make_view(_opt_final_gdf, pitch=30.0 if (show_linac_markers and map_pitch_on) else 0.0),
                                _opt_cmap_fn, _opt_vmin, _opt_vmax, _opt_cb_label,
                                dark=dark_mode, dark_text=app_dark_mode,
                            )

                            st.markdown(f"**{len(_opt_sug)} suggested placement(s)** — improvement over current configuration:")
                            st.dataframe(
                                pd.DataFrame(_opt_res["steps"]),
                                use_container_width=True, hide_index=True,
                            )

                            # Before / after summary
                            _ob_ratio = stats["total_rt_treated"] / stats["total_rt_demand"] if stats["total_rt_demand"] > 0 else 0.0
                            _oa_ratio = _opt_final_stats["total_rt_treated"] / _opt_final_stats["total_rt_demand"] if _opt_final_stats["total_rt_demand"] > 0 else 0.0
                            _ocol1, _ocol2, _ocol3 = st.columns(3)
                            _ocol1.metric("Access ratio — before", f"{_ob_ratio:.1%}")
                            _ocol2.metric("Access ratio — after", f"{_oa_ratio:.1%}", delta=f"+{(_oa_ratio - _ob_ratio):.1%}")
                            _ocol3.metric("Additional patients treated", f"+{_fmt_sigfig(_opt_final_stats['total_rt_treated'] - stats['total_rt_treated'])}")


# ---------------------------------------------------------------------------
# Geography-Only tab
# ---------------------------------------------------------------------------

with tab_geo:
    st.header(f"Geography-Only — {country}")
    st.caption(
        "These geography-only do not consider the capacity constraints."
    )
    _geo_map_result = st.session_state.get("_map_result")
    if _geo_map_result is None:
        st.info("Run **Calculate RT Access** in the sidebar first.")
    else:
        _geo_gdf_out = _geo_map_result["gdf_out"]
        _TT_MAX_ACC = _TT_MAX_BY_RES.get(h3_resolution, 14400) / 60.0
        _use_tt_geo = use_travel_time and "nearest_linac_min" in _geo_gdf_out.columns
        if _use_tt_geo:
            _near_col_geo = "nearest_linac_min"
            _dist_unit_geo = "min"
            _dist_label_geo = "Travel Time to Linac"
            _raw_geo_vals = _geo_gdf_out[_near_col_geo].to_numpy(dtype=np.float64)
            _geo_vals = np.where(np.isfinite(_raw_geo_vals), _raw_geo_vals, _TT_MAX_ACC)
            _has_capped_geo = not np.all(np.isfinite(_raw_geo_vals))
        else:
            _near_col_geo = "nearest_linac_km"
            _dist_unit_geo = "km"
            _dist_label_geo = "Distance to Linac"
            _geo_vals = _geo_gdf_out[_near_col_geo].fillna(0).to_numpy(dtype=np.float64)
            _has_capped_geo = False
            _raw_geo_vals = _geo_vals
        _geo_pop = _geo_gdf_out["population"].to_numpy(dtype=np.float64)
        _geo_median = float(np.median(_geo_vals))
        _gt_geo = "> " if _has_capped_geo and _geo_median >= _TT_MAX_ACC - 0.1 else ""
        _geo_sort_idx = np.argsort(_geo_vals)
        _geo_cum_pop = np.cumsum(_geo_pop[_geo_sort_idx])
        _geo_pop_total = _geo_pop.sum()
        _geo_pw_med_idx = np.searchsorted(_geo_cum_pop, _geo_pop_total * 0.5)
        _geo_pw_median = float(_geo_vals[_geo_sort_idx[min(_geo_pw_med_idx, len(_geo_sort_idx) - 1)]])
        _gt_geo_pw = "> " if _has_capped_geo and _geo_pw_median >= _TT_MAX_ACC - 0.1 else ""
        _geo_col1, _geo_col2 = st.columns(2)
        _geo_col1.metric(f"Median {_dist_label_geo}", f"{_gt_geo}{_geo_median:.1f} {_dist_unit_geo}")
        _geo_col2.metric(f"Pop-Weighted Median {_dist_label_geo}", f"{_gt_geo_pw}{_geo_pw_median:.1f} {_dist_unit_geo}")
        if len(_geo_vals) > 0:
            _geo_h_col1, _geo_h_col2 = st.columns(2)
            with _geo_h_col1:
                _fig_hist = go.Figure()
                _fig_hist.add_trace(go.Histogram(x=_geo_vals, nbinsx=40, marker_color="#4C9BE8"))
                _fig_hist.update_layout(
                    xaxis_title=f"{_dist_label_geo} ({_dist_unit_geo})"
                        + (f" — bar at {int(_TT_MAX_ACC)} includes all travel time > {int(_TT_MAX_ACC)}" if _has_capped_geo else ""),
                    yaxis_title="Number of hexagons",
                    height=240, margin=dict(l=40, r=20, t=30, b=40),
                    title_text=f"Hexagon count by {_dist_label_geo}", showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_hist, use_container_width=True)
            with _geo_h_col2:
                _fig_hist2 = go.Figure()
                _fig_hist2.add_trace(go.Histogram(
                    x=_geo_vals, y=_geo_pop, histfunc="sum", nbinsx=40, marker_color="#F97316",
                ))
                _fig_hist2.update_layout(
                    xaxis_title=f"{_dist_label_geo} ({_dist_unit_geo})"
                        + (f" — bar at {int(_TT_MAX_ACC)} includes all travel time > {int(_TT_MAX_ACC)}" if _has_capped_geo else ""),
                    yaxis_title="Population",
                    height=240, margin=dict(l=40, r=20, t=30, b=40),
                    title_text=f"Population by {_dist_label_geo}", showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_hist2, use_container_width=True)
        if _use_tt_geo and _has_capped_geo:
            _geo_unreachable_mask = ~np.isfinite(_raw_geo_vals)
            _geo_unreachable_pop = float(_geo_pop[_geo_unreachable_mask].sum())
            _geo_unreachable_pct = _geo_unreachable_pop / _geo_pop_total * 100 if _geo_pop_total > 0 else 0.0
            st.caption(
                f"**Unreachable hexes:** {int(_geo_unreachable_mask.sum()):,} hexes · "
                f"population {_fmt_sigfig(_geo_unreachable_pop)} ({_geo_unreachable_pct:.1f}% of total) · "
                f"(Note: TravelTime returns no route when a hex centroid cannot be snapped to the road network, "
                f"e.g. isolated area, lake, or river)."
            )

# ---------------------------------------------------------------------------
# Capacity-Only tab
# ---------------------------------------------------------------------------

with tab_cap:
    st.header(f"Capacity-Only — {country}")
    st.caption(
        "These capacity-only do not consider the geographic constraints."
    )
    with st.spinner("Loading population data…"):
        _cap_pop_gdf = _load_pop_region(country, 3) if _is_region else _load_pop(country, 5)
    _cap_total_pop = int(_cap_pop_gdf["population"].sum())
    _cap_linac_result = _load_dirac(country)
    _cap_facilities_df = _cap_linac_result[1] if _cap_linac_result[0] is not None else pd.DataFrame()
    _n_linacs_cap = int(_cap_facilities_df["n_linacs"].sum()) if _cap_facilities_df is not None and len(_cap_facilities_df) > 0 else 0
    _capacity_per_linac = int(capacity_per_machine_per_year)

    if not has_globocan_data(iso3):
        st.warning(f"No GLOBOCAN data for **{country}** — RT need cannot be estimated.")
    else:
        with st.spinner("Computing RT need…"):
            _rt_need = _data_tab_rt_need(iso3)
        _total_rt_cases = _rt_need["total_rt_cases"]

        st.markdown("**Calculation based on annual cancer incidence and optimal RT utilisation**")
        _linacs_required_incidence = _total_rt_cases / _capacity_per_linac
        _linacs_required_incidence_ceil = math.ceil(_linacs_required_incidence)
        _linac_gap_incidence = _linacs_required_incidence_ceil - _n_linacs_cap
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cancers requiring RT annually", f"{int(_total_rt_cases):,}")
        col2.metric("LINACs (DIRAC)", f"{_n_linacs_cap:,}")
        col3.metric("LINACs required (450 pts/yr/LINAC)", f"{_linacs_required_incidence_ceil:,}")
        _gap_inc_label = "LINAC shortage" if _linac_gap_incidence > 0 else "LINAC surplus"
        _gap_inc_color = "red" if _linac_gap_incidence > 0 else "green"
        col4.markdown(
            f"<div><span style='display:block;font-size:0.875rem;color:#808495;margin-bottom:0.25rem'>{_gap_inc_label}</span>"
            f"<span style='display:block;font-size:2rem;font-weight:600;line-height:1;color:{_gap_inc_color}'>{abs(_linac_gap_incidence):,}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Incidence-based RT need estimated by multiplying GLOBOCAN 2022 cancer incidence by optimal RT utilisation rates "
            "(Delaney et al. 2005) for each cancer site independently. "
            f"Capacity assumed at **{_capacity_per_linac} patients per LINAC per year** "
            "([Abdel-Wahab et al. 2025](https://doi.org/10.1016/S1470-2045(24)00678-8))."
        )

        st.markdown("**Calculation based on annual cancer incidence and proportional scaling**")
        _prop_fraction = access_rt_fraction if access_rt_method == "proportional" else 0.25
        _total_rt_cases_prop = _rt_need["total_cancer_excl_nmsc"] * _prop_fraction if "total_cancer_excl_nmsc" in _rt_need else _total_rt_cases
        _linacs_required_prop = _total_rt_cases_prop / _capacity_per_linac
        _linacs_required_prop_ceil = math.ceil(_linacs_required_prop)
        _linac_gap_prop = _linacs_required_prop_ceil - _n_linacs_cap
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cancers requiring RT annually", f"{int(_total_rt_cases_prop):,}")
        col2.metric("LINACs (DIRAC)", f"{_n_linacs_cap:,}")
        col3.metric("LINACs required (450 pts/yr/LINAC)", f"{_linacs_required_prop_ceil:,}")
        _gap_prop_label = "LINAC shortage" if _linac_gap_prop > 0 else "LINAC surplus"
        _gap_prop_color = "red" if _linac_gap_prop > 0 else "green"
        col4.markdown(
            f"<div><span style='display:block;font-size:0.875rem;color:#808495;margin-bottom:0.25rem'>{_gap_prop_label}</span>"
            f"<span style='display:block;font-size:2rem;font-weight:600;line-height:1;color:{_gap_prop_color}'>{abs(_linac_gap_prop):,}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Incidence-based RT need estimated by multiplying GLOBOCAN 2022 cancer incidence (excl. NMSC) by a "
            f"proportional scaling factor of **{_prop_fraction:.2f}**. "
            f"Capacity assumed at **{_capacity_per_linac} patients per LINAC per year** "
            "([Abdel-Wahab et al. 2025](https://doi.org/10.1016/S1470-2045(24)00678-8))."
        )

        st.markdown("**Calculation based on 5 machines per million of population**")
        _linacs_required_pop = _cap_total_pop / 1_000_000 * 5
        _linacs_required_pop_ceil = math.ceil(_linacs_required_pop)
        _linac_gap_pop_cap = _linacs_required_pop_ceil - _n_linacs_cap
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Population", f"{_cap_total_pop:,}")
        col2.metric("LINACs (DIRAC)", f"{_n_linacs_cap:,}")
        col3.metric("LINACs required (5 per million pop.)", f"{_linacs_required_pop_ceil:,}")
        _gap_pop_cap_label = "LINAC shortage" if _linac_gap_pop_cap > 0 else "LINAC surplus"
        _gap_pop_cap_color = "red" if _linac_gap_pop_cap > 0 else "green"
        col4.markdown(
            f"<div><span style='display:block;font-size:0.875rem;color:#808495;margin-bottom:0.25rem'>{_gap_pop_cap_label}</span>"
            f"<span style='display:block;font-size:2rem;font-weight:600;line-height:1;color:{_gap_pop_cap_color}'>{abs(_linac_gap_pop_cap):,}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Population-based benchmark: 5 LINACs per million population "
            "([IAEA DIRAC Database](https://dirac.iaea.org/))."
        )

# ---------------------------------------------------------------------------
# Method tab
# ---------------------------------------------------------------------------

with _tab_sep1:
    pass

with _tab_sep2:
    pass

# ---------------------------------------------------------------------------
# Introduction tab
# ---------------------------------------------------------------------------

with tab_intro:
    st.header("Introduction")

    st.subheader("About RadMaps")
    st.markdown(
        """
        RadMaps is open source and released under the
        [MIT License](https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access/blob/main/LICENSE).
        Source code is available on [GitHub](https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access).
        """
    )
    st.markdown(
        """
        Approximately half of all cancer cases require radiotherapy, yet worldwide access
        to RT remains unacceptably low. 

        Access to RT is constrained by two principal factors:

        - **Machine capacity** — the finite number of linear accelerators (linacs) within a
          country limits the total number of patients that can be treated each year.
        - **Geographic access** — RT treatment typically requires visiting a hospital every day over a period
          of a few weeks. Patients that live far from a facility may experience reduced treatment outcomes 
          ([Silverwood et al. 2024](https://doi.org/10.1016/j.adro.2024.101652)).

        Previous work has addressed each of these factors independently (e.g. capacity 
        ([Abdel-Wahab *et al.* 2025](https://doi.org/10.1016/S1470-2045(24)00678-8) and geography 
        ([Wawrzuta *et al.* 2025](https://doi.org/10.1016/j.radonc.2025.111061))). 
        
        RadMaps illuminate both constraints, and provide a measure of access based on both of them. 
        It provides fast visualisation and analysis of access to radiotherapy (RT) at
        the sub-national scale, within countries and regions. 
        

        
        """
    )

    st.subheader("What each tab does")
    st.markdown(
        """
        | Tab | Contents |
        |---|---|
        | **🗺️ Map Modelling** | Interactive H3 hexagon maps — population density, cancer burden, RT demand, geographic access probability, and capacity-limited access. Select a country or region in the sidebar and click **Generate Map**. |
        | **📊 Data** | Country-level data tables — cancer incidence by site (GLOBOCAN 2022), LINAC locations (DIRAC), and RT utilisation rates. |
        | **🌍 Geography-Only** | Distance/travel-time distributions to nearest LINAC after running Calculate RT Access. |
        | **⚡ Capacity-Only** | Headline LINAC gap estimates (optimal RTU, proportional, and population benchmarks) without geographic constraints. |
        | **📖 Method** | Full pipeline description with flowchart, data sources, and step-by-step methodology. |
        | **⚠️ Assumptions** | Tabulated model assumptions and limitations, ranked by likely impact, with suggested improvements. |
        | **🧪 Toy Example** | Step-by-step worked example showing how each pipeline stage transforms inputs into access outputs. |
        | **📐 Probability Models** | Explanation and visualisation of the four distance-decay models (exponential, Weibull, step function, uniform) used to compute geographic access probability. |
        """
    )

    st.subheader("Quick start guide")
    st.markdown(
        """
        1. **Select a country or region** from the dropdown in the left sidebar. Countries
           are limited to those with GLOBOCAN cancer incidence data. Regions (Africa, Europe,
           etc.) are also available at lower resolutions.

        2. **Choose a map type** — start with *Population Data* (then select *Population Density*)
           to see the underlying data, then *Cancer Incidence* or *Radiotherapy Demand* under
           the same dropdown to see cancer burden, then switch to *Radiotherapy Access* to see
           the combined model output.

        3. **Set the H3 resolution** — the map is built on an
           [H3 hexagonal grid](https://h3geo.org/). Resolution 8 (~400 m hexagons) gives the
           most detail for single countries; lower resolutions (3–5) are faster and better
           suited to regions.

        4. **Click Generate Map** — the first load for a new country downloads the Kontur
           population file (~1–60 seconds depending on country size); subsequent loads are
           instant.

        5. **Explore the access model** — under *Radiotherapy Access*, adjust the distance-
           decay model (exponential, step, or uniform), the decay parameter λ, and the
           capacity per machine to see how results change.

        6. **Check the Data tab** for country-level cancer and LINAC statistics, and the
           **⚡ Capacity-Only** tab for a headline capacity gap estimate independent of
           geographic constraints.
        """
    )

# ---------------------------------------------------------------------------
# Method tab
# ---------------------------------------------------------------------------

with tab_method:
    st.header("Method")

    # ------------------------------------------------------------------
    # Method / flowchart
    # ------------------------------------------------------------------
    st.subheader("Model Overview")

    import os as _os
    _flowchart_path = _os.path.join(_os.path.dirname(__file__), "assets", "flowchart.png")
    if _os.path.exists(_flowchart_path):
        st.image(_flowchart_path, width="stretch")

    st.markdown(
        """
        The model pipeline proceeds as follows:

        1. **Population density** — sub-national population is sourced from the
           [Kontur Population Dataset](https://www.kontur.io/portfolio/population-dataset/)
           (aggregated H3 hexagonal grid, ~400 m resolution at level 8). Each hexagon
           represents an area unit for all subsequent calculations.

        2. **Cancer incidence** — national cancer incidence figures are taken from
           [GLOBOCAN 2022](https://gco.iarc.fr/today/) (IARC). These are apportioned to
           individual hexagons in proportion to their population, under the assumption of
           spatially uniform cancer incidence rates (see Assumptions below).

        3. **Radiotherapy demand** — the number of patients requiring RT in each hexagon is
           estimated either by applying site-specific optimal RT utilisation fractions
           (Delaney *et al.* 2005) to each cancer type, or by applying a user-specified
           proportional rate to all cancers excluding non-melanoma skin cancer (NMSC, RT
           utilisation ≈ 0%).

        4. **Linac locations and capacity** — facility locations and machine counts are
           sourced from the
           [DIRAC database](https://dirac.iaea.org/) (IAEA). Each linac is assumed to treat
           a fixed number of patients per year (default: 450), giving a total national
           capacity.

        5. **Geographic access probability** — for each hexagon, the probability that a
           patient reaches *any* facility is computed as:

           $$P_{\\text{geo}} = 1 - \\prod_{i} \\left(1 - p(d_i)\\right)$$

           where $d_i$ is the straight-line distance to facility $i$ and $p(d_i)$ is the
           probability model (exponential decay, step function, or uniform). This metric
           is independent of capacity.

        6. **Capacity-limited (modelled) access** — linac capacity is allocated using a
           ring-based proportional algorithm: for each facility, hexagons are grouped into
           concentric rings of equal distance. Each ring is served in full before moving
           outward; if a ring would exhaust the facility's remaining capacity, that capacity
           is distributed *proportionally* across all hexagons in the ring by their demand
           weight. No hexagon receives more than its outstanding demand.
           The resulting ratio of treated to total demand per hexagon gives the **Modelled
           Access Probability**.
        """
    )


# ---------------------------------------------------------------------------
# Assumptions tab
# ---------------------------------------------------------------------------

with tab_assumptions:
    st.header("Assumptions and Limitations")
    st.markdown(
        "The model contains a number of simplifying assumptions. "
        "These are divided below by their likely impact on results."
    )

    import pandas as _pd

    _more_significant = _pd.DataFrame([
        {
            "Assumption": "Uniform cancer incidence",
            "Limitation": "Cancer incidence assumed proportional to population density; demographic and geographic variation not captured.",
            "To Improve": "Incorporate sub-national cancer incidence data at H3 resolution where available.",
        },
        {
            "Assumption": "No patient stratification",
            "Limitation": "All cancer patients treated as equivalent. Access barriers differ by age, mobility, socioeconomic status, cancer stage, and RT modality required.",
            "To Improve": "Stratify demand by cancer type, stage, and demographic; incorporate access modifiers where data permit.",
        },
        {
            "Assumption": "Probability model for geographic access",
            "Limitation": "The distance–RT uptake relationship is poorly characterised. No single model is universally accepted (Perez et al. 2016; Lin et al. 2015; Yap et al. 2023).",
            "To Improve": "Incorporate empirically validated, country-specific probability models.",
        },
        {
            "Assumption": "Greedy nearest-first allocation",
            "Limitation": "Assumes each facility serves its nearest patients first. Real referral patterns depend on clinical pathways, waiting times, and patient choice.",
            "To Improve": "Travel-time routing; incorporate referral pathway data where available.",
        }, 
    ])

    _less_significant = _pd.DataFrame([
        {
            "Assumption": "Incident (new) cancer cases only",
            "Limitation": "Demand based on new cases per year. Prevalent cases requiring re-treatment or delayed RT are excluded, so demand may be underestimated.",
            "To Improve": "Apply a correction factor based on the proportion of prevalent cases requiring RT.",
        },
        {
            "Assumption": "Linacs only",
            "Limitation": "Brachytherapy, orthovoltage, proton therapy, and other modalities excluded.",
            "To Improve": "Linacs dominate external-beam RT; fractional corrections for other modalities could be added.",
        },
        {
            "Assumption": "Equal weighting of facilities",
            "Limitation": "All facilities within range weighted by distance only. Referral networks may make distant specialist centres effectively inaccessible.",
            "To Improve": "Incorporate referral pathway data to weight facility accessibility.",
        },
        {
            "Assumption": "Static snapshot",
            "Limitation": "GLOBOCAN incidence and DIRAC machine counts are point-in-time. Population growth, ageing, and planned facilities are not modelled.",
            "To Improve": "Allow temporal projection using demographic growth rates and infrastructure pipelines.",
        },
        {
            "Assumption": "National boundaries as hard limits",
            "Limitation": "Cross-border access not modelled. Patients in small countries or border regions may realistically travel abroad for treatment.",
            "To Improve": "Allow cross-border facility access for hexagons within the distance cutoff of a foreign facility.",
        },
        {
            "Assumption": "Private and public facilities treated equally",
            "Limitation": "DIRAC includes private facilities, but access to private machines is not universal. Effective access may be lower than modelled.",
            "To Improve": "Allow the user to flag or exclude private facilities based on healthcare system context.",
        },
        {
            "Assumption": "Data quality",
            "Limitation": "DIRAC machine locations may be out of date or contain coordinate errors. GLOBOCAN data unavailable for some countries (e.g. Mongolia).",
            "To Improve": "Validate and correct data sources as errors are identified.",
        },
        {
            "Assumption": "Modifiable Areal Unit Problem (MAUP)",
            "Limitation": "All spatial estimates depend on the chosen H3 resolution. Aggregating data into larger hexagons smooths local variation and can change apparent patterns — a known issue in areal statistics.",
            "To Improve": "Examine results at multiple resolutions; report sensitivity to resolution choice.",
        },
    ])

    st.markdown("##### More Significant Assumptions")
    st.dataframe(_more_significant, use_container_width=True, hide_index=True)

    st.markdown("##### Less Significant Assumptions")
    st.dataframe(_less_significant, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Toy Example tab
# ---------------------------------------------------------------------------

with tab_toy:
    st.header("Toy Example")
    st.markdown(
        """
        The following figures walk through each stage of the model pipeline using a
        simplified toy scenario. This illustrates how the inputs are transformed into
        the final access outputs step by step.
        """
    )

    _toy_dir = _os.path.join(_os.path.dirname(__file__), "assets", "toy_example")

    _toy_figures = [
        (
            "PopulationDensity.png",
            "Step 1 — Population Density",
            "The spatial distribution of population across the region, sourced from the "
            "Kontur H3 dataset. Each hexagon represents the number of population living within "
            "that cell. This forms the base layer for all subsequent calculations.",
        ),
        (
            "AnnualNewCancerDensity.png",
            "Step 2 — Cancer Incidence",
            "National cancer incidence figures (GLOBOCAN) are apportioned to each hexagon "
            "in proportion to its population. This gives an estimate of the number of new "
            "cancer cases arising in each cell each year.",
        ),
        (
            "CancerCasesRequiringRT.png",
            "Step 3 — Cancer Cases Requiring Radiotherapy",
            "Each cancer type is multiplied by its site-specific optimal radiotherapy "
            "utilisation fraction (Delaney et al. 2005) and the results summed per hexagon. "
            "This gives the estimated number of patients in each cell who require RT annually.",
        ),
        (
            "GeographicProbability.png",
            "Step 4 — Geographic Access Probability",
            "For each hexagon, the probability that a patient can reach at least one facility "
            "is computed using the selected distance-decay model, combining contributions from "
            "all linacs. This is independent of machine capacity.",
        ),
        (
            "LinacCapacity.png",
            "Step 5 — Linac Capacity Allocation",
            "Machine capacity is distributed using a greedy nearest-first algorithm. Each "
            "linac fills its annual capacity by serving the nearest hexagons first, working "
            "outward until capacity is exhausted. The proportion of demand met in each "
            "hexagon gives the Modelled Access Ratio.",
        ),
        (
            "Untreated.png",
            "Step 6 — Modelled Inaccessible Patients",
            "The difference between RT demand and allocated capacity in each hexagon gives "
            "the estimated number of patients who cannot access treatment. This highlights "
            "which areas are most underserved, whether due to distance or capacity shortfall.",
        ),
    ]

    for fname, heading, caption in _toy_figures:
        fpath = _os.path.join(_toy_dir, fname)
        st.subheader(heading)
        if _os.path.exists(fpath):
            st.image(fpath, width="stretch")
        else:
            st.warning(f"Image not found: {fname}")
        st.caption(caption)
        st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "RadMaps · Released under the [MIT License](https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access/blob/main/LICENSE) · "
    "[GitHub](https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access)"
)
