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
from data.travel_time import compute_travel_time_matrix, CACHE_DIR as _TT_CACHE_DIR


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RadMaps",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
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
):
    gdf = _load_pop_region(country, h3_res) if region_flag else _load_pop(country, h3_res)

    # Build RT demand per hex from cancer data
    demand = None
    total_cancer_excl_nmsc = None
    try:
        if rt_method == "optimal":
            # Load all individual sites + derived types; multiply each by its own RT fraction
            all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES
            cancer_gdf = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False)
            excl_col = "All cancers excl. NMSC_incidence"
            if excl_col in cancer_gdf.columns:
                total_cancer_excl_nmsc = float(cancer_gdf[excl_col].clip(lower=0).sum())
            rt_cols = [
                c for c in cancer_gdf.columns
                if c.endswith("_optimal_rt")
                and c[:-len("_optimal_rt")].strip().lower() not in _AGGREGATE_CANCER_KEYS
            ]
            if rt_cols:
                demand = cancer_gdf[rt_cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)
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
):
    """Like _compute_access but loads a pre-computed travel time matrix from disk."""
    gdf = _load_pop_region(country, h3_res) if region_flag else _load_pop(country, h3_res)

    demand = None
    total_cancer_excl_nmsc = None
    try:
        if rt_method == "optimal":
            all_cancers = get_cancer_types() + DERIVED_CANCER_TYPES
            cancer_gdf = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False)
            excl_col = "All cancers excl. NMSC_incidence"
            if excl_col in cancer_gdf.columns:
                total_cancer_excl_nmsc = float(cancer_gdf[excl_col].clip(lower=0).sum())
            rt_cols = [
                c for c in cancer_gdf.columns
                if c.endswith("_optimal_rt")
                and c[:-len("_optimal_rt")].strip().lower() not in _AGGREGATE_CANCER_KEYS
            ]
            if rt_cols:
                demand = cancer_gdf[rt_cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)
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
    tt_matrix = np.load(cache_file)["matrix"] if cache_file.exists() else None

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

def _build_hex_layer(df: pd.DataFrame) -> pdk.Layer:
    return pdk.Layer(
        "H3HexagonLayer",
        data=df,
        get_hexagon="h3",
        get_fill_color="color",
        auto_highlight=True,
        pickable=True,
        opacity=0.7,
    )


_LINAC_COLORS = [
    [30, 120, 220, 220],   # blue
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
        max(hex_radius_km * 1000 * 0.6, country_span_km * 1000 * 0.0008) * height_scale
    )

    if style == "individual":
        rows = []
        for i, row_data in facilities_df.reset_index(drop=True).iterrows():
            rows.append({
                "lat": float(row_data["lat"]),
                "lon": float(row_data["lon"]),
                "elevation": float(row_data["n_linacs"]) * elevation_per_linac,
                "color": _LINAC_COLORS[i % len(_LINAC_COLORS)],
                "tip": (
                    f"<b>{row_data['name']}</b><br/>"
                    f"{row_data['city']}<br/>"
                    f"{int(row_data['n_linacs'])} LINAC{'s' if row_data['n_linacs'] != 1 else ''}"
                ),
            })
        ind_df = pd.DataFrame(rows)
        return [pdk.Layer(
            "ColumnLayer",
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

    # tiers[i] = rows for the i-th facility slot (sorted by n_linacs desc) across all hexes
    tiers: dict = {}
    for hex_id, group in df.groupby("hex_id"):
        group = group.sort_values("n_linacs", ascending=False).reset_index(drop=True)
        hc = h3.cell_to_latlng(hex_id)  # (lat, lon)
        cum = 0.0
        for i, row_data in group.iterrows():
            cum += float(row_data["n_linacs"]) * elevation_per_linac
            tiers.setdefault(i, []).append({
                "lat": hc[0],
                "lon": hc[1],
                "cum_height": cum,
                "color": _LINAC_COLORS[i % len(_LINAC_COLORS)],
                "tip": (
                    f"<b>{row_data['name']}</b><br/>"
                    f"{row_data['city']}<br/>"
                    f"{int(row_data['n_linacs'])} LINAC{'s' if row_data['n_linacs'] != 1 else ''}"
                ),
            })

    # Render from HIGHEST tier index to 0. depthMask=False so shorter layers
    # always paint over the bottom of taller ones (stacked-bar effect).
    layers = []
    for tier_idx in sorted(tiers.keys(), reverse=True):
        tier_df = pd.DataFrame(tiers[tier_idx])
        layers.append(pdk.Layer(
            "ColumnLayer",
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
    """Format a number to *sig* significant figures with k/M suffix."""
    if value is None:
        return "N/A"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1_000_000:
        scaled = value / 1_000_000
        rounded = float(f"{scaled:.{sig}g}")
        return f"{rounded:g} M"
    if abs(value) >= 1_000:
        scaled = value / 1_000
        rounded = float(f"{scaled:.{sig}g}")
        return f"{rounded:g} k"
    rounded = float(f"{value:.{sig}g}")
    return f"{rounded:g}"


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
        parts.append(f"hex diameter ≈ {hex_diam:.1f} km")
    return " · ".join(parts)


def _make_view(gdf, pitch: float = 0.0) -> pdk.ViewState:
    geom = gdf.geometry
    cx = float(geom.centroid.x.mean())
    cy = float(geom.centroid.y.mean())
    span_lat = float(geom.bounds["maxy"].max() - geom.bounds["miny"].min())
    zoom = max(3, min(8, int(8 - np.log2(max(span_lat, 0.5)))))
    return pdk.ViewState(latitude=cy, longitude=cx, zoom=zoom, pitch=pitch, bearing=0)


CARTO_LIGHT = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
CARTO_DARK  = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


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
) -> None:
    if not isinstance(layers, list):
        layers = [layers]
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=CARTO_DARK if dark else CARTO_LIGHT,
        tooltip={"html": "{tip}"},
    )
    col_map, col_cb = st.columns([7, 1])
    with col_map:
        st.pydeck_chart(deck, use_container_width=True)
    with col_cb:
        fig = _colorbar_fig(cmap_fn, vmin, vmax, cb_label, log_scale=log_scale, text_color="white" if (dark or dark_text) else "black", clamp=clamp)
        st.pyplot(fig, use_container_width=True)
        # if show_linac_legend:
        #     st.pyplot(_linac_legend_fig(dark=dark), use_container_width=True)


def _h3_caption(gdf) -> str:
    res = h3.get_resolution(str(gdf["h3"].iloc[0]))
    try:
        area = h3.average_hexagon_area(res, unit="km^2")
        return (
            f"H3 resolution {res} — ~{area:.2f} km² per hexagon | "
            f"{len(gdf):,} hexagons  ·  Empty areas = no Kontur population data"
        )
    except Exception:
        return f"H3 resolution {res} | {len(gdf):,} hexagons"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

MAP_TYPES = [
    "Population Density",
    "Cancer Incidence",
    "Radiotherapy Demand",
    "Radiotherapy Access",
    "Nearest Linac",
]

with st.sidebar:
    st.title("🏥 RadMaps")

    _all_options = _selection_options()
    country = st.selectbox(
        "Country / Region",
        options=_all_options,
        index=_all_options.index("United Kingdom") if "United Kingdom" in _all_options else 0,
    )
    _is_region = is_region(country)

    map_type = st.selectbox("Map type", MAP_TYPES, index=MAP_TYPES.index("Radiotherapy Access"))
    if map_type == "Cancer Incidence":
        st.caption("Data: GLOBOCAN 2022 estimates")

    if _is_region:
        _reg_def = get_region(country)
        _res_opts = [r for r in [1, 2, 3] if r <= _reg_def.max_resolution]
        _res_labels = {
            1: "H1 (~2.5M km²)", 2: "H2 (~87k km²)", 3: "H3 (~12,400 km²)",
        }
        h3_resolution = st.selectbox(
            "H3 resolution",
            options=_res_opts,
            index=min(1, len(_res_opts) - 1),  # default H2
            format_func=lambda r: _res_labels.get(r, str(r)),
            key="h3_res_region",
        )
    else:
        h3_resolution = st.selectbox(
            "H3 resolution",
            options=[8, 7, 6, 5, 4, 3],
            index=3,  # H5 default
            format_func=lambda r: {
                8: "H8 (~0.7 km²)", 7: "H7 (~5 km²)", 6: "H6 (~36 km²)",
                5: "H5 (~253 km²)", 4: "H4 (~1,770 km²)", 3: "H3 (~12,400 km²)",
            }[r],
            key="h3_res_country",
        )

    is_rt_demand_map = map_type == "Radiotherapy Demand"
    is_cancer = map_type in ("Cancer Incidence", "Radiotherapy Demand")
    is_access = map_type == "Radiotherapy Access"
    is_nearest = map_type == "Nearest Linac"
    needs_linac = is_access or is_nearest

    # ------ Cancer controls -----------------------------------------------
    selected_cancers: List[str] = []
    use_actual = False
    rt_method: str = "optimal"       # for "Cancer cases requiring RT" map
    rt_fraction: float = 0.25        # for proportional RT
    access_rt_method: str = "optimal"  # for Radiotherapy Access demand source
    access_rt_fraction: float = 0.25

    if is_cancer:
        all_cancers = get_cancer_types()
        # Non-aggregate individual site names
        _site_cancers = [c for c in all_cancers if c.strip().lower() not in _AGGREGATE_CANCER_KEYS]
        _site_cancers_no_nmsc = [
            c for c in _site_cancers
            if "non-melanoma" not in c.lower() and "nmsc" not in c.lower()
        ]

        _rt_scope_options = (
            ["All cancers", "Specific cancer site(s)"]
            if is_rt_demand_map
            else ["All cancers", "All cancers excl. NMSC", "Specific cancer site(s)"]
        )
        _map_cancer_view = st.radio(
            "Cancer scope",
            _rt_scope_options,
            index=0,
            horizontal=False,
            key="map_cancer_view",
        )
        if _map_cancer_view == "All cancers":
            selected_cancers = ["All cancers"]
        elif _map_cancer_view == "All cancers excl. NMSC":
            selected_cancers = ["All cancers excl. NMSC"]
        else:
            selected_cancers = st.multiselect(
                "Cancer site(s)",
                options=_site_cancers,
                default=[_site_cancers[0]] if _site_cancers else [],
                key="map_cancer_sites",
            )
        if is_rt_demand_map:
            _rt_label = st.radio(
                "RT demand method",
                ["Optimal RT (utilisation rates)", "Proportional RT (simple scaling)"],
                horizontal=False,
            )
            rt_method = "optimal" if "Optimal" in _rt_label else "proportional"
            if rt_method == "proportional":
                rt_fraction = st.slider(
                    "Fraction of cancer cases needing RT",
                    min_value=0.01, max_value=1.0, value=0.25, step=0.01,
                    format="%.2f",
                )

    # ------ Access model controls (outside expander — affect computation) -----
    lambda_km: float = 30.0
    max_distance_km: float = 100.0
    weibull_k: float = 2.0
    access_model: str = "weibull"
    access_display_metric: str = "Modelled Inaccessible"
    capacity_per_machine_per_year: float = 450.0
    use_travel_time: bool = False
    tt_mode: str = "driving"
    tt_app_id: str = ""
    tt_api_key: str = ""

    if needs_linac:
        # --- Distance method (shown for both Access and Nearest Linac maps) ---
        tt_method = st.radio(
            "Distance method",
            ["Straight-line distance", "Driving time", "Public transport time"],
            index=0,
            horizontal=True,
        )
        use_travel_time = tt_method != "Straight-line distance"
        tt_mode = "driving" if tt_method == "Driving time" else "public_transport"
        if use_travel_time:
            st.markdown(
                "**TravelTime API credentials** — [get a free key](https://traveltime.com/)"
            )
            tt_app_id = st.text_input("App ID", key="tt_app_id", type="default")
            tt_api_key = st.text_input("API Key", key="tt_api_key", type="password")

    if is_access:
        capacity_per_machine_per_year = float(
            st.slider(
                "Patients treated per machine per year",
                min_value=50, max_value=1000, value=450, step=50,
            )
        )
        _acc_rt_label = st.radio(
            "RT demand source",
            ["Optimal RT (utilisation rates)", "Proportional RT (simple scaling)"],
            horizontal=False,
        )
        access_rt_method = "optimal" if "Optimal" in _acc_rt_label else "proportional"
        if access_rt_method == "proportional":
            access_rt_fraction = st.slider(
                "Fraction of cancer cases needing RT",
                min_value=0.01, max_value=1.0, value=0.25, step=0.01,
                format="%.2f",
                key="access_rt_fraction",
            )

        access_display_metric = st.selectbox(
            "Display metric",
            [
                "Modelled Inaccessible",
                "Modelled Accessed",
                "Modelled Access Ratio",
                "Geographic Access Probability",
            ],
            index=0,
        )
        model_label = st.radio(
            "Access model",
            ["Weibull", "Step function", "Uniform (no decay)"],
            index=0,
            horizontal=True,
        )
        access_model = {
            "Weibull": "weibull",
            "Step function": "step",
            "Uniform (no decay)": "uniform",
        }[model_label]

        _unit = "min" if use_travel_time else "km"
        if access_model == "weibull":
            lambda_km = float(st.slider(f"Scale λ ({_unit})  —  P(λ) = 37%", 5, 200, 60 if use_travel_time else 150, step=5))
            weibull_k = float(st.slider("Shape k  —  higher = steeper (k=1 = exponential)", 1.0, 6.0, 4.0, step=0.5))
        elif access_model == "step":
            max_distance_km = float(
                st.slider(f"Max treatment {'time' if use_travel_time else 'distance'} ({_unit})", 10, 500, 60 if use_travel_time else 100, step=10)
            )

    # ------ Plot settings (expander) ---------------------------------------
    app_dark_mode: bool = False
    dark_mode: bool = False
    show_linac_markers: bool = False
    tower_height_scale: float = 1.0
    tower_radius_scale: float = 1.0
    linac_tower_style: str = "stacked"
    snap_linacs_to_hex: bool = False
    _default_log = map_type in ("Population Density", "Cancer Incidence")

    st.divider()
    show_linac_markers = st.checkbox("Show LINAC locations", value=needs_linac)
    if is_access:
        _use_latlng = st.checkbox(
            "Use LINAC lat/long coords",
            value=True,
            help="When enabled, the exact LINAC coordinates from DIRAC are used. "
                 "When disabled, each LINAC is projected to its H3 hex centroid at the "
                 "current resolution and co-located LINACs are merged.",
        )
        snap_linacs_to_hex = not _use_latlng

    with st.expander("⚙️ Plot settings"):
        app_dark_mode = st.toggle("Dark background", value=False)
        _apply_app_dark_mode(app_dark_mode)
        dark_mode = st.checkbox("Dark map", value=False)

        if show_linac_markers:
            st.divider()
            tower_height_scale = float(
                st.slider("Tower height scale", min_value=0.05, max_value=5.0, value=1.0, step=0.05)
            )
            tower_radius_scale = float(
                st.slider("Tower radius scale", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            )
            _tower_style_label = st.radio(
                "Tower style",
                ["Individual (one tower per centre)", "Stacked (segmented per centre)"],
                horizontal=False,
            )
            linac_tower_style = "individual" if "Individual" in _tower_style_label else "stacked"
            st.divider()

        _default_cmap_name = _DEFAULT_CMAP.get(map_type, "Purple → Yellow (Viridis)")
        _cmap_options = list(COLORMAPS.keys()) + ([BINARY_CMAP_NAME] if map_type == "Nearest Linac" else [])
        cb_cmap_name = st.selectbox(
            "Colour scheme",
            options=_cmap_options,
            index=_cmap_options.index(_default_cmap_name),
        )
        _binary_cmap = (cb_cmap_name == BINARY_CMAP_NAME)
        cb_cmap_fn = COLORMAPS.get(cb_cmap_name, _viridis_rgb)
        cb_log = st.checkbox("Log scale", value=_default_log)
        _binary_threshold = 60.0  # default; overridden below if binary selected
        if _binary_cmap:
            _binary_threshold = st.number_input(
                "Threshold (green if below)", min_value=0.0, value=60.0, step=5.0,
                help="Hexagons below this value are coloured green, above are red. Units match the distance/time measure in use.",
            )

        _count_maps = {
            "Population Density", "Cancer Incidence", "Radiotherapy Demand",
        }
        _count_access_metrics = {
            "Modelled Accessed", "Modelled Inaccessible",
        }
        _supports_per_km2 = (
            map_type in _count_maps
            or (is_access and access_display_metric in _count_access_metrics)
        )
        density_per_km2: bool = False
        if _supports_per_km2:
            _density_radio = st.radio(
                "Colour scale normalisation", ["Per hexagon", "Per 10 km²"],
                index=1, horizontal=True,
            )
            density_per_km2 = _density_radio == "Per 10 km²"

        cb_auto = st.checkbox("Auto range", value=True)
        cb_vmin_user: Optional[float] = None
        cb_vmax_user: Optional[float] = None
        if not cb_auto:
            cb_vmin_user = st.number_input("Min value", value=0.0, format="%.4g")
            cb_vmax_user = st.number_input("Max value", value=1.0, format="%.4g")
    if st.button("Generate Map", type="primary", use_container_width=True):
        st.session_state["_map_generated"] = True
        st.session_state["_switch_to_map_tab"] = True
    generate = st.session_state.get("_map_generated", False)
    st.markdown(
        "<div style='text-align:center;font-size:0.75rem;color:gray;margin-top:-8px'>"
        "Press again if hexagons do not render</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers used in map sections
# ---------------------------------------------------------------------------

def _color_values(values: np.ndarray, cmap_fn, auto_vmin: float, auto_vmax: float):
    """Apply colormap using user or auto range, optionally in log space."""
    vmin = cb_vmin_user if not cb_auto and cb_vmin_user is not None else auto_vmin
    vmax = cb_vmax_user if not cb_auto and cb_vmax_user is not None else auto_vmax
    if cb_log:
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

tab_map, tab_data, _tab_sep1, tab_intro, tab_method, tab_assumptions, _tab_sep2, tab_toy, tab_model = st.tabs([
    "🗺️ Map Modelling", "📊 Data", "│", "💡 Introduction", "📖 Method", "⚠️ Assumptions", "│", "🧪 Toy Example", "📐 Probability Models",
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
        "Source: [Kontur Population Dataset 2023](https://data.humdata.org/dataset/kontur-population-dataset) "
        "— population modelled at 400 m H3 resolution from GHSL, OSM, and census data."
    )

    st.divider()

    # ---- Annual cancer incidence -------------------------------------------
    st.subheader(f"Cancer Incidence — {country}")
    if not has_globocan_data(iso3):
        st.warning(f"**{country}** (ISO3: {iso3}) is not present in the GLOBOCAN dataset.")
    else:
        with st.spinner("Loading cancer data…"):
            _cancer_df = _data_tab_cancer(iso3)

        # Extract aggregate rows for headline metrics
        _agg_mask = _cancer_df["Cancer type"].str.strip().str.lower().isin(_AGGREGATE_CANCER_KEYS)
        _agg_rows = _cancer_df[_agg_mask].set_index(_cancer_df[_agg_mask]["Cancer type"].str.strip().str.lower())
        _site_df = _cancer_df[~_agg_mask].copy()

        _all_cancers_n = int(_agg_rows.loc["all cancers", "New cases"]) if "all cancers" in _agg_rows.index else _site_df["New cases"].sum()
        _nmsc_n = int(_agg_rows.loc["all cancers excl. nmsc", "New cases"]) if "all cancers excl. nmsc" in _agg_rows.index else None

        _cancer_view = st.radio(
            "View",
            ["All cancers", "All cancers excl. NMSC", "Specific cancer site(s)"],
            index=0,
            horizontal=True,
            key="data_cancer_view",
        )

        if _cancer_view == "All cancers":
            _headline_n = _all_cancers_n
            _headline_label = "Cancer Incidence: All Sites"
        elif _cancer_view == "All cancers excl. NMSC":
            _headline_n = _nmsc_n if _nmsc_n is not None else _all_cancers_n
            _headline_label = "Cancer Incidence: All Cancers excl. NMSC"
        else:
            _site_options = _site_df["Cancer type"].tolist()
            _selected_sites = st.multiselect(
                "Cancer site(s)",
                options=_site_options,
                default=[_site_options[0]] if _site_options else [],
                key="data_cancer_sites",
            )
            _headline_n = int(_site_df[_site_df["Cancer type"].isin(_selected_sites)]["New cases"].sum())
            _site_names = ", ".join(_selected_sites) if _selected_sites else "none selected"
            _headline_label = f"Cancer Incidence: {_site_names}"

        col1, col2 = st.columns(2)
        col1.metric(_headline_label, f"{_headline_n:,}")
        col2.metric(
            "% of population diagnosed annually",
            f"{100 * _headline_n / _total_pop:.2f}%" if _total_pop > 0 else "N/A",
        )

        st.markdown("**Cancer site breakdown**")
        _site_display = _site_df[["Cancer type", "New cases", "% of All Cancers"]].copy()
        _site_display["New cases"] = _site_display["New cases"].apply(lambda x: f"{x:,}")
        st.dataframe(_site_display, use_container_width=True, hide_index=True)
        st.caption(
            "Source: [GLOBOCAN 2022](https://gco.iarc.who.int/today/), International Agency for Research on Cancer (IARC). "
            "Cases are the estimated number of new cancer diagnoses in 2022. "
            "% of All Cancers uses the GLOBOCAN 'All cancers' aggregate as denominator."
        )

    st.divider()

    # ---- LINAC facilities --------------------------------------------------
    st.subheader(f"LINAC Facilities — {country}")
    _linac_result = _load_dirac(country)
    if _linac_result[0] is None:
        st.info(f"No LINAC data found for **{country}** in the DIRAC database.")
    else:
        _, _linac_df = _linac_result
        _linac_display = _linac_df.rename(columns={
            "name": "Facility name", "city": "City",
            "lat": "Latitude", "lon": "Longitude", "n_linacs": "LINACs",
        })
        st.dataframe(_linac_display, use_container_width=True, hide_index=True)
        st.caption(
            f"**{int(_linac_df['n_linacs'].sum())} LINACs** across **{len(_linac_df)} facilities**.  \n"
            "Source: [IAEA DIRAC Database](https://dirac.iaea.org/) — Directory of Radiotherapy Centres. "
            "Data downloaded 2025. "
            "Coordinates corrected via OpenStreetMap geocoding where missing or erroneous."
        )

    st.divider()

    # ---- Optimal RT utilisations (global table) ----------------------------
    st.subheader("Optimal Radiotherapy Utilisation Rates")
    with st.spinner("Loading…"):
        _opt_df = _data_tab_optimal_rt()

    # Show "All cancers" as a headline metric outside the table
    _all_cancers_row = _opt_df[_opt_df["Cancer type"] == "All cancers"]
    if not _all_cancers_row.empty:
        _all_cancers_pct = _all_cancers_row.iloc[0]["Optimal RT %"]
        st.metric("All Cancers — Optimal RT Utilisation", _all_cancers_pct)

    # Table: individual sites + NMSC + Other cancers; exclude aggregate rows
    _AGGREGATE_RT_KEYS = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
    _opt_table = _opt_df[
        ~_opt_df["Cancer type"].str.strip().str.lower().isin(_AGGREGATE_RT_KEYS)
    ][["Cancer type", "Optimal RT %"]].rename(
        columns={"Optimal RT %": "Optimal RT fraction"}
    )
    st.dataframe(_opt_table, use_container_width=True, hide_index=True)
    st.caption(
        "Source: Delaney G, Jacob S, Featherstone C, Barton M. "
        "*The role of radiotherapy in cancer treatment: estimating optimal utilization "
        "from a review of evidence-based clinical guidelines.* "
        "Cancer. 2005;104(6):1129–37. "
        "[doi:10.1002/cncr.21324](https://doi.org/10.1002/cncr.21324)"
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
        ["Exponential decay", "Weibull", "Step function", "Uniform (no decay)"],
        key="pm_model",
    )

    st.divider()

    if _pm_model == "Exponential decay":
        _pm_lambda = st.slider("Distance decay λ (km)", 5, 500, 30, step=5, key="pm_lambda")

        st.markdown("### Formula")
        st.latex(r"P(\text{treatment} \mid d) = e^{-d \,/\, \lambda}")
        st.markdown(
            "where $d$ is the distance to the nearest LINAC facility and "
            r"$\lambda$ is the distance-decay constant (half-length). "
            "When multiple facilities exist, contributions are combined as:"
        )
        st.latex(r"P_{\text{total}} = 1 - \prod_{i}\!\left(1 - e^{-d_i / \lambda}\right)^{w_i}")
        st.markdown(
            f"At the current setting ($\\lambda = {_pm_lambda}$ km), "
            f"a patient **{_pm_lambda} km** from a facility has a treatment probability of "
            f"**{np.exp(-1):.1%}** (i.e. $e^{{-1}} \\approx 37\\%$). "
            f"At **{2*_pm_lambda} km** the probability falls to "
            f"**{np.exp(-2):.1%}**."
        )

        _pm_dist = np.linspace(0, 1000, 500)
        _pm_prob = np.exp(-_pm_dist / _pm_lambda)

    elif _pm_model == "Weibull":
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

with tab_map:
    st.header(f"{map_type} — {country}")

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



        # ---- Population Density -----------------------------------------------
        if map_type == "Population Density":
            with st.spinner("Loading population data…"):
                gdf = _load_pop_region(country, h3_resolution) if _is_region else _load_pop(country, h3_resolution)

            pop = gdf["population"].to_numpy(dtype=np.float64)
            _areas_pop = _hex_areas_km2(gdf)
            if density_per_km2:
                plot_vals = pop / (_areas_pop / 10)
                pop_label = "People per 10 km²"
            else:
                plot_vals = pop
                pop_label = "People per hexagon"
            auto_vmin = float(max(plot_vals.min(), 1e-3))
            auto_vmax = float(plot_vals.max())
            colors, vmin, vmax = _color_values(plot_vals, cb_cmap_fn, auto_vmin, auto_vmax)

            gdf = gdf.copy()
            gdf["color"] = colors
            _s_area_pop = pd.Series(_areas_pop, index=gdf.index).apply(_fmt_sigfig)
            _s_pop_raw = gdf["population"].apply(_fmt_sigfig)
            _pop_tip = (
                "<b>" + gdf["h3"].astype(str) + "</b><br/>"
                + pop_label + ": " + pd.Series(plot_vals, index=gdf.index).round(2).astype(str) + "<br/>"
            )
            if density_per_km2:
                _pop_tip = _pop_tip + "<hr style='margin:3px 0'/>" + "People in hex: " + _s_pop_raw + "<br/>"
            _pop_tip = _pop_tip + "Hex area: " + _s_area_pop + " km²"
            gdf["tip"] = _pop_tip

            _geom_pop = gdf.geometry
            _lat_span_pop = float(_geom_pop.bounds["maxy"].max() - _geom_pop.bounds["miny"].min())
            _lon_span_pop = float(_geom_pop.bounds["maxx"].max() - _geom_pop.bounds["minx"].min())
            _lat_mid_pop = float((_geom_pop.bounds["maxy"].max() + _geom_pop.bounds["miny"].min()) / 2)
            country_span_km = max(_lat_span_pop * 111.32, _lon_span_pop * 111.32 * math.cos(math.radians(_lat_mid_pop)))

            df = pd.DataFrame({"h3": gdf["h3"], "color": gdf["color"], "tip": gdf["tip"]})
            _pop_layers = [_build_hex_layer(df)]
            _pop_pitch = 0.0
            if show_linac_markers and facilities_df is not None and not facilities_df.empty:
                _pop_layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style))
                _pop_pitch = 30.0
            _render_with_colorbar(
                _pop_layers,
                _make_view(gdf, pitch=_pop_pitch),
                cb_cmap_fn, vmin, vmax, pop_label, log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                show_linac_legend=show_linac_markers and facilities_df is not None and not facilities_df.empty,
            )
            st.caption(_h3_caption(gdf) + " · " + _scale_caption(gdf))
            col1, col2 = st.columns(2)
            col1.metric("Total population", f"{int(gdf['population'].sum()):,}")
            col2.metric("H3 hexagons", f"{len(gdf):,}")

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
                if map_type == "Radiotherapy Demand" and rt_method == "optimal" and _map_cancer_view != "Specific cancer site(s)":
                    if _map_cancer_view == "All cancers":
                        _load_cancers = _all_individual
                    else:  # All cancers excl. NMSC
                        _load_cancers = [c for c in _all_individual if c.lower() != "nmsc"]
                else:
                    _load_cancers = selected_cancers

                with st.spinner("Apportioning cancer incidence to H3 grid…"):
                    gdf = _load_cancer(country, iso3, tuple(_load_cancers), use_actual, h3_resolution, region_flag=_is_region)

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
                        label = f"Radiotherapy Demand{_per_suffix}"
                    else:
                        label = f"Cancer Incidence{_per_suffix}"

                    s_combined = pd.Series(plot_vals_c, index=gdf.index).round(2).astype(str)
                    _s_area_c = pd.Series(_areas_cancer, index=gdf.index).apply(_fmt_sigfig)
                    _s_pop_c = gdf["population"].apply(_fmt_sigfig)
                    gdf["tip"] = (
                        "<b>" + gdf["h3"].astype(str) + "</b><br/>"
                        + label + ": " + s_combined + "<br/>"
                        + "<hr style='margin:3px 0'/>"
                        + "People in hex: " + _s_pop_c + "<br/>"
                        + "Hex area: " + _s_area_c + " km²"
                    )

                    _geom_c = gdf.geometry
                    _lat_span_c = float(_geom_c.bounds["maxy"].max() - _geom_c.bounds["miny"].min())
                    _lon_span_c = float(_geom_c.bounds["maxx"].max() - _geom_c.bounds["minx"].min())
                    _lat_mid_c = float((_geom_c.bounds["maxy"].max() + _geom_c.bounds["miny"].min()) / 2)
                    country_span_km = max(_lat_span_c * 111.32, _lon_span_c * 111.32 * math.cos(math.radians(_lat_mid_c)))

                    df = pd.DataFrame({"h3": gdf["h3"], "color": gdf["color"], "tip": gdf["tip"]})
                    _cancer_layers = [_build_hex_layer(df)]
                    _cancer_pitch = 0.0
                    if show_linac_markers and facilities_df is not None and not facilities_df.empty:
                        _cancer_layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style))
                        _cancer_pitch = 30.0
                    _render_with_colorbar(
                        _cancer_layers,
                        _make_view(gdf, pitch=_cancer_pitch),
                        cb_cmap_fn, vmin, vmax, label, log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                        show_linac_legend=show_linac_markers and facilities_df is not None and not facilities_df.empty,
                    )
                    st.caption(_h3_caption(gdf) + " · " + _scale_caption(gdf))
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
                        if _map_cancer_view == "All cancers":
                            _scope_text = "Showing: all cancer sites (GLOBOCAN — All cancers)"
                        elif _map_cancer_view == "All cancers excl. NMSC":
                            _scope_text = "Showing: all cancer sites excl. non-melanoma skin cancer (GLOBOCAN — All cancers excl. NMSC)"
                        else:
                            _site_str = ", ".join(selected_cancers) if selected_cancers else "none selected"
                            _scope_text = f"Showing specific sites: {_site_str}"
                        st.caption(_scope_text)
                    if _map_cancer_view == "All cancers":
                        _cases_label = "Cancer Incidence: All Sites"
                    elif _map_cancer_view == "All cancers excl. NMSC":
                        _cases_label = "Cancer Incidence: All Cancers excl. NMSC"
                    else:
                        _site_str = ", ".join(selected_cancers) if selected_cancers else "none"
                        _cases_label = f"Cancer Incidence: {_site_str}"

                    if map_type == "Radiotherapy Demand":
                        incidence_cols = [c + "_incidence" for c in _load_cancers if (c + "_incidence") in gdf.columns]
                        total_incidence = float(gdf[incidence_cols].sum(axis=1).sum()) if incidence_cols else 0.0
                        col1, col2, col3 = st.columns(3)
                        col1.metric(_cases_label, f"{total_incidence:,.0f}")
                        col2.metric("Corresponding Cases Requiring RT", f"{combined.sum():,.0f}")
                        col3.metric("H3 hexagons", f"{len(gdf):,}")
                    else:
                        total_pop = float(gdf["population"].sum())
                        col1, col2, col3 = st.columns(3)
                        col1.metric(_cases_label, f"{combined.sum():,.0f}")
                        col2.metric("Country population", f"{int(total_pop):,}")
                        col3.metric("H3 hexagons", f"{len(gdf):,}")

        # ---- Radiotherapy Access / Nearest Linac ----------------------
        elif is_access or is_nearest:
            linac_locs_tuple = tuple(locs)

            if use_travel_time:
                if not tt_app_id or not tt_api_key:
                    st.warning(
                        "Enter your TravelTime App ID and API Key in the sidebar to use "
                        "driving or public transport times."
                    )
                    st.stop()

                # Build cache key from hex centroids + linac positions
                import hashlib as _hl, json as _json
                from data.population import load_population_at_resolution as _lpar
                _gdf_tmp = _load_pop_region(country, h3_resolution) if _is_region else _load_pop(country, h3_resolution)
                _hex_ll = [h3.cell_to_latlng(h) for h in _gdf_tmp["h3"]]
                _linac_ll = [(lat, lon) for lat, lon, _ in locs]
                _tt_payload = _json.dumps({"h": _hex_ll[:10], "l": _linac_ll, "n": len(_hex_ll)}, sort_keys=True)
                _tt_cache_key = _hl.md5(_tt_payload.encode()).hexdigest()[:16]
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
                            _hex_ll, _linac_ll, tt_mode,
                            tt_app_id, tt_api_key,
                            cache_key=_tt_cache_key,
                            progress_callback=_tt_cb,
                        )
                        _tt_progress.empty()
                        if _tt_errors:
                            st.warning(
                                f"{len(_tt_errors)} API batch(es) failed — affected hexes shown as >240 min. "
                                "Delete the cache and retry to attempt re-fetching.\n\n"
                                + "\n".join(f"• {e}" for e in _tt_errors[:5])
                                + ("" if len(_tt_errors) <= 5 else f"\n… and {len(_tt_errors) - 5} more")
                            )
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
                    )
            else:
                with st.spinner("Computing accessibility…"):
                    gdf_out, stats = _compute_access(
                        country, iso3, linac_locs_tuple,
                        float(lambda_km), access_model, float(max_distance_km),
                        capacity_per_machine_per_year, access_rt_method, access_rt_fraction,
                        h3_resolution, _is_region, snap_linacs_to_hex,
                        weibull_k=float(weibull_k),
                    )

            pitch = 30.0 if show_linac_markers else 0.0

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
                # Cap unreachable (inf) at 240 min for all stats/histograms
                _TT_MAX = 240.0
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
                if _binary_cmap:
                    _green = [34, 197, 94, 180]   # tailwind green-500
                    _red = [239, 68, 68, 180]      # tailwind red-500
                    # unreachable hexes are also red (they are by definition > threshold)
                    colors = [
                        (_green if (np.isfinite(v) and v <= _binary_threshold) else _red)
                        for v in dist_vals
                    ]
                    vmin, vmax = 0.0, _binary_threshold
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
                    + "People in hex: " + _s_pop_near + "<br/>"
                    + "Hex area: " + _s_area_near + " km²"
                )

                layers = [_build_hex_layer(
                    pd.DataFrame({"h3": gdf_out["h3"], "color": gdf_out["color"], "tip": gdf_out["tip"]})
                )]
                if show_linac_markers:
                    layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style))

                if _binary_cmap:
                    # Simple legend instead of colorbar
                    _map_col, _leg_col = st.columns([10, 1])
                    with _map_col:
                        st.pydeck_chart(pdk.Deck(
                            layers=layers,
                            initial_view_state=_make_view(gdf_out, pitch=pitch),
                            map_style="mapbox://styles/mapbox/dark-v9" if dark_mode else None,
                        ), use_container_width=True)
                    with _leg_col:
                        st.markdown(
                            f"<div style='margin-top:40px;font-size:0.75rem'>"
                            f"<div style='background:#22c55e;width:16px;height:16px;display:inline-block;border-radius:3px'></div> ≤ {_binary_threshold:.0f} {_near_tip_unit}<br/>"
                            f"<div style='background:#ef4444;width:16px;height:16px;display:inline-block;border-radius:3px;margin-top:6px'></div> > {_binary_threshold:.0f} {_near_tip_unit}"
                            + "</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    _render_with_colorbar(
                        layers, _make_view(gdf_out, pitch=pitch),
                        cb_cmap_fn, vmin, vmax, _near_label,
                        log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                        show_linac_legend=show_linac_markers,
                    )
                st.caption(_h3_caption(gdf_out) + " · " + _scale_caption(gdf_out))
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
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total LINACs", int(stats["total_machines"]))
                col2.metric(f"Median {_near_tip_label}", f"{_gt}{_median_val:.1f} {_near_tip_unit}")
                col3.metric(f"Pop-Weighted Median {_near_tip_label}", f"{_gt_pw}{_pw_median_val:.1f} {_near_tip_unit}")
                col4.metric("Average Geographic Access Probability", f"{_mean_geo_prob_nn:.1%}")
                if _has_tt:
                    st.caption(f"Travel time via TravelTime API · max {int(_TT_MAX)} min limit · {int((~valid).sum()):,} unreachable hexes shown at {int(_TT_MAX)} min")

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

                if access_display_metric == "Modelled Inaccessible":
                    display_vals = gdf_out["rt_untreated"].to_numpy(dtype=np.float64)
                    cb_label_access = "RT patients inaccessible/yr"
                    auto_vmin_a = 0.0
                    auto_vmax_a = float(np.nanmax(display_vals))
                    _tip_prefix = "RT patients inaccessible/yr"
                    _tip_extra = "RT patients accessing/yr"
                    _tip_extra_s = s_treated
                    metric_cmap_fn = _rdylgn_reversed_rgb

                elif access_display_metric == "Modelled Accessed":
                    display_vals = gdf_out["rt_treated"].to_numpy(dtype=np.float64)
                    cb_label_access = "RT patients accessing/yr"
                    auto_vmin_a = 0.0
                    auto_vmax_a = float(np.nanmax(display_vals))
                    _tip_prefix = "RT patients accessing/yr"
                    _tip_extra = "RT patients inaccessible/yr"
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
                        + "People in hex: " + s_pop_fmt + "<br/>"
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
                        + "People in hex: " + s_pop_fmt + "<br/>"
                        + "Hex area: " + _s_area_acc + " km²"
                    )
                    metric_cmap_fn = _rdylgn_rgb

                # Apply per-km² normalisation for count-based access metrics
                if access_display_metric in _count_access_metrics:
                    if density_per_km2:
                        display_vals = display_vals / (_areas_acc / 10)
                        auto_vmax_a = float(np.nanmax(display_vals))
                        cb_label_access += " per 10 km²"
                        s_vals = pd.Series(display_vals, index=gdf_out.index).round(3).astype(str)
                        tip_series = (
                            "<b>" + s_h3 + "</b><br/>"
                            + _tip_prefix + " per 10 km²: " + s_vals + "<br/>"
                            + _tip_extra + " per 10 km²: " + _tip_extra_s + "<br/>"
                            + "% treated: " + s_pct + "%<br/>"
                            + "<hr style='margin:3px 0'/>"
                            + "People in hex: " + s_pop_fmt + "<br/>"
                            + "Hex area: " + _s_area_acc + " km²"
                        )
                    else:
                        cb_label_access += " per hexagon"
                        s_vals = pd.Series(display_vals, index=gdf_out.index).round(1).astype(str)
                        tip_series = (
                            "<b>" + s_h3 + "</b><br/>"
                            + _tip_prefix + ": " + s_vals + "<br/>"
                            + _tip_extra + ": " + _tip_extra_s + "<br/>"
                            + "% treated: " + s_pct + "%<br/>"
                            + "<hr style='margin:3px 0'/>"
                            + "People in hex: " + s_pop_fmt + "<br/>"
                            + "Hex area: " + _s_area_acc + " km²"
                        )

                _map_default_cmap = _DEFAULT_CMAP.get(map_type, "Green → Red")
                active_cmap_fn = metric_cmap_fn if cb_cmap_name == _map_default_cmap else cb_cmap_fn
                colors, vmin, vmax = _color_values(display_vals, active_cmap_fn, auto_vmin_a, auto_vmax_a)

                gdf_out = gdf_out.copy()
                gdf_out["color"] = colors
                gdf_out["tip"] = tip_series.values

                layers = [_build_hex_layer(
                    pd.DataFrame({"h3": gdf_out["h3"], "color": gdf_out["color"], "tip": gdf_out["tip"]})
                )]
                if show_linac_markers:
                    layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style))

                if stats["total_rt_demand"] == 0:
                    st.warning(
                        f"RT demand is zero for **{country}** — this country may not be in the GLOBOCAN dataset. "
                        "Capacity allocation cannot be computed; geographic access probability is still valid."
                    )

                if access_display_metric in ("Modelled Access Ratio", "Geographic Access Probability"):
                    st.subheader(access_display_metric)

                _render_with_colorbar(
                    layers, _make_view(gdf_out, pitch=pitch),
                    active_cmap_fn, vmin, vmax, cb_label_access,
                    log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                    show_linac_legend=show_linac_markers,
                )
                st.caption(_h3_caption(gdf_out) + " · " + _scale_caption(gdf_out))
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

                # Row 1: machine/demand overview
                _globocan = stats.get("total_cancer_excl_nmsc")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("LINACs", int(stats["total_machines"]))
                col2.metric("Total LINAC Capacity", _fmt_sigfig(stats['total_national_capacity']))
                col3.metric("Cancer Incidence", _fmt_sigfig(_globocan) if _globocan is not None else "N/A")
                col4.metric("RT Demand", _fmt_sigfig(stats['total_rt_demand']))

                # Row 2: modelled (capacity + geography combined) access outcomes
                _pct_treated = (stats['total_rt_treated'] / stats['total_rt_demand'] * 100) if stats['total_rt_demand'] > 0 else 0.0
                col1b, col2b, col3b = st.columns(3)
                col1b.metric("Modelled RT Access", _fmt_sigfig(stats['total_rt_treated']))
                col2b.metric("Modelled RT Inaccessible", _fmt_sigfig(stats['total_rt_demand'] - stats['total_rt_treated']))
                col3b.metric("Modelled RT Access Ratio", f"{_pct_treated:.1f}%")

                # Row 3: single-constraint and combined ratios for direct comparison
                _geo_access = stats.get("mean_access_probability", 0.0)
                _cap_only = (stats['total_national_capacity'] / stats['total_rt_demand']) if stats['total_rt_demand'] > 0 else None
                _modelled_ratio = (stats['total_rt_treated'] / stats['total_rt_demand']) if stats['total_rt_demand'] > 0 else 0.0
                col1c, col2c, col3c, _ = st.columns([1, 1, 1, 1])
                col1c.metric("Modelled RT Access Ratio",
                             f"{_modelled_ratio:.1%}",
                             help="Modelled RT Access ÷ RT Demand — fraction of demand met accounting for both capacity and geography")
                col2c.metric("Capacity-Only Limited Access",
                             f"{_cap_only:.1%}" if _cap_only is not None else "N/A",
                             help="Total LINAC Capacity ÷ RT Demand — fraction of demand serviceable by machines alone, assuming no geographic barrier")
                col3c.metric("Geographic-Only Limited Access",
                             f"{_geo_access:.1%}",
                             help="Population-weighted mean geographic access probability, assuming unlimited machine capacity")
                _demand_info = (
                    f"demand: optimal RT utilisations"
                    if access_rt_method == "optimal"
                    else f"demand: proportional RT ({access_rt_fraction:.0%} of cancer cases)"
                )
                st.caption(
                    f"{model_info} | {int(capacity_per_machine_per_year)} patients/machine/yr | "
                    f"{_demand_info} | {stats['n_hexagons']:,} hexagons"
                )

                # ---- Geography Only Calculations ---------------------------
                st.divider()
                st.subheader(f"Geography Only Calculations — {country}")
                _use_tt_geo = use_travel_time and "nearest_linac_min" in gdf_out.columns
                _TT_MAX_ACC = 240.0
                if _use_tt_geo:
                    _near_col = "nearest_linac_min"
                    _dist_unit = "min"
                    _dist_label = "Travel Time to Linac"
                    _raw_geo_vals = gdf_out[_near_col].to_numpy(dtype=np.float64)
                    _geo_vals = np.where(np.isfinite(_raw_geo_vals), _raw_geo_vals, _TT_MAX_ACC)
                    _has_capped_geo = not np.all(np.isfinite(_raw_geo_vals))
                else:
                    _near_col = "nearest_linac_km"
                    _dist_unit = "km"
                    _dist_label = "Distance to Linac"
                    _geo_vals = gdf_out[_near_col].fillna(0).to_numpy(dtype=np.float64)
                    _has_capped_geo = False
                _geo_pop = gdf_out["population"].to_numpy(dtype=np.float64)
                _geo_median = float(np.median(_geo_vals))
                _gt_geo = "> " if _has_capped_geo and _geo_median >= _TT_MAX_ACC - 0.1 else ""
                # Pop-weighted median
                _geo_sort_idx = np.argsort(_geo_vals)
                _geo_cum_pop = np.cumsum(_geo_pop[_geo_sort_idx])
                _geo_pop_total = _geo_pop.sum()
                _geo_pw_med_idx = np.searchsorted(_geo_cum_pop, _geo_pop_total * 0.5)
                _geo_pw_median = float(_geo_vals[_geo_sort_idx[min(_geo_pw_med_idx, len(_geo_sort_idx) - 1)]])
                _gt_geo_pw = "> " if _has_capped_geo and _geo_pw_median >= _TT_MAX_ACC - 0.1 else ""
                _geo_col1, _geo_col2 = st.columns(2)
                _geo_col1.metric(f"Median {_dist_label}", f"{_gt_geo}{_geo_median:.1f} {_dist_unit}")
                _geo_col2.metric(f"Pop-Weighted Median {_dist_label}", f"{_gt_geo_pw}{_geo_pw_median:.1f} {_dist_unit}")
                if len(_geo_vals) > 0:
                    _geo_h_col1, _geo_h_col2 = st.columns(2)
                    with _geo_h_col1:
                        _fig_hist = go.Figure()
                        _fig_hist.add_trace(go.Histogram(
                            x=_geo_vals,
                            nbinsx=40,
                            marker_color="#4C9BE8",
                        ))
                        _fig_hist.update_layout(
                            xaxis_title=f"{_dist_label} ({_dist_unit})"
                                + (f" — bar at {int(_TT_MAX_ACC)} includes all travel time > {int(_TT_MAX_ACC)}" if _has_capped_geo else ""),
                            yaxis_title="Number of hexagons",
                            height=240,
                            margin=dict(l=40, r=20, t=30, b=40),
                            title_text="Hexagon count by distance",
                            showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(_fig_hist, use_container_width=True)
                    with _geo_h_col2:
                        _fig_hist2 = go.Figure()
                        _fig_hist2.add_trace(go.Histogram(
                            x=_geo_vals,
                            y=_geo_pop,
                            histfunc="sum",
                            nbinsx=40,
                            marker_color="#F97316",
                        ))
                        _fig_hist2.update_layout(
                            xaxis_title=f"{_dist_label} ({_dist_unit})"
                                + (f" — bar at {int(_TT_MAX_ACC)} includes all travel time > {int(_TT_MAX_ACC)}" if _has_capped_geo else ""),
                            yaxis_title="Population",
                            height=240,
                            margin=dict(l=40, r=20, t=30, b=40),
                            title_text="Population by distance",
                            showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(_fig_hist2, use_container_width=True)

                # ---- Capacity Only Calculations ----------------------------
                st.divider()
                st.subheader(f"Capacity Only Calculations — {country}")
                st.caption(
                    "NOTE: These calculations do not include access considerations based on "
                    "geographic access limitations, as modelled in the map above."
                )
                if not has_globocan_data(iso3):
                    st.warning(f"No GLOBOCAN data for **{country}** — RT need cannot be estimated.")
                else:
                    with st.spinner("Computing RT need…"):
                        _rt_need = _data_tab_rt_need(iso3)
                    _total_rt_cases = _rt_need["total_rt_cases"]
                    _n_linacs_need = int(facilities_df["n_linacs"].sum()) if facilities_df is not None and len(facilities_df) > 0 else 0
                    _capacity_per_linac = 450

                    st.markdown("**Calculation based on annual cancer incidence and optimal RT utilisation**")
                    _linacs_required_incidence = _total_rt_cases / _capacity_per_linac
                    _linacs_required_incidence_ceil = math.ceil(_linacs_required_incidence)
                    _linac_gap_incidence = _linacs_required_incidence_ceil - _n_linacs_need
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Cancers requiring RT annually", f"{int(_total_rt_cases):,}")
                    col2.metric("LINACs (DIRAC)", f"{_n_linacs_need:,}")
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
                    _linac_gap_prop = _linacs_required_prop_ceil - _n_linacs_need
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Cancers requiring RT annually", f"{int(_total_rt_cases_prop):,}")
                    col2.metric("LINACs (DIRAC)", f"{_n_linacs_need:,}")
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
                    _linacs_required_pop = _total_pop / 1_000_000 * 5
                    _linacs_required_pop_ceil = math.ceil(_linacs_required_pop)
                    _linac_gap_pop = _linacs_required_pop_ceil - _n_linacs_need
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Population", f"{_total_pop:,}")
                    col2.metric("LINACs (DIRAC)", f"{_n_linacs_need:,}")
                    col3.metric("LINACs required (5 per million pop.)", f"{_linacs_required_pop_ceil:,}")
                    _gap_pop_label = "LINAC shortage" if _linac_gap_pop > 0 else "LINAC surplus"
                    _gap_pop_color = "red" if _linac_gap_pop > 0 else "green"
                    col4.markdown(
                        f"<div><span style='display:block;font-size:0.875rem;color:#808495;margin-bottom:0.25rem'>{_gap_pop_label}</span>"
                        f"<span style='display:block;font-size:2rem;font-weight:600;line-height:1;color:{_gap_pop_color}'>{abs(_linac_gap_pop):,}</span></div>",
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

    st.subheader("About this tool")
    st.markdown(
        """
        This tool is open source and released under the
        [MIT License](https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access/blob/main/LICENSE).
        Source code is available on [GitHub](https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access).
        """
    )
    st.markdown(
        """
        This tool provides fast visualisation and analysis of access to radiotherapy (RT) at
        the sub-national scale, within countries and regions. It is designed to illuminate
        both the **problem landscape** — identifying which areas are underserved and whether
        this is driven by a shortage of machines or their geographic distribution — and the
        **solution landscape** — highlighting where new or relocated facilities would have
        the greatest impact.

        Approximately half of all cancer cases require radiotherapy, yet worldwide access
        to RT remains unacceptably low. This gap has been well characterised at the national,
        regional, and global levels
        ([Abdel-Wahab *et al.* 2025](https://doi.org/10.1016/S1470-2045(24)00678-8);
        [Burnet *et al.* 2025](https://doi.org/10.1016/j.radonc.2025.111061);
        [Atun *et al.* 2015](https://doi.org/10.1016/S1470-2045(15)00222-3)).

        Access to RT is constrained by two principal factors:

        - **Machine capacity** — the finite number of linear accelerators (linacs) within a
          country limits the total number of patients that can be treated each year.
        - **Geographic access** — RT requires attendance over several weeks; patients who
          live far from a facility are substantially less likely to complete a course of treatment.

        Previous work has addressed each of these factors independently. This tool is a
        first attempt to combine both constraints simultaneously, providing a unified view of
        where patients are most at risk of not receiving treatment.
        """
    )

    st.subheader("What each tab does")
    st.markdown(
        """
        | Tab | Contents |
        |---|---|
        | **🗺️ Map Modelling** | Interactive H3 hexagon maps — population density, cancer burden, RT demand, geographic access probability, and capacity-limited access. Select a country or region in the sidebar and click **Generate Map**. |
        | **📊 Data** | Country-level data tables — cancer incidence by site (GLOBOCAN 2022), LINAC locations (DIRAC), and optimal RT utilisation rates. |
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

        2. **Choose a map type** — start with *Population Density* to see the underlying
           data, then *Cancer Incidence* to see where cancer burden is concentrated,
           then *Radiotherapy Access* to see the combined model output.

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
           *Capacity Only Calculations* panel beneath the access map for a
           headline capacity gap estimate.
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
        st.image(_flowchart_path, use_column_width=True)

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
            "Assumption": "Distance rather than travel time",
            "Limitation": "Straight-line distance used as accessibility proxy. Travel time — accounting for roads, terrain, and transport — is the more meaningful barrier, especially in LMICs.",
            "To Improve": "Replace Euclidean distance with travel-time estimates (e.g. OpenRouteService or Google Maps APIs).",
        },
        {
            "Assumption": "Probability model for geographic access",
            "Limitation": "The distance–RT uptake relationship is poorly characterised. No single model is universally accepted (Perez et al. 2016; Lin et al. 2015; Yap et al. 2023).",
            "To Improve": "Incorporate empirically validated, country-specific probability models.",
        },
        {
            "Assumption": "Optimal RT utilisation rates",
            "Limitation": "Based on Delaney et al. (2005) evidence from Australia. May not reflect current practice or country-specific targets; does not account for hypofractionation.",
            "To Improve": "Allow country-specific utilisation targets and fractionation corrections.",
        },
        {
            "Assumption": "Greedy nearest-first allocation",
            "Limitation": "Assumes each facility serves its nearest patients first. Real referral patterns depend on clinical pathways, waiting times, and patient choice.",
            "To Improve": "Travel-time routing; incorporate referral pathway data where available.",
        },
        {
            "Assumption": "No patient stratification",
            "Limitation": "All cancer patients treated as equivalent. Access barriers differ by age, mobility, socioeconomic status, cancer stage, and RT modality required.",
            "To Improve": "Stratify demand by cancer type, stage, and demographic; incorporate access modifiers where data permit.",
        },
    ])

    _less_significant = _pd.DataFrame([
        {
            "Assumption": "Uniform linac capacity",
            "Limitation": "All linacs assumed to treat the same number of patients per year. Throughput varies with machine type, staffing, and operating hours.",
            "To Improve": "User can adjust global capacity; per-facility capacity could be incorporated if data are available.",
        },
        {
            "Assumption": "Full machine availability",
            "Limitation": "100% uptime assumed. Maintenance downtime and staffing shortages reduce effective capacity, particularly in LMICs.",
            "To Improve": "Apply a utilisation factor to effective machine capacity based on reported or estimated uptime.",
        },
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
            "Kontur H3 dataset. Each hexagon represents the number of people living within "
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
            st.image(fpath, use_column_width=True)
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
