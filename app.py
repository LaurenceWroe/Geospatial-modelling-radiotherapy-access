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

from data.population import (
    load_population_at_resolution,
    load_region_population,
    get_region_skipped_countries,
)
from data.linacs import load_linacs_from_dirac_db, load_linacs_for_region
from data.regions import is_region, get_region, REGIONS, REGION_GLOBOCAN_CODES
from data.cancer import (
    get_cancer_types, apportion_cancer_to_h3, has_globocan_data,
    get_national_cases, get_optimal_rt_fractions, DERIVED_CANCER_TYPES,
    XARRAY_PATH,
)
from analysis.accessibility import compute_accessibility, aggregate_access_gdf
from compute import (
    _country_options, _region_options, _region_country_summary, _load_pop, _load_pop_region, _load_cancer, _load_cancer_region_percountry, _compute_access, _compute_access_travel_time, _compute_access_one_country_for_region, _compute_access_region_percountry, _load_dirac, _data_tab_cancer, _data_tab_rt_need, _run_convergence_cached,
    _AGGREGATE_CANCER_KEYS,
)
from explanatory_tabs import (
    render_introduction, render_method, render_assumptions, render_toy_example,
)
from data.travel_time import compute_travel_time_matrix, CACHE_DIR as _TT_CACHE_DIR, MAX_TRAVEL_TIME_BY_RES as _TT_MAX_BY_RES, TT_SUPPORTED_RESOLUTIONS as _TT_SUPPORTED_RES

# Countries with no TravelTime driving-network coverage (confirmed via API error 16/17):
# China, Russia, North Korea, South Korea, Macao. Driving-time analysis is unavailable here.
_TT_UNSUPPORTED_ISO3 = {"CHN", "RUS", "PRK", "KOR", "MAC"}


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
    /* Separator tabs — non-interactive */
    [role="tablist"] > button:nth-child(2),
    [role="tablist"] > button:nth-child(5),
    [role="tablist"] > button:nth-child(10),
    [role="tablist"] > button:nth-child(12),
    [role="tablist"] > button:nth-child(17) {
        pointer-events: none !important;
        cursor: default !important;
        opacity: 0.35 !important;
    }
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
        [220, 50, 50],
        [230, 120, 30],
        [240, 200, 30],
        [80, 190, 60],
        [30, 120, 30],
    ]
    t = float(np.clip(t, 0, 1))
    n = len(stops) - 1
    i = min(int(t * n), n - 1)
    lo, hi = stops[i], stops[i + 1]
    f = t * n - i
    return [int(lo[j] + f * (hi[j] - lo[j])) for j in range(3)]


def _rdylgn_reversed_rgb(t: float) -> List[int]:
    return _rdylgn_rgb(1.0 - t)


# Cividis — perceptually uniform and colour-blind safe (blue → yellow)
_CIVIDIS = [
    [0, 32, 76],
    [0, 67, 128],
    [87, 117, 144],
    [166, 168, 130],
    [255, 233, 69],
]


def _cividis_rgb(t: float) -> List[int]:
    t = float(np.clip(t, 0, 1))
    n = len(_CIVIDIS) - 1
    i = min(int(t * n), n - 1)
    lo, hi = _CIVIDIS[i], _CIVIDIS[i + 1]
    f = t * n - i
    return [int(lo[j] + f * (hi[j] - lo[j])) for j in range(3)]


# Blue → Orange diverging: colour-blind safe substitute for Red→Green.
# Low (bad) = blue, high (good) = orange; both distinguishable in all common
# colour-vision deficiencies.
_BLUE_ORANGE = [
    [5, 48, 97],
    [67, 147, 195],
    [230, 230, 230],
    [244, 165, 130],
    [178, 24, 43],
]


def _blueorange_rgb(t: float) -> List[int]:
    t = float(np.clip(t, 0, 1))
    n = len(_BLUE_ORANGE) - 1
    i = min(int(t * n), n - 1)
    lo, hi = _BLUE_ORANGE[i], _BLUE_ORANGE[i + 1]
    f = t * n - i
    return [int(lo[j] + f * (hi[j] - lo[j])) for j in range(3)]


def _blueorange_reversed_rgb(t: float) -> List[int]:
    return _blueorange_rgb(1.0 - t)


# Named colormaps available to users
COLORMAPS = {
    "Purple → Yellow (Viridis)": _viridis_rgb,
    "Red → Green": _rdylgn_rgb,
    "Green → Red": _rdylgn_reversed_rgb,
    "Blue → Red (CB-safe)": _blueorange_rgb,
    "Red → Blue (CB-safe)": _blueorange_reversed_rgb,
    "Cividis (CB-safe)": _cividis_rgb,
}

# Colour-blind-safe substitutes applied when the user enables CB-safe mode.
_CB_SAFE_CMAP_SWAP = {
    "Red → Green": "Blue → Red (CB-safe)",
    "Green → Red": "Red → Blue (CB-safe)",
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

# Fixed colorbar figure width (inches) and its pixel width at 100 dpi. The PNG
# is displayed at exactly this pixel width via st.image, so its height is the
# figure's true pixel height and cannot be stretched by the column.
_CB_FIG_W_IN = 1.15
_CB_IMG_W_PX = int(_CB_FIG_W_IN * 100)


def _render_colorbar_fig(fig, width_px: int = _CB_IMG_W_PX) -> None:
    """Render a matplotlib colorbar figure at an exact pixel width via st.image."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True)
    plt.close(fig)
    buf.seek(0)
    st.image(buf, width=width_px)


def _colorbar_fig(
    cmap_fn,
    vmin: float,
    vmax: float,
    label: str,
    log_scale: bool = False,
    text_color: str = "black",
    clamp: bool = False,
    height_px: int = 400,
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
    # Figure height in inches maps 1:1 (at 100 dpi) to the map pixel height, so
    # the rendered colorbar always matches the map. Width is fixed at
    # _CB_FIG_W_IN so the PNG can be displayed at an exact pixel width (via
    # st.image) — this stops the column from stretching it taller than the map.
    _dpi = 100.0
    fig, ax = plt.subplots(figsize=(_CB_FIG_W_IN, max(1.5, height_px / _dpi)), dpi=_dpi)
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
# Map layer builders
# ---------------------------------------------------------------------------

_HEX_LAYER_ID = "hex-layer"


def _build_hex_layer(df: pd.DataFrame, opacity: float = 0.7) -> pdk.Layer:
    return pdk.Layer(
        "H3HexagonLayer",
        id=_HEX_LAYER_ID,
        data=df,
        get_hexagon="h3",
        get_fill_color="color",
        auto_highlight=True,
        pickable=True,
        opacity=opacity,
    )


# Natural Earth 50 m country boundaries (bundled).
_BORDERS_PATH = Path(__file__).resolve().parent / "assets" / "country_borders_50m.geojson"


@st.cache_data(show_spinner=False)
def _load_country_borders() -> Optional[dict]:
    """Load the bundled Natural Earth 50 m country boundaries (GeoJSON dict), or None."""
    import json
    if not _BORDERS_PATH.exists():
        return None
    try:
        with open(_BORDERS_PATH) as fh:
            return json.load(fh)
    except Exception:
        return None


def _build_borders_layer(dark: bool) -> Optional[pdk.Layer]:
    """Thin country outline layer for orientation at region/world zoom."""
    gj = _load_country_borders()
    if gj is None:
        return None
    line = [255, 255, 255, 140] if dark else [40, 40, 40, 160]
    return pdk.Layer(
        "GeoJsonLayer",
        id="country-borders",
        data=gj,
        stroked=True,
        filled=False,
        get_line_color=line,
        line_width_min_pixels=0.7,
        pickable=False,
    )


@st.cache_data(show_spinner=False)
def _no_data_fill_geojson():
    """FeatureCollection of countries with no GLOBOCAN cancer data.

    Used as a grey underlay on region/world maps: because it sits *beneath* the
    hex layer it only shows through where a country has no hexes — i.e. exactly
    the countries we can't compute demand/access for — while staying hidden on
    the population map (where those countries do have hexes).
    """
    gj = _load_country_borders()
    if gj is None:
        return None
    feats = []
    for f in gj.get("features", []):
        a2 = f.get("properties", {}).get("iso_a2")
        if not a2 or a2 == "-99":
            continue
        obj = pycountry.countries.get(alpha_2=a2)
        if obj is not None and has_globocan_data(obj.alpha_3):
            continue  # has data — leave it to the hex layers
        name = obj.name if obj is not None else a2
        nf = dict(f)
        nf["properties"] = {**f.get("properties", {}), "tip": f"{name}: no cancer data"}
        feats.append(nf)
    return {"type": "FeatureCollection", "features": feats}


def _build_no_data_fill_layer():
    """Grey fill layer for no-GLOBOCAN countries (None if there are none)."""
    gj = _no_data_fill_geojson()
    if gj is None or not gj["features"]:
        return None
    return pdk.Layer(
        "GeoJsonLayer",
        id="no-data-fill",
        data=gj,
        stroked=False,
        filled=True,
        get_fill_color=[158, 158, 158, 190],
        pickable=True,
    )


def _maybe_add_no_data_fill(layers: list) -> list:
    """Prepend the no-data grey fill (bottom layer) on region/world maps only."""
    if not globals().get("_is_region", False):
        return layers
    ndl = _build_no_data_fill_layer()
    if ndl is None:
        return layers
    return [ndl] + list(layers)


_LINAC_BLUE = [30, 120, 220, 220]

# Discrete colour scale: red → orange → yellow → green → dark green (up to 5 bands)
_DISCRETE_PALETTE_RDYLGN = [
    [220,  50,  50, 220],  # red
    [230, 120,  30, 220],  # orange
    [240, 200,  30, 220],  # yellow
    [ 60, 180,  60, 220],  # green
    [ 30, 120,  30, 220],  # dark green
]
# Colour-blind-safe discrete palette (dark blue → light blue → grey → orange → dark red)
_DISCRETE_PALETTE_CBSAFE = [
    [  5,  48,  97, 220],  # dark blue  (worst)
    [ 67, 147, 195, 220],  # light blue
    [210, 210, 210, 220],  # grey
    [244, 165,  60, 220],  # orange
    [178,  24,  43, 220],  # dark red   (best)
]
# Active discrete palette — reassigned from the sidebar CB-safe toggle.
_DISCRETE_PALETTE = _DISCRETE_PALETTE_RDYLGN
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
                    + (f"<br/>Capacity: {int(row_data['capacity']):,} pts/yr" if "capacity" in row_data and pd.notna(row_data['capacity']) else "")
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
                    + (f"<br/>Capacity: {int(row_data['capacity']):,} pts/yr" if "capacity" in row_data and pd.notna(row_data['capacity']) else "")
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
    """Format a number to 1 decimal place with k/M/B suffix (e.g. 11.0 M, not 10994.6 k)."""
    if value is None:
        return "N/A"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} B"
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


def _mercator_y(lat_deg: float) -> float:
    """Web-Mercator y as a fraction of the world height for a latitude (deg)."""
    lat = max(min(lat_deg, 85.05), -85.05)
    s = math.sin(math.radians(lat))
    return 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)


def _fit_zoom(lon_span: float, lat_min: float, lat_max: float,
              map_w_px: float, map_h_px: float, pad: float = 0.12) -> float:
    """Web-Mercator zoom that fits a bounding box into a viewport.

    Returns the zoom at which the extreme hexes fit within the binding
    dimension (``min`` of the two = whichever fills first). ``pad`` (0–1) leaves
    a fractional margin so the country isn't flush to (or clipped by) the edge —
    the true browser viewport aspect is unknown server-side, so a small margin
    guards against slightly over-zooming.
    """
    _WORLD = 512.0
    lon_frac = max(lon_span, 1e-6) / 360.0
    lat_frac = max(abs(_mercator_y(lat_max) - _mercator_y(lat_min)), 1e-6)
    z_lon = math.log2(max(map_w_px, 1) * (1 - pad) / (_WORLD * lon_frac))
    z_lat = math.log2(max(map_h_px, 1) * (1 - pad) / (_WORLD * lat_frac))
    return min(z_lon, z_lat)


def _make_view(gdf, pitch: float = 0.0) -> pdk.ViewState:
    geom = gdf.geometry
    b = geom.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    lon_span = maxx - minx
    lat_span = maxy - miny

    map_h = float(globals().get("_MAP_HEIGHT", 560))
    # The map column width is unknown server-side (it scales with the browser).
    # Assume a moderately-landscape viewport; erring narrow means we zoom out a
    # touch rather than clip the east/west hexes.
    map_w = map_h * 1.5

    # Centre on the geometric middle of the hex bounding box so every hex is
    # equidistant from centre and the fit shows the whole extent — north to
    # south and east to west. (A population-weighted centre pulls the frame
    # toward dense regions and clips sparse edges, e.g. northern Scotland.)
    cx = (minx + maxx) * 0.5
    cy = (miny + maxy) * 0.5

    # Fit the full hex extent — min(z_lon, z_lat) picks whichever dimension
    # binds, so the most east/west OR north/south hexes just fit inside the
    # frame. Applies to the world too (previously a hard-coded zoom that clipped).
    fit = _fit_zoom(lon_span, miny, maxy, map_w, map_h)
    zoom = max(0.2, min(12.0, fit))
    # No min_zoom lock — the user can zoom in/out freely from this framing.
    return pdk.ViewState(
        latitude=cy, longitude=cx, zoom=zoom,
        max_zoom=13, pitch=pitch, bearing=0,
    )


_CARTO_LIGHT_LABELS   = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
_CARTO_LIGHT_NOLABELS = "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json"
_CARTO_DARK_LABELS    = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
_CARTO_DARK_NOLABELS  = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json"
# Defaults — overwritten after sidebar renders with show_map_labels value
CARTO_LIGHT = _CARTO_LIGHT_NOLABELS
CARTO_DARK  = _CARTO_DARK_NOLABELS


def _render_discrete_legend(bounds: list, palette: list, unit: str, text_color: str = "black", title: str = "") -> None:
    """Render a discrete colour legend in the current column."""
    n = len(palette)
    def _hex_color(rgba):
        return "#{:02x}{:02x}{:02x}".format(rgba[0], rgba[1], rgba[2])
    # Fill the map height (less ~30px for the copyright line) so bands align with the map.
    _legend_h = globals().get("_MAP_HEIGHT", 460) - 30
    band_h = max(28, int(_legend_h / n))
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
            f"<span style='font-size:16px;color:{text_color};margin-left:4px;line-height:1.2'>{label}</span></div>"
        )
    _cr_color = "#aaa" if text_color == "white" else "#888"
    st.markdown(
        # f"<p style='font-size:13px;font-family:monospace;color:{_cr_color};margin:0;text-align:center;'>© RadMaps 2026</p>",
        f"<p style='font-size:12px;font-family:monospace;color:{_cr_color};margin:-7px;text-align:center;'>© RadMaps 2026</p>",
        unsafe_allow_html=True,
    )
    title_html = (
        f"<div style='writing-mode:vertical-rl;transform:rotate(180deg);font-size:20px;"
        f"color:{text_color};white-space:nowrap;align-self:center;"
        f"margin-right:-4px'>{title}</div>"
        if title else ""
    )
    st.markdown(
        f"<div style='display:flex;flex-direction:row;align-items:stretch;margin-top:8px'>"
        f"{title_html}"
        f"<div>{''.join(items)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _maybe_add_borders(layers: list, dark: bool) -> list:
    """Append the country-borders layer when the sidebar toggle is on."""
    if not globals().get("show_borders", False):
        return layers
    bl = _build_borders_layer(dark)
    if bl is None:
        return layers
    return list(layers) + [bl]


def _view_key(base):
    """Build a stable pydeck widget key from a base (or None for Streamlit auto-key).

    A key that stays constant across reruns lets a click-mode map keep its
    selection state and view; the base already encodes country + resolution so
    changing either remounts the chart and recenters.
    """
    if base is None:
        return None
    return str(base)


def _render_map_no_cb(layers, view: pdk.ViewState, dark: bool, on_select=None, map_key=None):
    """Render a pydeck map with an empty right column (matching _render_with_colorbar layout)."""
    if not isinstance(layers, list):
        layers = [layers]
    layers = _maybe_add_no_data_fill(layers)
    layers = _maybe_add_borders(layers, dark)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=CARTO_DARK if dark else CARTO_LIGHT,
        tooltip={"html": "{tip}"},
    )
    _mh = globals().get("_MAP_HEIGHT", 500)
    _k = _view_key(map_key)
    col_map, col_cb = st.columns([7, 1])
    chart_state = None
    with col_map:
        if on_select and _PYDECK_CLICK_SUPPORTED:
            chart_state = st.pydeck_chart(deck, use_container_width=True, height=_mh, key=_k,
                                          on_select=on_select, selection_mode="single-object")
        else:
            st.pydeck_chart(deck, use_container_width=True, height=_mh, key=_k)
    with col_cb:
        _cr_color_nocb = "#aaa" if dark else "#888"
        st.markdown(
            f"<p style='font-size:12px;font-family:monospace;color:{_cr_color_nocb};margin:-7px;text-align:center;'>© RadMaps 2026</p>",
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
    map_key=None,
):
    if not isinstance(layers, list):
        layers = [layers]
    layers = _maybe_add_no_data_fill(layers)
    layers = _maybe_add_borders(layers, dark)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=CARTO_DARK if dark else CARTO_LIGHT,
        tooltip={"html": "{tip}"},
    )
    _mh = globals().get("_MAP_HEIGHT", 500)
    _k = _view_key(map_key)
    col_map, col_cb = st.columns([7, 1])
    chart_state = None
    with col_map:
        if on_select and _PYDECK_CLICK_SUPPORTED:
            chart_state = st.pydeck_chart(deck, use_container_width=True, height=_mh, key=_k,
                                          on_select=on_select, selection_mode="single-object")
        else:
            st.pydeck_chart(deck, use_container_width=True, height=_mh, key=_k)
    with col_cb:
        _cr_color = "#aaa" if (dark or dark_text) else "#888"
        st.markdown(
            f"<p style='font-size:12px;font-family:monospace;color:{_cr_color};margin:-7px;text-align:center;'>© RadMaps 2026</p>",
            unsafe_allow_html=True,
        )
        # height_px trimmed by the caption above so the bar bottom lines up with
        # the map bottom.
        fig = _colorbar_fig(cmap_fn, vmin, vmax, cb_label, log_scale=log_scale,
                            text_color="white" if (dark or dark_text) else "black",
                            clamp=clamp, height_px=_mh - 18)
        _render_colorbar_fig(fig)
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


@st.cache_data(show_spinner=False, max_entries=4)
def _gdf_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a result table (geometry dropped) to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def _download_results_button(gdf, filename: str, key: str) -> None:
    """Offer the computed per-hex result table as a CSV download."""
    try:
        _df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    except Exception:
        _df = pd.DataFrame(gdf)
    # Drop bulky/internal helper columns
    _df = _df[[c for c in _df.columns if not c.startswith(("centroid_", "color", "tip", "_"))]]
    st.download_button(
        "⬇️ Download hex data (CSV)",
        data=_gdf_to_csv_bytes(_df),
        file_name=filename,
        mime="text/csv",
        key=key,
        help="Export the per-hexagon results table for your own analysis.",
    )


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
            "linac_locs_tuple": None,
            "facilities_df": None,
        })
        st.session_state.setdefault("_map_generated", True)


# Dark modes — read from session state so they're available before sidebar and tabs render
app_dark_mode: bool = st.session_state.get("_app_dark_mode", False)
dark_mode: bool = st.session_state.get("_map_dark_mode", False)
_apply_app_dark_mode(app_dark_mode)

with st.sidebar:
    st.title("🏥 RadMaps")
    st.subheader("Country / Region Selection")

    # ── Location ──────────────────────────────────────────────────────────
    _location_type = st.radio(
        "Analyse by", ["Country", "Region"],
        index=1,  # default to Region (World)
        horizontal=True, key="_loc_type_radio", label_visibility="collapsed",
    )
    if _location_type == "Region":
        _reg_opts = _region_options()
        _default_reg_idx = _reg_opts.index("World") if "World" in _reg_opts else 0
        country = st.selectbox("Region", options=_reg_opts, index=_default_reg_idx)
        _is_region = True
    else:
        _ctry_opts = _country_options()
        _default_ctry = "United Kingdom"
        _default_ctry_idx = _ctry_opts.index(_default_ctry) if _default_ctry in _ctry_opts else 0
        country = st.selectbox("Country", options=_ctry_opts, index=_default_ctry_idx)
        _is_region = False

    # Regions always use per-country mode (individual country cancer data + LINACs)
    _region_percountry = _is_region

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
            "H3 resolution", options=[7, 6, 5, 4, 3], index=3,
            format_func=lambda r: {
                7: "H7 (~5 km²)", 6: "H6 (~36 km²)",
                5: "H5 (~253 km²)", 4: "H4 (~1,770 km²)", 3: "H3 (~12,400 km²)",
            }[r],
            key="h3_res_country",
        )

    st.divider()

    # ── Calculation Settings ──────────────────────────────────────────────
    st.subheader("Calculation Settings")
    map_type = st.selectbox(
        "Calculation Type", MAP_TYPES, index=MAP_TYPES.index("Radiotherapy Access"), key="_map_type_select",
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

    access_display_metric: str = "Modelled Access Ratio"
    if is_access:
        access_display_metric = st.selectbox(
            "Display metric",
            ["Modelled Access Deficit", "Modelled Accessed", "Modelled Access Ratio", "Geographic Access Probability", "RT Demand"],
            index=2,  # default: Modelled Access Ratio
        )

    # ── Radiotherapy Utilisation Rate (part of Calculation Settings) ──────
    rt_method: str = "optimal"
    rt_fraction: float = 0.25

    _rt_label = st.radio(
        "Radiotherapy Utilisation Rate (RTU)",
        ["Optimal", "Custom", "Proportional"],
        horizontal=False, key="rt_demand_method_radio",
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

    # ── Access calculation parameters (part of Calculation Settings) ──────
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

    capacity_per_machine_per_year = float(st.slider(
        "Capacity per LINAC (patients/yr)", min_value=50, max_value=1000, value=450, step=50,
        help="Individual facility capacities can be modified in the Data tab.",
    ))

    tt_method = st.radio(
        "Geographic Decay Metric",
        ["Straight-line distance", "Driving time", "Public transport time"],
        index=0, horizontal=True,
    )
    use_travel_time = tt_method != "Straight-line distance"
    tt_mode = "driving" if tt_method == "Driving time" else "public_transport"

    if use_travel_time:
        if h3_resolution not in _TT_SUPPORTED_RES:
            st.info(
                f"Resolution {h3_resolution} is not supported by the TravelTime API directly. "
                "Travel times will be aggregated from a resolution 5 matrix using population-weighted averaging. "
                "Compute a resolution 5 matrix first if one does not yet exist."
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
        "Access Decay model",
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

    st.divider()

    # Read click-mode toggle from session state before the map renders.
    # The toggle widget itself is shown later (inside the Add Additional LINACs section),
    # but its value must be known when building the map layers with on_select.
    click_mode: bool = bool(st.session_state.get("click_mode_toggle", False)) if is_access else False

    # ── Map Colour Settings ───────────────────────────────────────────────
    st.subheader("Map Colour Settings")

    cb_safe_mode = st.checkbox(
        "Colour-blind safe", value=False,
        help="Swap red/green ramps for colour-blind-safe blue/orange equivalents "
             "(affects continuous and discrete scales).",
    )
    # Swap the active discrete palette to match the toggle
    _DISCRETE_PALETTE = _DISCRETE_PALETTE_CBSAFE if cb_safe_mode else _DISCRETE_PALETTE_RDYLGN

    _default_cmap_name = _DEFAULT_CMAP.get(map_type, "Purple → Yellow (Viridis)")
    if cb_safe_mode:
        _default_cmap_name = _CB_SAFE_CMAP_SWAP.get(_default_cmap_name, _default_cmap_name)
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
    _scale_type = st.selectbox("Scale", _scale_opts, index=_scale_default_idx, key="scale_type_select")
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
        elif map_type == "Radiotherapy Demand":
            _disc_default, _disc_step = 100.0, 10.0
        elif map_type == "Cancer Incidence":
            _disc_default, _disc_step = 200.0, 20.0
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

    _count_maps = {"Population Density", "Cancer Incidence", "Radiotherapy Demand"}
    _count_access_metrics = {"Modelled Accessed", "Modelled Access Deficit"}
    _supports_per_km2 = map_type in _count_maps or (is_access and access_display_metric in (_count_access_metrics | {"RT Demand"}))
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

    st.divider()

    # ── Map View Settings ─────────────────────────────────────────────────
    st.subheader("Map View Settings")

    show_map_labels = st.checkbox("Show place names", value=False)
    show_borders = st.checkbox(
        "Show country borders in black", value=_is_region,
        help="Overlay national boundaries — useful for orientation at region and world zoom.",
    )
    _MAP_HEIGHT = int(st.slider(
        "Map height (px)", min_value=400, max_value=1000, value=560, step=20,
        help="Height of the map canvas. The colourbar and legend scale to match — "
             "raise it on large screens.",
    ))

    hex_opacity = st.slider(
        "Hexagon transparency", min_value=0, max_value=100, value=0,
        format="%d%%",
        help="Transparency of the hex fill layer (0% = fully opaque, 100% = fully transparent).",
    )
    _hex_opacity_f = (100 - hex_opacity) / 100.0

    if st.button("Update Map", use_container_width=True,
                 help="Re-render with current display settings — no recomputation."):
        st.session_state["_map_generated"] = True

    _dm_c1, _dm_c2 = st.columns(2)
    with _dm_c1:
        if st.button(
            "☀️ App" if app_dark_mode else "🌙 App",
            key="_sb_app_dm", use_container_width=True,
            help="Toggle dark/light app background",
        ):
            st.session_state["_app_dark_mode"] = not app_dark_mode
            st.rerun()
    with _dm_c2:
        if st.button(
            "🔆 Map" if dark_mode else "🌑 Map",
            key="_sb_map_dm", use_container_width=True,
            help="Toggle dark/light map tiles",
        ):
            st.session_state["_map_dark_mode"] = not dark_mode
            st.rerun()

    generate = st.session_state.get("_map_generated", False)

    st.divider()

    show_linac_markers: bool = False
    tower_height_scale: float = 1.0
    tower_radius_scale: float = 1.0
    linac_tower_style: str = "stacked"
    linac_multi_color: bool = False

    show_linac_markers = st.checkbox("Show Facility Locations", value=False)
    map_pitch_on: bool = False
    if show_linac_markers:
        tower_height_scale = float(st.slider("Tower height scale", 0.05, 5.0, 1.0, step=0.05))
        tower_radius_scale = float(st.slider("Tower radius scale", 0.1, 5.0, 1.0, step=0.1))
        map_pitch_on = st.toggle("3D view (map pitch)", value=True)
        _tower_style_label = st.radio(
            "Tower style",
            ["Individual (tower per centre)", "Stacked (tower per hex)"],
            index=1,
            horizontal=False,
        )
        linac_tower_style = "individual" if "Individual" in _tower_style_label else "stacked"
        linac_multi_color = st.checkbox("Multiple colours", value=False,
                                        help="Assign a distinct colour to each facility; otherwise all shown in blue.")

CARTO_LIGHT = _CARTO_LIGHT_LABELS if show_map_labels else _CARTO_LIGHT_NOLABELS
CARTO_DARK  = _CARTO_DARK_LABELS  if show_map_labels else _CARTO_DARK_NOLABELS


# ---------------------------------------------------------------------------
# Helpers used in map sections
# ---------------------------------------------------------------------------

def _color_values(values: np.ndarray, cmap_fn, auto_vmin: float, auto_vmax: float,
                  invert_binary: bool = False, disc_base: float | None = None):
    """Apply colormap using user or auto range, optionally in log/binary space.

    invert_binary: when True, above-threshold = red (for metrics where high = bad).
    disc_base: override the sidebar _discrete_base for auto-scaled discrete maps.
    """
    vmin = cb_vmin_user if not cb_auto and cb_vmin_user is not None else auto_vmin
    vmax = cb_vmax_user if not cb_auto and cb_vmax_user is not None else auto_vmax
    if _discrete_scale:
        _bounds = [n * (disc_base if disc_base is not None else _discrete_base) for n in range(1, _discrete_steps)]
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

tab_intro, _tab_sep0, tab_map, tab_data, _tab_sep1, tab_choropleth, tab_country, tab_cap, tab_geo, _tab_sep2, tab_plan, _tab_sep3, tab_method, tab_assumptions, tab_toy, tab_model, _tab_sep4, tab_sensitivity, tab_convergence = st.tabs([
    "💡 Introduction", "│", "🗺️ Access Maps", "📊 Data", "│", "🌐 World Choropleth", "📈 Country Analysis", "⚡ Capacity-Only", "🌍 Geography-Only", "│", "🔧 Machine Planning", "│", "📖 Method", "⚠️ Assumptions", "🧪 Toy Example", "📐 Probability Models", "│", "📉 Sensitivity", "🔬 Convergence",
], default="🗺️ Access Maps")

# ---------------------------------------------------------------------------
# Data tab — always available, no Generate button required
# ---------------------------------------------------------------------------

with tab_data:
    st.header(f"Data — {country}")

    # ---- Per-country summary (shown when per-country regional mode is active) ----
    if _is_region and _region_percountry:
        st.subheader(f"Per-Country Summary — {country}")
        st.caption(
            "Cancer incidence from GLOBOCAN; LINAC counts from DIRAC. "
            "Sorted by cancer incidence (descending)."
        )
        with st.spinner("Loading per-country summary…"):
            _pc_summary_df = _region_country_summary(country)
        if _pc_summary_df.empty:
            st.info("No per-country data available for this region.")
        else:
            st.dataframe(
                _pc_summary_df.style.format({
                    "Cancer incidence (excl. NMSC)": "{:,}",
                    "LINACs (DIRAC)": "{:,}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        st.divider()

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
    # One stable key per country+resolution; the reset nonce is folded in by
    # _view_key. Changing country/resolution remounts the chart so it recenters;
    # unrelated reruns keep the user's current pan/zoom.
    _main_map_key = f"mainmap_{iso3}_{h3_resolution}"
    _map_header = f"Population Data › {map_type} — {country}" if _is_pop_data else f"{map_type} — {country}"
    st.header(_map_header)

    # Download button is rendered at the very bottom of this tab; branches set this.
    _dl_info = None

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
            _pop_layers = [] if _no_hex else [_build_hex_layer(df, _hex_opacity_f)]
            if show_linac_markers and facilities_df is not None and not facilities_df.empty:
                _pop_layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style, color=None if linac_multi_color else _LINAC_BLUE))
            if _no_hex:
                _render_map_no_cb(_pop_layers, _make_view(gdf, pitch=_pop_pitch), dark_mode, map_key=_main_map_key)
            else:
                _render_with_colorbar(
                    _pop_layers,
                    _make_view(gdf, pitch=_pop_pitch),
                    cb_cmap_fn, vmin, vmax, pop_label, log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                    show_linac_legend=show_linac_markers and facilities_df is not None and not facilities_df.empty,
                    map_key=_main_map_key,
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
                    colors, vmin, vmax = _color_values(plot_vals_c, cb_cmap_fn, auto_vmin, auto_vmax,
                                                       invert_binary=(map_type == "Radiotherapy Demand"))

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
                    _cancer_layers = [] if _no_hex else [_build_hex_layer(df, _hex_opacity_f)]
                    if show_linac_markers and facilities_df is not None and not facilities_df.empty:
                        _cancer_layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style, color=None if linac_multi_color else _LINAC_BLUE))
                    _cancer_invert_binary = map_type == "Radiotherapy Demand"
                    if _no_hex:
                        _render_map_no_cb(_cancer_layers, _make_view(gdf, pitch=_cancer_pitch), dark_mode, map_key=_main_map_key)
                    elif _discrete_scale:
                        _canc_map_col, _canc_leg_col = st.columns([7, 1])
                        with _canc_map_col:
                            st.pydeck_chart(pdk.Deck(
                                layers=_maybe_add_borders(_cancer_layers, dark_mode),
                                initial_view_state=_make_view(gdf, pitch=_cancer_pitch),
                                map_style=CARTO_DARK if dark_mode else CARTO_LIGHT,
                                tooltip={"html": "{tip}"},
                            ), use_container_width=True, height=_MAP_HEIGHT, key=_view_key(_main_map_key))
                        with _canc_leg_col:
                            _canc_disc_palette = list(reversed(_DISCRETE_PALETTE[:_discrete_steps])) if _cancer_invert_binary else _DISCRETE_PALETTE[:_discrete_steps]
                            _render_discrete_legend(
                                [n * _discrete_base for n in range(1, _discrete_steps)],
                                _canc_disc_palette,
                                "", "white" if (dark_mode or app_dark_mode) else "black",
                                title=label,
                            )
                    else:
                        _render_with_colorbar(
                            _cancer_layers,
                            _make_view(gdf, pitch=_cancer_pitch),
                            cb_cmap_fn, vmin, vmax, label, log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                            show_linac_legend=show_linac_markers and facilities_df is not None and not facilities_df.empty,
                            map_key=_main_map_key,
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
                        "Travel time mode is not supported for regional analysis. "
                        "Select a single country to use driving or public transport times."
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
                        if h3_resolution not in _TT_SUPPORTED_RES:
                            # Aggregate from resolution 5 — try matching time budget first, then max (10h)
                            _res5_candidates = [
                                f"{iso3}_res5_{_tt_max_h}h_{_linac_hash}",
                                f"{iso3}_res5_10h_{_linac_hash}",
                            ]
                            _res5_cache_key = next(
                                (k for k in _res5_candidates
                                 if (_TT_CACHE_DIR / f"{k}_{tt_mode}.npz").exists()),
                                None,
                            )
                            if _res5_cache_key is None:
                                st.error(
                                    f"No resolution 5 travel time matrix found for {iso3} with these LINACs. "
                                    "Switch to resolution 5, fetch the matrix, then return to resolution "
                                    f"{h3_resolution}."
                                )
                                st.stop()
                            with st.spinner(f"Aggregating resolution 5 → {h3_resolution} travel times…"):
                                from data.travel_time import aggregate_tt_matrix as _aggregate_tt_matrix
                                _mat_res5 = np.load(_TT_CACHE_DIR / f"{_res5_cache_key}_{tt_mode}.npz")["matrix"]
                                _gdf_res5 = _load_pop_region(country, 5) if _is_region else _load_pop(country, 5)
                                _mat_agg = _aggregate_tt_matrix(
                                    _mat_res5,
                                    list(_gdf_res5["h3"]),
                                    _gdf_res5["population"].to_numpy(dtype=np.float64),
                                    _hex_ids,
                                    target_resolution=h3_resolution,
                                    unreachable_fill_min=tt_max_travel_time_sec / 60.0,
                                )
                            _TT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(_tt_cache_file, matrix=_mat_agg)
                        else:
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
                st.session_state["_map_result"] = {
                    "gdf_out": gdf_out, "stats": stats, "region_percountry": _region_percountry,
                    "linac_locs_tuple": linac_locs_tuple, "facilities_df": facilities_df,
                }

                # Persist world default to disk so future sessions load instantly
                _wp = _WORLD_DEFAULT_PARAMS
                if (country == _wp["country"] and access_model == _wp["model"]
                        and abs(float(max_distance_km) - _wp["max_distance_km"]) < 1
                        and h3_resolution == _wp["h3_res"]
                        and access_rt_method == _wp["rt_method"]
                        and abs(capacity_per_machine_per_year - _wp["capacity"]) < 1
                        and _region_percountry == _wp["region_percountry"]):
                    _save_world_default(gdf_out, stats)

            # --- Data quality warnings (shown on every rerun) ---
            if stats.get("demand_fallback"):
                st.warning(
                    "⚠️ Cancer incidence data could not be loaded for this selection — "
                    "the map is using **raw population** as a proxy for RT demand. "
                    "Counts shown are people, not patients."
                )
            _skipped = stats.get("skipped_countries") or []
            if _skipped:
                st.warning(
                    f"⚠️ {len(_skipped)} countr{'y was' if len(_skipped) == 1 else 'ies were'} "
                    "excluded (no population or GLOBOCAN data): "
                    + ", ".join(sorted(_skipped))
                    + ". Regional totals understate true demand."
                )
            if _is_region:
                _pop_skipped = get_region_skipped_countries(country, h3_resolution)
                if _pop_skipped:
                    st.warning(
                        "⚠️ The cached population map for this region was built without: "
                        + ", ".join(_pop_skipped)
                        + ". Delete the region cache file in `H3_region_cache/` to rebuild."
                    )

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
                layers = [] if _no_hex else [_build_hex_layer(_near_df, _hex_opacity_f)]
                if show_linac_markers:
                    layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, radius_scale=tower_radius_scale, style=linac_tower_style, color=None if linac_multi_color else _LINAC_BLUE))

                if click_mode and _PYDECK_CLICK_SUPPORTED:
                    st.info("Click a hexagon on the map to place a LINAC there, then click **Generate Map** to recompute.")

                _near_chart_state = None
                if _no_hex:
                    _near_chart_state = _render_map_no_cb(layers, _make_view(gdf_out, pitch=pitch), dark_mode,
                                                          on_select="rerun" if click_mode else None, map_key=_main_map_key)
                elif _discrete_scale:
                    _map_col, _leg_col = st.columns([7, 1])
                    with _map_col:
                        _bin_deck = pdk.Deck(layers=_maybe_add_borders(layers, dark_mode), initial_view_state=_make_view(gdf_out, pitch=pitch),
                                             map_style=CARTO_DARK if dark_mode else CARTO_LIGHT, tooltip={"html": "{tip}"})
                        if click_mode and _PYDECK_CLICK_SUPPORTED:
                            _near_chart_state = st.pydeck_chart(_bin_deck, use_container_width=True, height=_MAP_HEIGHT, key=_view_key(_main_map_key),
                                                                on_select="rerun", selection_mode="single-object")
                        else:
                            st.pydeck_chart(_bin_deck, use_container_width=True, height=_MAP_HEIGHT, key=_view_key(_main_map_key))
                    with _leg_col:
                        _disc_bounds_near = [n * _discrete_base for n in range(1, _discrete_steps)]
                        _render_discrete_legend(_disc_bounds_near, _DISCRETE_PALETTE[:_discrete_steps], _near_tip_unit,
                                                "white" if (dark_mode or app_dark_mode) else "black",
                                                title=_near_label)
                else:
                    _near_chart_state = _render_with_colorbar(
                        layers, _make_view(gdf_out, pitch=pitch),
                        cb_cmap_fn, vmin, vmax, _near_label,
                        log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                        show_linac_legend=show_linac_markers,
                        on_select="rerun" if click_mode else None,
                        map_key=_main_map_key,
                    )
                if click_mode and _process_click_event(_near_chart_state):
                    st.rerun()
                st.caption(_h3_caption(gdf_out) + _scale_caption(gdf_out))
                _dl_info = (gdf_out, f"nearest_linac_{iso3}_res{h3_resolution}.csv", "dl_nearest")
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
                    metric_cmap_fn = _rdylgn_reversed_rgb
                    tip_series = None  # built below in _count_access_metrics block

                elif access_display_metric == "Modelled Accessed":
                    display_vals = gdf_out["rt_treated"].to_numpy(dtype=np.float64)
                    cb_label_access = "RT accessed"
                    auto_vmin_a = 0.0
                    auto_vmax_a = float(np.nanmax(display_vals))
                    metric_cmap_fn = _rdylgn_rgb
                    tip_series = None  # built below in _count_access_metrics block

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

                elif access_display_metric == "RT Demand":
                    display_vals = gdf_out["rt_demand"].to_numpy(dtype=np.float64)
                    cb_label_access = "RT demand"
                    auto_vmin_a = 0.0
                    auto_vmax_a = float(np.nanmax(display_vals))
                    metric_cmap_fn = _rdylgn_reversed_rgb
                    _s_rt_demand_raw = gdf_out["rt_demand"].round(1).astype(str)
                    if density_per_km2:
                        display_vals = display_vals / (_areas_acc / 10)
                        auto_vmax_a = float(np.nanmax(display_vals))
                        cb_label_access += " per 10 km²"
                    else:
                        cb_label_access += " per hexagon"
                    tip_series = (
                        "<b>" + s_h3 + "</b><br/>"
                        + "RT demand: " + _s_rt_demand_raw + "<br/>"
                        + "RT accessed: " + s_treated + "<br/>"
                        + "RT deficit: " + s_untreated + "<br/>"
                        + "<hr style='margin:3px 0'/>"
                        + "Population: " + s_pop_fmt + "<br/>"
                        + "Hex area: " + _s_area_acc + " km²"
                    )

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
                    if access_display_metric == "Modelled Accessed":
                        tip_series = (
                            "<b>" + s_h3 + "</b><br/>"
                            + "RT accessed: " + s_treated + "<br/>"
                            + "RT access deficit: " + s_untreated + "<br/>"
                            + "Percent accessed: " + s_pct + "%<br/>"
                            + "<hr style='margin:3px 0'/>"
                            + "Population: " + s_pop_fmt + "<br/>"
                            + "Hex area: " + _s_area_acc + " km²"
                        )
                    else:  # Modelled Access Deficit
                        tip_series = (
                            "<b>" + s_h3 + "</b><br/>"
                            + "RT access deficit: " + s_untreated + "<br/>"
                            + "RT accessed: " + s_treated + "<br/>"
                            + "Percent accessed: " + s_pct + "%<br/>"
                            + "<hr style='margin:3px 0'/>"
                            + "Population: " + s_pop_fmt + "<br/>"
                            + "Hex area: " + _s_area_acc + " km²"
                        )

                _map_default_cmap = _DEFAULT_CMAP.get(map_type, "Green → Red")
                active_cmap_fn = metric_cmap_fn if cb_cmap_name == _map_default_cmap else cb_cmap_fn
                # For deficit/demand, higher = worse/more → invert binary colours (above = red)
                _acc_invert_binary = access_display_metric in ("Modelled Access Deficit", "RT Demand")
                colors, vmin, vmax = _color_values(display_vals, active_cmap_fn, auto_vmin_a, auto_vmax_a,
                                                   invert_binary=_acc_invert_binary)

                gdf_out = gdf_out.copy()
                gdf_out["color"] = colors
                # In per-country region mode, lead the tooltip with the country name.
                if "country" in gdf_out.columns:
                    _s_country = gdf_out["country"].fillna("").astype(str)
                    tip_series = np.where(
                        _s_country.values != "",
                        "<b>" + _s_country.values + "</b><br/>" + tip_series.values,
                        tip_series.values,
                    )
                    gdf_out["tip"] = tip_series
                else:
                    gdf_out["tip"] = tip_series.values

                _acc_df = pd.DataFrame({"h3": gdf_out["h3"], "color": gdf_out["color"], "tip": gdf_out["tip"]})
                layers = [] if _no_hex else [_build_hex_layer(_acc_df, _hex_opacity_f)]
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
                                                         on_select="rerun" if click_mode else None, map_key=_main_map_key)
                elif _discrete_scale:
                    _acc_map_col, _acc_leg_col = st.columns([7, 1])
                    with _acc_map_col:
                        _acc_deck = pdk.Deck(layers=_maybe_add_borders(layers, dark_mode), initial_view_state=_make_view(gdf_out, pitch=pitch),
                                             map_style=CARTO_DARK if dark_mode else CARTO_LIGHT, tooltip={"html": "{tip}"})
                        if click_mode and _PYDECK_CLICK_SUPPORTED:
                            _acc_chart_state = st.pydeck_chart(_acc_deck, use_container_width=True, height=_MAP_HEIGHT, key=_view_key(_main_map_key),
                                                               on_select="rerun", selection_mode="single-object")
                        else:
                            st.pydeck_chart(_acc_deck, use_container_width=True, height=_MAP_HEIGHT, key=_view_key(_main_map_key))
                            _acc_chart_state = None
                    with _acc_leg_col:
                        _disc_palette_acc = list(reversed(_DISCRETE_PALETTE[:_discrete_steps])) if _acc_invert_binary else _DISCRETE_PALETTE[:_discrete_steps]
                        _disc_bounds_acc = [n * _discrete_base for n in range(1, _discrete_steps)]
                        _render_discrete_legend(_disc_bounds_acc, _disc_palette_acc, "",
                                                "white" if (dark_mode or app_dark_mode) else "black",
                                                title=cb_label_access)
                else:
                    _acc_chart_state = _render_with_colorbar(
                        layers, _make_view(gdf_out, pitch=pitch),
                        active_cmap_fn, vmin, vmax, cb_label_access,
                        log_scale=cb_log, dark=dark_mode, dark_text=app_dark_mode, clamp=not cb_auto,
                        show_linac_legend=show_linac_markers,
                        on_select="rerun" if click_mode else None,
                        map_key=_main_map_key,
                    )
                if click_mode and _process_click_event(_acc_chart_state):
                    st.rerun()
                st.caption(_h3_caption(gdf_out) + _scale_caption(gdf_out))
                _dl_info = (gdf_out, f"rt_access_{iso3}_res{h3_resolution}.csv", "dl_access")
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
                    return _fmt_sigfig(float(v))

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
                st.caption(f"**Calculated Assuming:** RTU = {_rtu_label}")
                st.metric("RT Demand",
                          _pct_num(_demand, _rt_pct_of_cancer) if _rt_pct_of_cancer is not None else _fmt_k(_demand),
                          help="Annual patients requiring RT · % of cancer incidence")

                st.divider()

                # ── Calculations ─────────────────────────────────────────────
                st.markdown("**Calculations**")
                if access_model == "step":
                    _params_str = (
                        f"**Calculated Assuming:** H3 Resolution = {h3_resolution}, Access Model = Step function, "
                        f"Cut-off = {int(max_distance_km)} km, Capacity per machine = {int(capacity_per_machine_per_year)}"
                    )
                elif access_model == "weibull":
                    _params_str = (
                        f"**Calculated Assuming:** H3 Resolution = {h3_resolution}, Access Model = Weibull, "
                        f"λ = {lambda_km} km, k = {weibull_k}, Capacity per machine = {int(capacity_per_machine_per_year)}"
                    )
                elif access_model == "uniform":
                    _params_str = (
                        f"**Calculated Assuming:** H3 Resolution = {h3_resolution}, Access Model = Uniform (no decay), "
                        f"Capacity per machine = {int(capacity_per_machine_per_year)}"
                    )
                else:
                    _params_str = (
                        f"**Calculated Assuming:** H3 Resolution = {h3_resolution}, Access Model = Exponential, "
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

    # ── Download (bottom of page) ─────────────────────────────────────────
    if _dl_info is not None:
        st.divider()
        _download_results_button(*_dl_info)


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
# Machine Planning tab
# ---------------------------------------------------------------------------

with tab_plan:
    st.header("Machine Planning")
    _plan_result = st.session_state.get("_map_result")
    if _plan_result is None:
        st.info("Run **Calculate RT Access** in the sidebar first.")
    else:
        _plan_gdf_out       = _plan_result["gdf_out"]
        _plan_stats         = _plan_result["stats"]
        _plan_locs_tuple    = _plan_result.get("linac_locs_tuple") or ()
        _plan_facilities_df = _plan_result.get("facilities_df")

        # If locs weren't cached (e.g. loaded from world default pkl), reload from DIRAC
        if not _plan_locs_tuple:
            try:
                if _is_region:
                    from data.linacs import load_linacs_for_region as _llr
                    _plan_locs_raw, _plan_facilities_df = _llr(country)
                else:
                    from data.linacs import load_linacs_from_dirac_db as _lldb
                    _plan_locs_raw, _plan_facilities_df = _lldb(country)
                _plan_locs_tuple = tuple(_plan_locs_raw)
            except Exception:
                _plan_locs_tuple = ()

        # country_span_km — needed for tower rendering
        _plan_geom = _plan_gdf_out.geometry
        _plan_lat_span = float(_plan_geom.bounds["maxy"].max() - _plan_geom.bounds["miny"].min())
        _plan_lon_span = float(_plan_geom.bounds["maxx"].max() - _plan_geom.bounds["minx"].min())
        _plan_lat_mid  = float((_plan_geom.bounds["maxy"].max() + _plan_geom.bounds["miny"].min()) / 2)
        _plan_span_km  = max(
            _plan_lat_span * 111.32,
            _plan_lon_span * 111.32 * math.cos(math.radians(_plan_lat_mid)),
        )

        _has_any_additions = bool(
            st.session_state.get("custom_linacs") or st.session_state.get("_opt_result")
        )
        if st.button("🔄 Reset — clear all added LINACs and optimisation results", type="secondary",
                     use_container_width=True, disabled=not _has_any_additions):
            st.session_state.pop("custom_linacs", None)
            st.session_state.pop("_opt_result", None)
            st.rerun()

        _plan_tab_opt, _plan_tab_custom = st.tabs(["Suggest Optimal Placements", "Add Custom LINACs"])

        # ── Custom LINACs ────────────────────────────────────────────────────
        with _plan_tab_custom:
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
                _add_lat  = st.number_input("Latitude",  min_value=-90.0,  max_value=90.0,  value=0.0, format="%.5f", key="add_linac_lat")
                _add_lon  = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, format="%.5f", key="add_linac_lon")
                _add_cap  = st.number_input("Capacity (pts/yr)", min_value=1, value=450, step=50, key="add_linac_cap")
                if st.button("Add Centre", use_container_width=True, key="add_linac_btn"):
                    _ex = st.session_state.setdefault("custom_linacs", [])
                    _ex.append({
                        "name": _add_name or f"Custom Centre {len(_ex) + 1}",
                        "lat": round(_add_lat, 5), "lon": round(_add_lon, 5),
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
                    num_rows="dynamic", use_container_width=True, key="custom_linacs_editor",
                )
                _cc1, _cc2 = st.columns(2)
                if _cc1.button("Apply edits", key="apply_custom_linacs"):
                    st.session_state["custom_linacs"] = _edited_custom.dropna(subset=["lat", "lon"]).to_dict("records")
                    st.rerun()
                if _cc2.button("Clear all", key="clear_custom_linacs"):
                    st.session_state["custom_linacs"] = []
                    st.rerun()

        # ── Suggest Optimal Placements ────────────────────────────────────────
        with _plan_tab_opt:
            if not _plan_locs_tuple:
                st.warning("No LINAC location data available — recompute RT Access to enable optimisation.")
            else:
                # ---- Mode selector
                _opt_mode = st.radio(
                    "Optimisation mode", ["Add fixed number of facilities", "Add facilities up to threshold"],
                    horizontal=True, key="opt_mode",
                )
                _oc1, _oc2 = st.columns(2)
                _machines_per_new = float(_oc2.number_input(
                    "Machines per new facility", min_value=1, max_value=10, value=1, step=1, key="opt_machines",
                ))
                if _opt_mode == "Add fixed number of facilities":
                    _n_suggest = int(_oc1.number_input(
                        "New facilities to suggest", min_value=1, max_value=9999, value=3, step=1, key="opt_n",
                    ))
                    _thresh_mode = None; _thresh_deficit = None; _thresh_distance = None; _thresh_access = None
                    _opt_max_steps = _n_suggest
                else:
                    _oc1.number_input("Maximum facilities (safety cap)", min_value=1, max_value=9999, value=50, step=1, key="opt_n")
                    _opt_max_steps = int(st.session_state.get("opt_n", 50))
                    _thresh_mode = st.radio(
                        "Threshold:",
                        [
                            "Access percentage",
                            "Deficit per hexagon",
                            "Distance to nearest LINAC",
                        ],
                        horizontal=True, key="opt_thresh_type",
                    )
                    _tc1, _tc2 = st.columns(2)
                    if _thresh_mode == "Access percentage":
                        _thresh_access   = float(_tc1.number_input("Target RadMaps access (%)", min_value=0.01, max_value=99.99, value=80.0, step=0.01, format="%.2f", key="opt_thresh_access")) / 100.0
                        _thresh_deficit  = None; _thresh_distance = None
                        _cur_demand      = float(_plan_stats.get("total_rt_demand", 0))
                        _remaining       = _cur_demand * (1.0 - _thresh_access)
                        _tc2.caption(f"Remaining deficit at target: **{_fmt_sigfig(_remaining)}** patients/yr")
                    elif _thresh_mode == "Deficit per hexagon":
                        _thresh_deficit  = float(_tc1.number_input("Max deficit per hexagon (patients/yr)", min_value=0.1, max_value=10000.0, value=10.0, step=1.0, key="opt_thresh_deficit"))
                        _thresh_distance = None; _thresh_access = None
                        _tc2.caption("Stops when no hexagon exceeds this unmet RT demand.")
                    else:
                        _thresh_distance = float(_tc1.number_input("Max distance to nearest LINAC (km)", min_value=1.0, max_value=2000.0, value=200.0, step=10.0, key="opt_thresh_dist"))
                        _thresh_deficit  = None; _thresh_access = None
                        _tc2.caption("Stops when every populated hexagon is within this distance of a LINAC.")
                    _n_suggest = _opt_max_steps

                _opt_metric_label = st.radio(
                    "Placement / Optimisation metric",
                    ["Highest Unmet RT Demand", "Most Geographically Isolated"],
                    horizontal=True, key="opt_metric",
                )
                _opt_metric = "rt_access" if "Unmet" in _opt_metric_label else "geographic"

                _snap_to_existing = st.checkbox(
                    "Add machines to nearest existing facility if within distance threshold",
                    key="opt_snap_enabled",
                    help="When optimal placement is near an existing or already-suggested facility, add machines there instead.",
                )
                if _snap_to_existing:
                    _sc1, _sc2 = st.columns(2)
                    _snap_km  = float(_sc1.number_input("Threshold — distance (km)", min_value=1, max_value=500, value=50, step=5, key="opt_snap_km"))
                    _snap_min = float(_sc2.number_input("Threshold — travel time (min)", min_value=1, max_value=300, value=60, step=5, key="opt_snap_min", help="Only applied when a TravelTime matrix is available."))
                else:
                    _snap_km = None; _snap_min = None

                if st.button("Run optimisation", key="run_opt", type="primary"):
                    from pyproj import Geod as _OptGeod
                    _geod_opt = _OptGeod(ellps="WGS84")
                    _opt_tt_matrix = None
                    if use_travel_time and _plan_locs_tuple:
                        import hashlib as _hl_opt, json as _json_opt
                        _linac_ll_opt = [(lat, lon) for lat, lon, _ in _plan_locs_tuple]
                        _linac_hash_opt = _hl_opt.md5(_json_opt.dumps(sorted(_linac_ll_opt)).encode()).hexdigest()[:8]
                        _tt_max_h_opt = tt_max_travel_time_sec // 3600
                        _tt_cache_key_opt = f"{iso3}_res{h3_resolution}_{_tt_max_h_opt}h_{_linac_hash_opt}"
                        _opt_tt_cache_file = _TT_CACHE_DIR / f"{_tt_cache_key_opt}_{tt_mode}.npz"
                        if _opt_tt_cache_file.exists():
                            _opt_tt_matrix = np.load(_opt_tt_cache_file)["matrix"]

                    _opt_locs    = list(_plan_locs_tuple)
                    _opt_gdf     = _plan_gdf_out
                    _opt_stats   = _plan_stats
                    # Older cached results (e.g. world_default.pkl) may predate
                    # centroid columns on aggregated gdfs — derive if missing.
                    if "centroid_lat" not in _opt_gdf.columns:
                        import h3 as _h3_opt
                        _cent_opt = _opt_gdf["h3"].apply(lambda _h: _h3_opt.cell_to_latlng(_h))
                        _opt_gdf = _opt_gdf.copy()
                        _opt_gdf["centroid_lat"] = _cent_opt.apply(lambda c: c[0])
                        _opt_gdf["centroid_lon"] = _cent_opt.apply(lambda c: c[1])
                    _opt_suggested: list = []
                    _opt_steps:    list = []
                    _opt_progress = st.progress(0, text="Running optimisation…")
                    _opt_stop_reason: str = ""

                    for _opt_step in range(_n_suggest):
                        if _thresh_mode is not None:
                            if _thresh_deficit is not None:
                                _cur_max_deficit = float(_opt_gdf["rt_untreated"].max())
                                if _cur_max_deficit < _thresh_deficit:
                                    _opt_stop_reason = f"Threshold met after {_opt_step} facilit{'y' if _opt_step == 1 else 'ies'} — max deficit {_cur_max_deficit:.1f} < {_thresh_deficit} pts/hex"
                                    break
                            elif _thresh_distance is not None:
                                _cur_max_dist = float(_opt_gdf["nearest_linac_km"].replace([np.inf], np.nan).dropna().max() if not _opt_gdf["nearest_linac_km"].dropna().empty else np.inf)
                                if _cur_max_dist < _thresh_distance:
                                    _opt_stop_reason = f"Threshold met after {_opt_step} facilit{'y' if _opt_step == 1 else 'ies'} — max distance {_cur_max_dist:.0f} km < {_thresh_distance:.0f} km"
                                    break
                            elif _thresh_access is not None:
                                _cur_ratio = (_opt_stats["total_rt_treated"] / _opt_stats["total_rt_demand"]) if _opt_stats["total_rt_demand"] > 0 else 0.0
                                if _cur_ratio >= _thresh_access:
                                    _opt_stop_reason = f"Threshold met after {_opt_step} facilit{'y' if _opt_step == 1 else 'ies'} — access {_cur_ratio:.1%} ≥ {_thresh_access:.1%}"
                                    break

                        _olats = _opt_gdf["centroid_lat"].to_numpy()
                        _olons = _opt_gdf["centroid_lon"].to_numpy()
                        _opop  = _opt_gdf["population"].to_numpy()
                        if _opt_metric == "rt_access":
                            _oscores = _opt_gdf["rt_untreated"].to_numpy(dtype=np.float64)
                        else:
                            _oscores = _opt_gdf["nearest_linac_km"].fillna(0).to_numpy(dtype=np.float64) * np.where(_opop > 0, _opop, 0.0)

                        _obest    = int(np.argmax(_oscores))
                        _onew_lat = float(_olats[_obest])
                        _onew_lon = float(_olons[_obest])

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
                            _sfl, _sfo, _sfw = _opt_locs[_snap_idx]
                            _opt_locs[_snap_idx] = (_sfl, _sfo, _sfw + _machines_per_new)
                            _onew_lat, _onew_lon = _sfl, _sfo
                            _placement_note = f"Merged into existing ({_sfl:.3f}, {_sfo:.3f})"
                        else:
                            _opt_locs.append((_onew_lat, _onew_lon, _machines_per_new))
                            _placement_note = "New facility"

                        _onew_gdf, _onew_stats = _compute_access(
                            country, iso3, tuple(_opt_locs),
                            float(lambda_km), access_model, float(max_distance_km),
                            capacity_per_machine_per_year, access_rt_method, access_rt_fraction,
                            h3_resolution, _is_region, snap_linacs_to_hex,
                            weibull_k=float(weibull_k), custom_rtu=access_custom_rtu,
                        )
                        _oimprove  = _onew_stats["total_rt_treated"] - _opt_stats["total_rt_treated"]
                        _onew_ratio = _onew_stats["total_rt_treated"] / _onew_stats["total_rt_demand"] if _onew_stats["total_rt_demand"] > 0 else 0.0
                        _opt_suggested.append({
                            "name": f"Suggested {_opt_step + 1}",
                            "lat": round(_onew_lat, 4), "lon": round(_onew_lon, 4),
                            "n_linacs": _machines_per_new,
                            "capacity": _machines_per_new * capacity_per_machine_per_year,
                        })
                        _opt_steps.append({
                            "Step": _opt_step + 1, "Placement": _placement_note,
                            "Latitude": round(_onew_lat, 4), "Longitude": round(_onew_lon, 4),
                            "RT treated improvement": f"+{_fmt_sigfig(_oimprove)}",
                            "Cumulative access ratio": f"{_onew_ratio:.1%}",
                        })
                        _opt_gdf   = _onew_gdf
                        _opt_stats = _onew_stats
                        _pct_done  = (_opt_step + 1) / _n_suggest
                        if _thresh_deficit is not None:
                            _thresh_text = f" — max deficit {float(_opt_gdf['rt_untreated'].max()):.1f} pts/hex"
                        elif _thresh_distance is not None:
                            _cur_dist_now = float(_opt_gdf["nearest_linac_km"].replace([np.inf], np.nan).dropna().max() if not _opt_gdf["nearest_linac_km"].dropna().empty else np.inf)
                            _thresh_text = f" — max distance {_cur_dist_now:.0f} km"
                        elif _thresh_access is not None:
                            _cur_ratio_now = (_opt_stats["total_rt_treated"] / _opt_stats["total_rt_demand"]) if _opt_stats["total_rt_demand"] > 0 else 0.0
                            _thresh_text = f" — access {_cur_ratio_now:.1%} / {_thresh_access:.1%}"
                        else:
                            _thresh_text = f" — +{_fmt_sigfig(_oimprove)} patients treated"
                        _opt_progress.progress(_pct_done, text=f"Step {_opt_step + 1}/{_n_suggest}{_thresh_text}")

                    _opt_progress.empty()
                    if _opt_stop_reason:
                        st.success(_opt_stop_reason)
                    st.session_state["_opt_result"] = {
                        "suggested": _opt_suggested, "steps": _opt_steps,
                        "final_gdf": _opt_gdf, "final_stats": _opt_stats,
                        "final_locs": _opt_locs, "baseline_stats": _plan_stats,
                    }

                _opt_res = st.session_state.get("_opt_result")
                if _opt_res:
                    _opt_sug          = _opt_res["suggested"]
                    _opt_final_gdf    = _opt_res["final_gdf"]
                    _opt_final_stats  = _opt_res["final_stats"]
                    _opt_baseline     = _opt_res.get("baseline_stats", _plan_stats)

                    # ── Map ──────────────────────────────────────────────────
                    _opt_raw    = _opt_final_gdf["rt_untreated"].to_numpy(dtype=np.float64)
                    _opt_vmin   = 0.0
                    _opt_vmax   = float(np.nanmax(_opt_raw)) if _opt_raw.size > 0 else 1.0
                    _opt_colors, _, _ = _color_values(_opt_raw, _rdylgn_reversed_rgb, _opt_vmin, _opt_vmax, invert_binary=True)
                    _opt_final_gdf = _opt_final_gdf.copy()
                    _opt_final_gdf["color"] = _opt_colors
                    _opt_final_gdf["tip"] = (
                        "<b>" + _opt_final_gdf["h3"].astype(str) + "</b><br/>"
                        + "RT deficit (optimised): " + _opt_final_gdf["rt_untreated"].round(1).astype(str)
                    )
                    _opt_hex_layer = _build_hex_layer(pd.DataFrame({
                        "h3": _opt_final_gdf["h3"], "color": _opt_final_gdf["color"], "tip": _opt_final_gdf["tip"],
                    }), _hex_opacity_f)
                    _opt_existing_df = _plan_facilities_df.copy() if _plan_facilities_df is not None and not _plan_facilities_df.empty else pd.DataFrame()
                    _opt_new_df = pd.DataFrame(_opt_sug)
                    _opt_combined_parts = []
                    if show_linac_markers and not _opt_existing_df.empty:
                        _ex = _opt_existing_df.copy()
                        _ex["color"] = [[0, 140, 255, 240]] * len(_ex)
                        _ex["_stack_order"] = 0
                        _opt_combined_parts.append(_ex)
                    if not _opt_new_df.empty:
                        _nw = _opt_new_df.copy()
                        _nw["color"] = [[160, 32, 240, 240]] * len(_nw)
                        _nw["_stack_order"] = 1
                        _opt_combined_parts.append(_nw)
                    _opt_layers = [_opt_hex_layer]
                    if _opt_combined_parts:
                        _opt_layers.extend(_build_linac_columns(
                            pd.concat(_opt_combined_parts, ignore_index=True),
                            h3_res=h3_resolution, country_span_km=_plan_span_km,
                            height_scale=tower_height_scale, radius_scale=tower_radius_scale,
                            style=linac_tower_style,
                        ))
                    st.caption("RT Access Deficit after suggested placements. Blue = existing facilities, purple = suggested placements.")
                    _opt_view = _make_view(_opt_final_gdf, pitch=30.0 if (show_linac_markers and map_pitch_on) else 0.0)
                    _opt_map_key = f"optmap_{iso3}_{h3_resolution}"
                    if _discrete_scale:
                        _opt_map_col, _opt_leg_col = st.columns([7, 1])
                        with _opt_map_col:
                            st.pydeck_chart(pdk.Deck(layers=_maybe_add_borders(_opt_layers, dark_mode), initial_view_state=_opt_view,
                                                     map_style=CARTO_DARK if dark_mode else CARTO_LIGHT,
                                                     tooltip={"html": "{tip}"}), use_container_width=True, height=_MAP_HEIGHT, key=_view_key(_opt_map_key))
                        with _opt_leg_col:
                            _render_discrete_legend(
                                [n * _discrete_base for n in range(1, _discrete_steps)],
                                list(reversed(_DISCRETE_PALETTE[:_discrete_steps])),
                                "", "white" if (dark_mode or app_dark_mode) else "black",
                                title="RT access deficit per hexagon",
                            )
                    else:
                        _render_with_colorbar(
                            _opt_layers, _opt_view,
                            _rdylgn_reversed_rgb, _opt_vmin, _opt_vmax, "RT access deficit",
                            dark=dark_mode, dark_text=app_dark_mode,
                            map_key=_opt_map_key,
                        )

                    # ── Step table ───────────────────────────────────────────
                    st.markdown(f"**{len(_opt_sug)} placement(s)** — step-by-step improvement:")
                    st.dataframe(pd.DataFrame(_opt_res["steps"]), use_container_width=True, hide_index=True)

                    # ── Summary metrics ──────────────────────────────────────
                    st.divider()
                    st.markdown("**Summary**")
                    _ob  = _opt_baseline
                    _oa  = _opt_final_stats
                    _dem = _oa["total_rt_demand"]

                    # Counts
                    _n_fac_before  = int(_ob["n_facilities"])
                    _n_fac_after   = int(_oa["n_facilities"])
                    _n_mac_before  = int(_ob["total_machines"])
                    _n_mac_after   = int(_oa["total_machines"])
                    _n_new_fac     = _n_fac_after - _n_fac_before
                    _n_new_mac     = _n_mac_after - _n_mac_before

                    # Access ratios
                    _ob_radmaps   = _ob["total_rt_treated"] / _ob["total_rt_demand"] if _ob["total_rt_demand"] > 0 else 0.0
                    _oa_radmaps   = _oa["total_rt_treated"] / _dem if _dem > 0 else 0.0
                    _ob_cap       = min(_ob["total_national_capacity"] / _ob["total_rt_demand"], 1.0) if _ob["total_rt_demand"] > 0 else 0.0
                    _oa_cap       = min(_oa["total_national_capacity"] / _dem, 1.0) if _dem > 0 else 0.0
                    _ob_geo       = _ob.get("mean_access_probability", 0.0)
                    _oa_geo       = _oa.get("mean_access_probability", 0.0)

                    _sm1, _sm2, _sm3, _sm4 = st.columns(4)
                    _sm1.metric("New facilities", f"+{_n_new_fac}")
                    _sm2.metric("New machines", f"+{_n_new_mac}")
                    _sm3.metric("Total facilities", _n_fac_after, delta=f"+{_n_new_fac}")
                    _sm4.metric("Total machines", _n_mac_after, delta=f"+{_n_new_mac}")

                    _ob_treated   = _ob["total_rt_treated"]
                    _oa_treated   = _oa["total_rt_treated"]
                    _ob_cap_pts   = min(_ob["total_national_capacity"], _dem)
                    _oa_cap_pts   = min(_oa["total_national_capacity"], _dem)
                    _oa_cap_def   = max(_dem - _oa["total_national_capacity"], 0.0)

                    st.divider()
                    st.markdown("**RT Demand**")
                    _sd1, _sd2 = st.columns(2)
                    _sd1.metric("RT Demand", _fmt_k(_dem), help="Unchanged by adding facilities")
                    _sd2.metric("RT Treated (before)", _pct_num(_ob_treated, _ob_radmaps))

                    st.divider()
                    st.markdown("**RadMaps Access**")
                    _sr1, _sr2, _sr3 = st.columns(3)
                    _sr1.metric("Before", _pct_num(_ob_treated, _ob_radmaps))
                    _sr2.metric("After",  _pct_num(_oa_treated, _oa_radmaps),
                                delta=f"+{(_oa_radmaps - _ob_radmaps):.1%}")
                    _sr3.metric("Additional treated",
                                _pct_num(_oa_treated - _ob_treated, _oa_radmaps - _ob_radmaps))

                    st.markdown("**Capacity-only Access**")
                    _sc1b, _sc2b, _sc3b = st.columns(3)
                    _sc1b.metric("Before", _pct_num(_ob_cap_pts, _ob_cap))
                    _sc2b.metric("After",  _pct_num(_oa_cap_pts, _oa_cap),
                                 delta=f"+{(_oa_cap - _ob_cap):.1%}")
                    _sc3b.metric("Capacity deficit (after)", _pct_num(_oa_cap_def, 1.0 - _oa_cap))

                    st.markdown("**Geography-only Access**")
                    _sg1, _sg2 = st.columns(2)
                    _sg1.metric("Before", _pct_num(_ob_geo * _dem, _ob_geo))
                    _sg2.metric("After",  _pct_num(_oa_geo * _dem, _oa_geo),
                                delta=f"+{(_oa_geo - _ob_geo):.1%}")

# ---------------------------------------------------------------------------
# Method tab
# ---------------------------------------------------------------------------

with _tab_sep0:
    pass

with _tab_sep1:
    pass

with _tab_sep2:
    pass

with _tab_sep3:
    pass

# ---------------------------------------------------------------------------
# Introduction tab
# ---------------------------------------------------------------------------

with tab_intro:
    render_introduction()

# ---------------------------------------------------------------------------
# Method tab
# ---------------------------------------------------------------------------

with tab_method:
    render_method()


# ---------------------------------------------------------------------------
# Assumptions tab
# ---------------------------------------------------------------------------

with tab_assumptions:
    render_assumptions()

# ---------------------------------------------------------------------------
# Toy Example tab
# ---------------------------------------------------------------------------

with tab_toy:
    render_toy_example()

# ---------------------------------------------------------------------------
# Sensitivity & Equity tab
# ---------------------------------------------------------------------------

with tab_sensitivity:
    st.header("Sensitivity & Equity")
    st.markdown(
        "All headline figures are point estimates from hand-set parameters. "
        "This tab quantifies how much they move when the two most uncertain "
        "inputs — the decay scale **λ** and the **capacity per LINAC** — are "
        "varied, and summarises how *equally* access is distributed."
    )

    # ---- Equity summary from the current computed map (no recompute) ----
    _sens_result = st.session_state.get("_map_result")
    if not _sens_result or "gdf_out" not in _sens_result:
        st.info(
            "Compute an **Access Map** first (press *Calculate RT Access*). "
            "The equity and sensitivity analysis runs on that result."
        )
    else:
        _sg = _sens_result["gdf_out"]
        _spop = _sg["population"].to_numpy(dtype=np.float64)
        _sprob = _sg["access_probability"].to_numpy(dtype=np.float64)
        _tot_pop = float(_spop.sum())

        st.subheader("Equity of access")
        st.caption(
            "Access is population-weighted geographic access probability per hexagon. "
            "The Gini coefficient ranges 0 (everyone equal) to 1 (maximally unequal)."
        )

        # Population-weighted Gini of the access shortfall (1 - access)
        # Sort by access ascending; build Lorenz curve of "people with access".
        _order = np.argsort(_sprob)
        _pop_sorted = _spop[_order]
        _access_sorted = _sprob[_order]
        _cum_pop = np.cumsum(_pop_sorted)
        _served = _pop_sorted * _access_sorted
        _cum_served = np.cumsum(_served)
        _tot_served = float(_cum_served[-1]) if _cum_served.size else 0.0

        if _tot_pop > 0 and _tot_served > 0:
            _lx = np.concatenate([[0.0], _cum_pop / _tot_pop])
            _ly = np.concatenate([[0.0], _cum_served / _tot_served])
            # Gini = 1 - 2 * area under Lorenz curve
            # np.trapz was renamed np.trapezoid in numpy 2.0 (removed in newer builds)
            _trapz_fn = getattr(np, "trapezoid", None) or np.trapz
            _gini = 1.0 - 2.0 * float(_trapz_fn(_ly, _lx))
        else:
            _lx = np.array([0.0, 1.0]); _ly = np.array([0.0, 1.0]); _gini = 0.0

        # Coverage thresholds
        _has_min_col = "nearest_linac_min" in _sg.columns
        if _has_min_col:
            _dvals = _sg["nearest_linac_min"].to_numpy(dtype=np.float64)
            _cov_thresholds = [(30, "min"), (60, "min"), (120, "min")]
        else:
            _dvals = _sg["nearest_linac_km"].to_numpy(dtype=np.float64)
            _cov_thresholds = [(50, "km"), (100, "km"), (200, "km")]

        c_eq0, c_eq1, c_eq2, c_eq3 = st.columns(4)
        c_eq0.metric("Access Gini", f"{_gini:.3f}")
        _mean_access = float(np.nansum(_sprob * _spop) / _tot_pop) if _tot_pop > 0 else 0.0
        c_eq1.metric("Mean access (pop-wtd)", f"{_mean_access:.1%}")
        for _c, (_thr, _u) in zip((c_eq2, c_eq3), _cov_thresholds[:2]):
            _within = float(_spop[np.isfinite(_dvals) & (_dvals <= _thr)].sum())
            _c.metric(f"Pop within {_thr} {_u}", f"{(_within / _tot_pop if _tot_pop > 0 else 0):.1%}")

        # Lorenz curve
        _fig_lor = go.Figure()
        _fig_lor.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#888", dash="dash"), name="Perfect equality",
        ))
        _fig_lor.add_trace(go.Scatter(
            x=_lx, y=_ly, mode="lines", fill="tonexty",
            line=dict(color="#1f77b4", width=2.5), name="Access distribution",
        ))
        _fig_lor.update_layout(
            title="Lorenz curve — cumulative access vs cumulative population",
            xaxis_title="Cumulative share of population (poorest access first)",
            yaxis_title="Cumulative share of RT access",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
            height=380, margin=dict(t=40, b=40),
        )
        st.plotly_chart(_fig_lor, use_container_width=True)

        st.divider()

        # ---- Parameter sensitivity sweep (single country, non-TT only) ----
        st.subheader("Parameter sensitivity")
        _sens_supported = (not _is_region) and (not use_travel_time)
        if not _sens_supported:
            st.info(
                "The parameter sweep runs for a **single country** using "
                "**straight-line distance**. Select one country and disable travel "
                "time to enable it. (Regional and travel-time sweeps are too expensive "
                "to run interactively.)"
            )
        else:
            _sweep = st.slider("Vary each parameter by ± this fraction", 10, 50, 25, step=5,
                               format="%d%%", key="_sens_sweep") / 100.0
            _sens_locs = _sens_result.get("linac_locs_tuple") or ()
            if not _sens_locs:
                st.caption("No stored LINAC locations in the current result — recompute the access map.")
            if st.button("Run sensitivity sweep", key="_run_sens", disabled=not _sens_locs):
                _base_lambda = float(lambda_km)
                _base_cap = float(capacity_per_machine_per_year)
                _factors = [(1 - _sweep), 1.0, (1 + _sweep)]
                _rows = []
                _prog = st.progress(0.0, text="Running scenarios…")
                _scenarios = [("λ", f, 1.0) for f in _factors] + [("Capacity", 1.0, f) for f in _factors]
                # Deduplicate the shared baseline (λ×1, cap×1)
                _seen = set()
                _n = len(_scenarios)
                for _i, (_which, _lf, _cf) in enumerate(_scenarios):
                    _key = (round(_lf, 3), round(_cf, 3))
                    if _key in _seen:
                        _prog.progress((_i + 1) / _n)
                        continue
                    _seen.add(_key)
                    try:
                        _g_s, _st_s = _compute_access(
                            country, iso3, tuple(_sens_locs),
                            _base_lambda * _lf, access_model, float(max_distance_km),
                            _base_cap * _cf, access_rt_method, access_rt_fraction,
                            h3_resolution, _is_region, snap_linacs_to_hex,
                            weibull_k=float(weibull_k), custom_rtu=access_custom_rtu,
                        )
                    except Exception as _e:
                        _prog.progress((_i + 1) / _n)
                        continue
                    _tp = _st_s.get("total_population", 0.0) or 0.0
                    _pa = _st_s.get("pop_with_access", 0.0) or 0.0
                    _rows.append({
                        "Scenario": f"{_which} ×{(_lf if _which == 'λ' else _cf):.2f}",
                        "λ (km)": round(_base_lambda * _lf, 1),
                        "Capacity": round(_base_cap * _cf, 0),
                        "% population with access": round(100 * _pa / _tp, 1) if _tp > 0 else 0.0,
                        "% RT demand met": round(100 * _st_s.get("mean_capacity_limited_probability", 0.0), 1),
                    })
                    _prog.progress((_i + 1) / _n)
                _prog.empty()

                if _rows:
                    _sdf = pd.DataFrame(_rows)
                    st.session_state["_sens_table"] = _sdf

            if "_sens_table" in st.session_state:
                _sdf = st.session_state["_sens_table"]
                st.dataframe(_sdf, use_container_width=True, hide_index=True)
                _acc_rng = _sdf["% population with access"]
                _met_rng = _sdf["% RT demand met"]
                st.markdown(
                    f"**Population with access** ranges "
                    f"**{_acc_rng.min():.1f}% – {_acc_rng.max():.1f}%** "
                    f"(spread {_acc_rng.max() - _acc_rng.min():.1f} pts). "
                    f"**RT demand met** ranges **{_met_rng.min():.1f}% – {_met_rng.max():.1f}%** "
                    f"(spread {_met_rng.max() - _met_rng.min():.1f} pts) across the swept range."
                )

    st.divider()

    # ---- Calibration against observed RTU (if data present) ----
    st.subheader("Calibration against observed utilisation")
    from data.cancer import ACTUAL_DIR as _ACTUAL_DIR
    _actual_files = sorted(_ACTUAL_DIR.glob("*.csv")) if _ACTUAL_DIR.exists() else []
    if not _actual_files:
        st.info(
            "No observed radiotherapy-utilisation data is bundled "
            f"(`{_ACTUAL_DIR}` is empty or absent). When per-country actual RTU "
            "files are added, this section will compare modelled treated-case "
            "counts against observed utilisation so λ and k can be calibrated "
            "rather than assumed. **Caveat:** calibration also inherits strong "
            "assumptions — notably that every LINAC treats the same number of "
            "patients per year — so calibrated parameters should be read as "
            "indicative, not definitive."
        )
    else:
        st.caption(
            f"{len(_actual_files)} countries have observed RTU data. Comparison "
            "uses the current model settings."
        )
        _cal_rows = []
        for _f in _actual_files:
            _ciso = _f.stem
            try:
                _obs = pd.read_csv(_f, header=None, names=["cancer", "fraction"])
                _obs_mean = float(_obs["fraction"].mean())
                _cal_rows.append({"ISO3": _ciso, "Mean observed RTU fraction": round(_obs_mean, 3)})
            except Exception:
                continue
        if _cal_rows:
            st.dataframe(pd.DataFrame(_cal_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Convergence Testing tab
# ---------------------------------------------------------------------------

with tab_convergence:
    st.header("🔬 Convergence Testing")
    st.markdown(
        "How much does the access estimate depend on H3 resolution? This runs the "
        "full **A_C / A_G / A_RM** calculation across resolutions 3–8 for one country, "
        "using both **straight-line distance** (computed natively at every resolution) "
        "and **driving time** (fetched natively from the TravelTime API at res 5–8, "
        "then aggregated down to res 3–4). It shows where each metric stabilises — and "
        "where coarse grids mislead."
    )
    st.caption(
        "Hex size by resolution:  res 3 ≈ 12,400 km²  ·  4 ≈ 1,770  ·  5 ≈ 253  ·  "
        "6 ≈ 36  ·  7 ≈ 5.2  ·  8 ≈ 0.74 km².  A_G is demand-weighted; A_C is "
        "capacity ÷ demand (resolution-invariant)."
    )

    _conv_opts = _country_options()
    _conv_default = _conv_opts.index("Norway") if "Norway" in _conv_opts else 0
    _cc1, _cc2 = st.columns([3, 1])
    with _cc1:
        _conv_country = st.selectbox("Country", _conv_opts, index=_conv_default, key="_conv_country")
    with _cc2:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        _conv_go = st.button("Run convergence test", type="primary", use_container_width=True)

    if _conv_go:
        st.session_state["_conv_active"] = _conv_country

    _conv_active = st.session_state.get("_conv_active")
    if not _conv_active:
        st.info("Pick a country and click **Run convergence test**. Driving-time fetches "
                "are cached to disk, so re-runs are instant. Small countries (Norway, Nepal, "
                "Ecuador) run in seconds; large ones take longer at res 8.")
    else:
        try:
            _conv_iso3 = pycountry.countries.lookup(_conv_active).alpha_3
        except LookupError:
            st.error(f"Could not resolve country: {_conv_active!r}")
            st.stop()
        if _conv_iso3 in _TT_UNSUPPORTED_ISO3:
            st.info(f"ℹ️ TravelTime has no driving-network coverage for **{_conv_active}**, so the "
                    "driving-time panel will be empty — only the straight-line distance panel is meaningful here.")
        _conv_tt = st.secrets.get("traveltime", {}) if hasattr(st, "secrets") else {}
        if not _conv_tt.get("app_id") or not _conv_tt.get("api_key"):
            st.warning("No TravelTime API credentials found — the driving-time panel needs them "
                       "(add `[traveltime]` app_id/api_key to `.streamlit/secrets.toml`). "
                       "The distance panel works without them.")
        with st.spinner(f"Running convergence for {_conv_active} — first run fetches driving "
                        "times at res 5–8 (~15–90 s); cached thereafter…"):
            try:
                _conv_df = _run_convergence_cached(_conv_active, _conv_iso3)
            except ValueError as _conv_exc:
                st.warning(str(_conv_exc) + " Pick a country that has radiotherapy facilities.")
                st.stop()

        _conv_errs = _conv_df.attrs.get("errors") or []
        if _conv_errs:
            st.warning("Some driving-time batches failed (country may be partly unsupported by "
                       "TravelTime): " + "; ".join(_conv_errs))
        _conv_nfac = _conv_df.attrs.get("n_facilities", 0)
        _conv_ac = float(_conv_df["A_C"].iloc[0]) if len(_conv_df) else float("nan")
        st.caption(f"**{_conv_active}** — {_conv_nfac} facilities  ·  "
                   f"A_C = {_conv_ac:.1%} (flat across all resolutions)")

        _conv_metric_label = st.radio(
            "Metric to plot", ["A_G — geographic access", "A_RM — modelled access ratio"],
            horizontal=True, key="_conv_metric",
        )
        _conv_mcol = "A_G" if _conv_metric_label.startswith("A_G") else "A_RM"

        _cd, _ct = st.columns(2)
        for _pcol, _kind, _unit, _ptitle in [
            (_cd, "distance", "km", "Straight-line distance"),
            (_ct, "time", "min", "Driving time"),
        ]:
            with _pcol:
                _sub = _conv_df[_conv_df["metric"] == _kind]
                _fig = go.Figure()
                for _thr in sorted(_sub["threshold"].unique()):
                    _s = _sub[_sub["threshold"] == _thr].sort_values("resolution")
                    _fig.add_trace(go.Scatter(
                        x=_s["resolution"], y=_s[_conv_mcol] * 100,
                        mode="lines+markers", name=f"{_thr:g} {_unit}",
                    ))
                _fig.update_layout(
                    title=f"{_ptitle} — {_conv_mcol}",
                    xaxis_title="H3 resolution (finer →)",
                    yaxis_title=f"{_conv_mcol} (%)",
                    xaxis=dict(dtick=1), yaxis=dict(range=[0, 100]),
                    height=430, legend_title="Threshold", margin=dict(t=45, b=40),
                )
                st.plotly_chart(_fig, use_container_width=True)

        st.caption(
            "Reading it: flat lines = converged (finer resolution wouldn't change the answer). "
            "For distance, metrics typically settle by res 5–6. For driving time the small "
            "thresholds (30 min) can still be climbing at res 8, and coarse resolutions (3–4, "
            "aggregated) can over- or under-state access depending on threshold."
        )

        with st.expander("Full data table"):
            _conv_show = _conv_df[["metric", "resolution", "threshold", "unit", "A_C", "A_G", "A_RM"]].copy()
            for _c in ("A_C", "A_G", "A_RM"):
                _conv_show[_c] = (_conv_show[_c] * 100).round(1)
            st.dataframe(_conv_show, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# World Choropleth tab — global per-country overview from the precomputed lookup
# ---------------------------------------------------------------------------

_COUNTRY_METRICS_PATH = Path(__file__).resolve().parent / "data" / "country_metrics.parquet"


@st.cache_data(show_spinner=False)
def _load_country_metrics():
    """Load the precomputed per-country access lookup (or None if absent)."""
    if not _COUNTRY_METRICS_PATH.exists():
        return None
    return pd.read_parquet(_COUNTRY_METRICS_PATH)


with tab_choropleth:
    st.header("🗺️ World Choropleth")
    st.caption(
        "One value per country — the **national, demand-weighted** access metric. "
        "This is a coarser view than the per-hexagon Access Maps (which resolve access "
        "*within* a country); here each country is a single number. Precomputed at H3 "
        "resolution 5 from the corrected DIRAC facilities and cached driving-time matrices."
    )

    _cm = _load_country_metrics()
    if _cm is None or _cm.empty:
        st.info(
            "Country metrics lookup not found. Generate it with "
            "`python scripts/build_country_metrics.py`."
        )
    else:
        _cc1, _cc2 = st.columns([1.1, 1.4])
        _chor_metric = _cc1.radio(
            "Metric", ["Capacity", "Geography", "RadMaps"],
            horizontal=True, key="chor_metric",
            help="Capacity = supply adequacy A_C = min(1, machines·450 / demand). "
                 "Geography = demand-weighted geographic reach A_G. "
                 "RadMaps = realised treatment A_RM (supply × access).",
        )
        _chor_is_cap = _chor_metric == "Capacity"
        _chor_thr = _cc2.radio(
            "Threshold", ["50 km", "100 km", "200 km", "30 min", "60 min", "120 min"],
            index=5, horizontal=True, key="chor_thr", disabled=_chor_is_cap,
            help="Distance or driving-time cut-off. Not applicable to Capacity — "
                 "supply adequacy is threshold-free.",
        )

        # Resolve the column + a human title.
        if _chor_is_cap:
            _chor_col = "A_C"
            _chor_title = "Capacity — supply adequacy (A_C)"
            _chor_is_time = False
        else:
            _chor_code = {"Geography": "A_G", "RadMaps": "A_RM"}[_chor_metric]
            _chor_unit = _chor_thr.replace(" ", "")           # "50km" / "120min"
            _chor_col = f"{_chor_code}_{_chor_unit}"
            _chor_is_time = _chor_unit.endswith("min")
            _chor_metric_full = {"Geography": "Geographic reach (A_G)",
                                 "RadMaps": "Realised treatment (A_RM)"}[_chor_metric]
            _chor_title = f"{_chor_metric_full} within {_chor_thr}"

        _cdf = _cm[["iso3", "name", "n_machines", "time_source", _chor_col]].copy()
        _cdf = _cdf.rename(columns={_chor_col: "value"}).dropna(subset=["value"])
        _cdf["pct"] = (_cdf["value"] * 100).round(1)

        # Discrete 5-band colouring, reusing the app's discrete palette (honours the
        # colour-blind-safe toggle via _DISCRETE_PALETTE). Bands are fixed 20% steps.
        _NO_DATA = "No data"
        _band_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
        _band_labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
        _cdf["band"] = pd.cut(_cdf["value"], bins=_band_edges, labels=_band_labels,
                              include_lowest=True).astype(str)
        _pal = _DISCRETE_PALETTE[:5]
        _band_color = {lab: f"rgb({c[0]},{c[1]},{c[2]})" for lab, c in zip(_band_labels, _pal)}
        _band_color[_NO_DATA] = "rgb(158,158,158)"   # matches the map's no-data land grey

        # Per-country hover note.
        def _chor_note(r):
            if r["n_machines"] == 0:
                return "no radiotherapy facilities"
            if _chor_is_time and r["time_source"] == "proxy":
                return "driving time estimated (distance proxy)"
            return ""
        _cdf["note"] = _cdf.apply(_chor_note, axis=1)
        _cdf["hlabel"] = _cdf.apply(
            lambda r: f"<b>{r['name']}</b><br>value: {r['pct']:.1f}%"
                      + (f"<br>{r['note']}" if r["note"] else ""), axis=1)

        # Add every other world country as an explicit "No data" band so it renders
        # grey *and* gets a hover ("no GLOBOCAN cancer data"), instead of being a
        # silent hole on the base map.
        _have_iso3 = set(_cdf["iso3"])
        _missing_rows = [
            {"iso3": c.alpha_3, "name": c.name, "band": _NO_DATA, "pct": float("nan"),
             "hlabel": f"<b>{c.name}</b><br>no GLOBOCAN cancer data — not computed"}
            for c in pycountry.countries
            if getattr(c, "alpha_3", None) and c.alpha_3 not in _have_iso3
        ]
        _full = pd.concat([_cdf, pd.DataFrame(_missing_rows)], ignore_index=True)

        import plotly.express as px
        _fig_chor = px.choropleth(
            _full, locations="iso3", locationmode="ISO-3",
            color="band", color_discrete_map=_band_color,
            category_orders={"band": _band_labels + [_NO_DATA]},
            custom_data=["hlabel"],
            title=_chor_title,
        )
        _fig_chor.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>",
        )
        _fig_chor.update_layout(
            legend_title_text="Access band",
            margin=dict(l=0, r=0, t=40, b=0),
            height=560,
            geo=dict(showframe=False, showcoastlines=False,
                     showcountries=True, countrycolor="white", countrywidth=0.5,
                     # Clear mid-grey for countries we can't compute (no GLOBOCAN
                     # data) — distinct from the ocean so "no data" reads as such.
                     landcolor="#9e9e9e", showland=True,
                     projection_type="natural earth", bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_fig_chor, use_container_width=True)

        _n_grey = int((_cdf["n_machines"] == 0).sum())
        _mean_v = float(_cdf["value"].mean())
        st.caption(
            f"{len(_cdf)} countries with GLOBOCAN data · mean = {_mean_v*100:.0f}% · "
            f"{_n_grey} have no radiotherapy facilities (shown at 0%). "
            "Countries we can't compute (no GLOBOCAN cancer data) are shown in grey. "
            + ("Driving times for countries TravelTime does not cover use a fitted "
               "distance proxy (flagged on hover)." if _chor_is_time else "")
        )

# ---------------------------------------------------------------------------
# Country Analysis tab — supply vs access bottleneck across all countries
# ---------------------------------------------------------------------------

with tab_country:
    st.header("📈 Country Analysis")
    st.caption(
        "Every country's radiotherapy access split into its two constraints, as a "
        "fraction of national RT demand: **supply adequacy** $A_C$ and **geographic "
        "reach** $A_G$. Realised treatment $A_{RM}$ is bounded by both — where it falls "
        "below the lower ceiling, supply and reachable demand sit in different places "
        "(*spatial mismatch*)."
    )

    _cam = _load_country_metrics()
    if _cam is None or _cam.empty:
        st.info(
            "Country metrics lookup not found. Generate it with "
            "`python scripts/build_country_metrics.py`."
        )
    else:
        _ca_thr = st.radio(
            "Threshold", ["50 km", "100 km", "200 km", "30 min", "60 min", "120 min"],
            index=5, horizontal=True, key="ca_thr",
            help="Distance or driving-time cut-off applied to A_G and A_RM. "
                 "A_C (supply) is threshold-free.",
        )
        _ca_unit = _ca_thr.replace(" ", "")          # "50km" / "120min"
        _ca_gcol, _ca_rcol = f"A_G_{_ca_unit}", f"A_RM_{_ca_unit}"

        _ca = _cam[["iso3", "name", "n_machines", "A_C", _ca_gcol, _ca_rcol]].copy()
        _ca = _ca.rename(columns={_ca_gcol: "A_G", _ca_rcol: "A_RM"}).dropna(subset=["A_G", "A_RM"])
        _ca["mismatch"] = _ca[["A_C", "A_G"]].min(axis=1) - _ca["A_RM"]

        import plotly.express as px
        import plotly.graph_objects as _go

        # ---- Scatter: A_C vs A_G, colour = A_RM ----
        _fig_sc = px.scatter(
            _ca, x="A_C", y="A_G", color="A_RM",
            color_continuous_scale="Viridis", range_color=[0, 1],
            hover_name="name",
            hover_data={"A_C": ":.0%", "A_G": ":.0%", "A_RM": ":.0%"},
            labels={"A_C": "A_C — supply adequacy", "A_G": "A_G — geographic reach",
                    "A_RM": "A_RM"},
        )
        _fig_sc.update_traces(marker=dict(size=9, line=dict(width=0.5, color="rgba(0,0,0,0.4)")))
        _fig_sc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color="grey", dash="dash", width=1))
        _fig_sc.add_annotation(x=0.97, y=0.03, text="supply-limited", showarrow=False,
                               font=dict(size=11, color="#888"), xanchor="right")
        _fig_sc.add_annotation(x=0.03, y=0.97, text="access-limited", showarrow=False,
                               font=dict(size=11, color="#888"), xanchor="left")
        _fig_sc.update_layout(
            title=f"Where is the bottleneck? (within {_ca_thr})",
            xaxis=dict(range=[-0.02, 1.03], tickformat=".0%"),
            yaxis=dict(range=[-0.02, 1.03], tickformat=".0%"),
            coloraxis_colorbar=dict(title="A_RM", tickformat=".0%"),
            height=560, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(_fig_sc, use_container_width=True)

        # ---- Bar chart: A_RM bars with A_C / A_G ceilings ----
        st.subheader("Demand, ceilings & realised treatment")
        _bc1, _bc2 = st.columns([2, 1])
        _worst_first = _bc2.toggle("Worst access first", value=True, key="ca_worst")
        _ca_n = _bc1.slider(
            "Countries to show", min_value=10, max_value=len(_ca),
            value=min(25, len(_ca)), step=5, key="ca_n",
            help="Ranked by realised access (A_RM). Slide to the maximum to see every country.",
        )
        _cab = _ca.sort_values("A_RM", ascending=_worst_first).head(_ca_n)
        _C_REAL, _C_SUP, _C_ACC = "#2c7fb8", "#d95f02", "#1b9e77"

        _fig_bar = _go.Figure()
        _fig_bar.add_trace(_go.Bar(
            y=_cab["name"], x=_cab["A_RM"], orientation="h",
            marker_color=_C_REAL, name="A_RM realised",
            hovertemplate="<b>%{y}</b><br>A_RM realised: %{x:.0%}<extra></extra>",
        ))
        _fig_bar.add_trace(_go.Scatter(
            y=_cab["name"], x=_cab["A_C"], mode="markers", name="A_C supply ceiling",
            marker=dict(symbol="line-ns", color=_C_SUP, size=11,
                        line=dict(width=2.5, color=_C_SUP)),
            hovertemplate="<b>%{y}</b><br>A_C supply ceiling: %{x:.0%}<extra></extra>",
        ))
        _fig_bar.add_trace(_go.Scatter(
            y=_cab["name"], x=_cab["A_G"], mode="markers", name="A_G access ceiling",
            marker=dict(symbol="line-ns", color=_C_ACC, size=11,
                        line=dict(width=2.5, color=_C_ACC)),
            hovertemplate="<b>%{y}</b><br>A_G access ceiling: %{x:.0%}<extra></extra>",
        ))
        _fig_bar.update_layout(
            barmode="overlay",
            xaxis=dict(title="fraction of national RT demand", range=[0, 1.02], tickformat=".0%"),
            yaxis=dict(autorange="reversed", title=None, automargin=True, tickfont=dict(size=11)),
            height=max(400, len(_cab) * 24),
            margin=dict(r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1),
        )
        st.plotly_chart(_fig_bar, use_container_width=True)
        st.caption(
            f"Showing {len(_cab)} of {len(_ca)} countries "
            f"({'lowest' if _worst_first else 'highest'} access first). "
            "Bars = realised treatment $A_{RM}$; ticks = the supply ($A_C$, orange) and "
            "access ($A_G$, green) ceilings. A bar short of its nearest tick is spatial "
            "mismatch. $A_C$ is threshold-free, so its ticks don't move with the selector."
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "RadMaps · Released under the [MIT License](https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access/blob/main/LICENSE) · "
    "[GitHub](https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access)"
)
