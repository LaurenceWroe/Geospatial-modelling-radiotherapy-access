"""
Radiotherapy Access — Interactive H3 Map (Streamlit)

Run with:
    streamlit run app.py

Map types
---------
Population Density        — Kontur H3 population per hexagon (log scale)
Cancer Incidence          — Estimated cases per hexagon (proportional to pop)
Optimal RT Treatment      — Cases that should receive RT (optimal fractions)
Actual RT Treatment       — Cases that are receiving RT (actual fractions)
Radiotherapy Access       — P(patient can access a LINAC) per hexagon
Nearest LINAC Distance    — Distance (km) to the closest LINAC facility
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
import numpy as np
import pandas as pd
import pydeck as pdk
import pycountry
import streamlit as st

from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
from data.cancer import get_cancer_types, apportion_cancer_to_h3, has_globocan_data
from analysis.accessibility import compute_accessibility


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Radiotherapy Access Maps",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
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

_DEFAULT_CMAP = {
    "Population Density": "Purple → Yellow (Viridis)",
    "Cancer Incidence": "Purple → Yellow (Viridis)",
    "Cancer cases requiring RT": "Purple → Yellow (Viridis)",
    "Radiotherapy Access": "Green → Red",
    "Nearest LINAC Distance": "Green → Red",
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
    cbar.set_label(label, fontsize=8, color=text_color)
    cbar.ax.tick_params(labelsize=7, labelcolor=text_color, color=text_color)
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
def _load_pop(country: str, h3_res: int = 8):
    return load_population_at_resolution(country, target_resolution=h3_res)


@st.cache_data(show_spinner=False)
def _load_cancer(country: str, iso3: str, cancers: tuple, use_actual: bool, h3_res: int = 8):
    gdf = _load_pop(country, h3_res)
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
):
    gdf = _load_pop(country, h3_res)

    # Build RT demand per hex from cancer data (all cancer types)
    demand = None
    try:
        all_cancers = get_cancer_types()
        cancer_gdf = apportion_cancer_to_h3(gdf, iso3, all_cancers, use_actual_rt=False)
        if rt_method == "optimal":
            cols = [c for c in cancer_gdf.columns if c.endswith("_optimal_rt")]
        else:  # proportional
            cols = [c for c in cancer_gdf.columns if c.endswith("_incidence")]
        if cols:
            demand = cancer_gdf[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)
            if rt_method == "proportional":
                demand = demand * rt_fraction
    except Exception:
        pass  # fallback: compute_accessibility uses raw population

    gdf_out, stats = compute_accessibility(
        gdf,
        list(linac_locs),
        lambda_km=lambda_km,
        model=model,
        max_distance_km=max_distance_km,
        capacity_per_machine_per_year=capacity_per_machine_per_year,
        demand=demand,
    )
    return gdf_out, stats


@st.cache_data(show_spinner=False)
def _load_dirac(country: str):
    try:
        return load_linacs_from_dirac_db(country)
    except (ValueError, FileNotFoundError) as e:
        return None, str(e)


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
    style: str = "stacked",
) -> List[pdk.Layer]:
    """Return ColumnLayers for LINAC towers.

    style="stacked"    — co-located facilities (same H3 cell) merged into one
                         tower with proportional segments per centre.
    style="individual" — one column per facility at its own lat/lon.
    """
    hex_area_km2 = h3.average_hexagon_area(h3_res, unit="km^2")
    hex_radius_km = math.sqrt(hex_area_km2 / math.pi)
    col_radius_m = int(hex_radius_km * 1000 * 0.45)
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
    parts = [f"Initial view width ≈ {width_km:,.0f} km"]
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


def _render_with_colorbar(
    layers,
    view: pdk.ViewState,
    cmap_fn,
    vmin: float,
    vmax: float,
    cb_label: str,
    log_scale: bool = False,
    dark: bool = False,
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
        fig = _colorbar_fig(cmap_fn, vmin, vmax, cb_label, log_scale=log_scale, text_color="white")
        st.pyplot(fig, use_container_width=True)


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
    "Cancer cases requiring RT",
    "Radiotherapy Access",
    "Nearest LINAC Distance",
]

with st.sidebar:
    st.title("🏥 Radiotherapy Access")

    country = st.selectbox(
        "Country",
        options=sorted(c.name for c in pycountry.countries),
        index=sorted(c.name for c in pycountry.countries).index("United Kingdom"),
    )

    map_type = st.selectbox("Map type", MAP_TYPES)

    h3_resolution = st.selectbox(
        "H3 resolution",
        options=[8, 7, 6, 5, 4, 3],
        index=2,  # H6 default
        format_func=lambda r: {
            8: "H8 (~0.7 km²)", 7: "H7 (~5 km²)", 6: "H6 (~36 km²)",
            5: "H5 (~253 km²)", 4: "H4 (~1,770 km²)", 3: "H3 (~12,400 km²)",
        }[r],
    )

    is_rt_demand_map = map_type == "Cancer cases requiring RT"
    is_cancer = map_type in ("Cancer Incidence", "Cancer cases requiring RT")
    is_access = map_type == "Radiotherapy Access"
    is_nearest = map_type == "Nearest LINAC Distance"
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
        default_cancers = [
            c for c in all_cancers
            if "non-melanoma" not in c.lower() and "nmsc" not in c.lower()
        ]
        selected_cancers = st.multiselect(
            "Cancer types",
            options=all_cancers,
            default=default_cancers,
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
                    min_value=0.05, max_value=1.0, value=0.25, step=0.05,
                    format="%.2f",
                )

    # ------ Access model controls (outside expander — affect computation) -----
    lambda_km: float = 30.0
    max_distance_km: float = 100.0
    access_model: str = "exponential"
    access_display_metric: str = "Capacity-limited population untreated"
    capacity_per_machine_per_year: float = 450.0

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
                min_value=0.05, max_value=1.0, value=0.25, step=0.05,
                format="%.2f",
                key="access_rt_fraction",
            )

        access_display_metric = st.selectbox(
            "Display metric",
            [
                "Capacity-limited population untreated",
                "Probability based on distance",
                "Capacity-limited probability",
                "Capacity-limited population treated",
            ],
            index=0,
        )
        model_label = st.radio(
            "Access model",
            ["Exponential decay", "Step function", "Uniform (no decay)"],
            horizontal=True,
        )
        access_model = {
            "Exponential decay": "exponential",
            "Step function": "step",
            "Uniform (no decay)": "uniform",
        }[model_label]

        if access_model == "exponential":
            lambda_km = float(st.slider("Distance decay λ (km)", 5, 200, 30, step=5))
        elif access_model == "step":
            max_distance_km = float(
                st.slider("Max treatment distance (km)", 10, 500, 100, step=10)
            )

    # ------ Plot settings (expander) ---------------------------------------
    show_linac_markers: bool = False
    tower_height_scale: float = 1.0
    linac_tower_style: str = "stacked"
    _default_log = map_type in ("Population Density", "Cancer Incidence")

    with st.expander("⚙️ Plot settings"):
        dark_mode = st.checkbox("Dark map", value=False)

        if needs_linac:
            st.divider()
            show_linac_markers = st.checkbox("Show LINAC locations", value=True)
            if show_linac_markers:
                tower_height_scale = float(
                    st.slider("Tower height scale", min_value=0.05, max_value=5.0, value=1.0, step=0.05)
                )
                _tower_style_label = st.radio(
                    "Tower style",
                    ["Stacked (segmented per centre)", "Individual (one tower per centre)"],
                    horizontal=False,
                )
                linac_tower_style = "stacked" if "Stacked" in _tower_style_label else "individual"

        st.divider()
        _default_cmap_name = _DEFAULT_CMAP.get(map_type, "Purple → Yellow (Viridis)")
        cb_cmap_name = st.selectbox(
            "Colour scheme",
            options=list(COLORMAPS.keys()),
            index=list(COLORMAPS.keys()).index(_default_cmap_name),
        )
        cb_cmap_fn = COLORMAPS[cb_cmap_name]
        cb_log = st.checkbox("Log scale", value=_default_log)
        cb_auto = st.checkbox("Auto range", value=True)
        cb_vmin_user: Optional[float] = None
        cb_vmax_user: Optional[float] = None
        if not cb_auto:
            cb_vmin_user = st.number_input("Min value", value=0.0, format="%.4g")
            cb_vmax_user = st.number_input("Max value", value=1.0, format="%.4g")
    generate = st.button("Generate Map", type="primary", use_container_width=True)


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

st.header(f"{map_type} — {country}")

if not generate:
    st.info("Configure options in the sidebar and click **Generate Map**.")
    st.stop()

try:
    iso3 = pycountry.countries.lookup(country).alpha_3
except LookupError:
    st.error(f"Could not resolve country: {country!r}")
    st.stop()

# Load LINAC data (needed for access and nearest-distance maps)
locs: Optional[List[Tuple[float, float, float]]] = None
facilities_df: Optional[pd.DataFrame] = None
if needs_linac:
    with st.spinner("Loading LINAC data from DIRAC database…"):
        result = _load_dirac(country)
    if result[0] is None:
        st.error(f"LINAC data unavailable: {result[1]}")
        st.stop()
    locs, facilities_df = result


# ---------------------------------------------------------------------------
# Population Density
# ---------------------------------------------------------------------------

if map_type == "Population Density":
    with st.spinner("Loading population data…"):
        gdf = _load_pop(country, h3_resolution)

    pop = gdf["population"].to_numpy(dtype=np.float64)
    auto_vmin = float(max(pop.min(), 1))
    auto_vmax = float(pop.max())
    colors, vmin, vmax = _color_values(pop, cb_cmap_fn, auto_vmin, auto_vmax)

    gdf = gdf.copy()
    gdf["color"] = colors
    gdf["tip"] = (
        "<b>" + gdf["h3"].astype(str) + "</b><br/>"
        "People per hexagon: " + gdf["population"].round(0).astype(int).astype(str)
    )

    df = pd.DataFrame({"h3": gdf["h3"], "color": gdf["color"], "tip": gdf["tip"]})
    _render_with_colorbar(
        [_build_hex_layer(df)],
        _make_view(gdf),
        cb_cmap_fn, vmin, vmax, "People per hexagon", log_scale=cb_log, dark=dark_mode,
    )
    st.caption(_h3_caption(gdf) + " · " + _scale_caption(gdf))
    col1, col2 = st.columns(2)
    col1.metric("Total population", f"{int(gdf['population'].sum()):,}")
    col2.metric("H3 hexagons", f"{len(gdf):,}")


# ---------------------------------------------------------------------------
# Cancer maps
# ---------------------------------------------------------------------------

elif is_cancer:
    if not selected_cancers:
        st.warning("Please select at least one cancer type.")
        st.stop()

    if not has_globocan_data(iso3):
        st.warning(
            f"**{country}** (ISO3: {iso3}) is not present in this GLOBOCAN dataset. "
            "Cancer case counts will be zero. The population map is still available."
        )

    with st.spinner("Apportioning cancer incidence to H3 grid…"):
        gdf = _load_cancer(country, iso3, tuple(selected_cancers), use_actual, h3_resolution)

    if map_type == "Cancer cases requiring RT":
        suffix = "_optimal_rt" if rt_method == "optimal" else "_incidence"
    else:
        suffix = "_incidence"

    cols_of_interest = [c + suffix for c in selected_cancers if (c + suffix) in gdf.columns]
    if not cols_of_interest:
        st.error("No matching columns found in data.")
        st.stop()

    combined = gdf[cols_of_interest].sum(axis=1).to_numpy(dtype=np.float64)

    # Scale for proportional RT method
    if map_type == "Cancer cases requiring RT" and rt_method == "proportional":
        combined = combined * rt_fraction

    auto_vmin = float(max(combined.min(), 0.001))
    auto_vmax = float(combined.max())
    colors, vmin, vmax = _color_values(combined, cb_cmap_fn, auto_vmin, auto_vmax)

    gdf = gdf.copy()
    gdf["color"] = colors

    if map_type == "Cancer cases requiring RT":
        method_str = "Optimal RT" if rt_method == "optimal" else f"Proportional RT ({rt_fraction:.0%})"
        label = f"Cancer cases requiring RT ({method_str})"
    else:
        label = "Cancer incidence"

    s_combined = pd.Series(combined, index=gdf.index).round(2).astype(str)
    gdf["tip"] = "<b>" + gdf["h3"].astype(str) + "</b><br/>" + label + ": " + s_combined

    df = pd.DataFrame({"h3": gdf["h3"], "color": gdf["color"], "tip": gdf["tip"]})
    _render_with_colorbar(
        [_build_hex_layer(df)],
        _make_view(gdf),
        cb_cmap_fn, vmin, vmax, label, log_scale=cb_log, dark=dark_mode,
    )
    st.caption(_h3_caption(gdf) + " · " + _scale_caption(gdf))
    if map_type == "Cancer cases requiring RT":
        incidence_cols = [c + "_incidence" for c in selected_cancers if (c + "_incidence") in gdf.columns]
        total_incidence = float(gdf[incidence_cols].sum(axis=1).sum()) if incidence_cols else 0.0
        col1, col2, col3 = st.columns(3)
        col1.metric("Total cancer cases requiring RT", f"{combined.sum():,.0f}")
        col2.metric("Total cancer cases", f"{total_incidence:,.0f}")
        col3.metric("H3 hexagons", f"{len(gdf):,}")
    else:
        total_pop = float(gdf["population"].sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("Total cancer cases", f"{combined.sum():,.0f}")
        col2.metric("Country population", f"{int(total_pop):,}")
        col3.metric("H3 hexagons", f"{len(gdf):,}")


# ---------------------------------------------------------------------------
# Radiotherapy Access / Nearest LINAC Distance
# ---------------------------------------------------------------------------

elif is_access or is_nearest:
    linac_locs_tuple = tuple(locs)

    with st.spinner("Computing accessibility…"):
        gdf_out, stats = _compute_access(
            country,
            iso3,
            linac_locs_tuple,
            float(lambda_km),
            access_model,
            float(max_distance_km),
            capacity_per_machine_per_year,
            access_rt_method,
            access_rt_fraction,
            h3_resolution,
        )

    pitch = 30.0 if show_linac_markers else 0.0

    # Country span for scaling LINAC column heights
    _geom = gdf_out.geometry
    _lat_span = float(_geom.bounds["maxy"].max() - _geom.bounds["miny"].min())
    _lon_span = float(_geom.bounds["maxx"].max() - _geom.bounds["minx"].min())
    _lat_mid = float((_geom.bounds["maxy"].max() + _geom.bounds["miny"].min()) / 2)
    country_span_km = max(
        _lat_span * 111.32,
        _lon_span * 111.32 * math.cos(math.radians(_lat_mid)),
    )

    # ---- Nearest LINAC Distance ------------------------------------------
    if is_nearest:
        dist_km = gdf_out["nearest_linac_km"].to_numpy(dtype=np.float64)
        valid = np.isfinite(dist_km)
        auto_vmin = 0.0
        auto_vmax = float(np.nanpercentile(dist_km[valid], 95)) if valid.any() else 500.0
        colors, vmin, vmax = _color_values(dist_km, cb_cmap_fn, auto_vmin, auto_vmax)

        gdf_out = gdf_out.copy()
        gdf_out["color"] = colors
        gdf_out["tip"] = (
            "<b>" + gdf_out["h3"].astype(str) + "</b><br/>"
            "Nearest LINAC: " + dist_km.round(1).astype(str) + " km"
        )

        layers = [_build_hex_layer(
            pd.DataFrame({"h3": gdf_out["h3"], "color": gdf_out["color"], "tip": gdf_out["tip"]})
        )]
        if show_linac_markers:
            layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, style=linac_tower_style))
        pass  # scale info shown in caption

        _render_with_colorbar(
            layers, _make_view(gdf_out, pitch=pitch),
            cb_cmap_fn, vmin, vmax, "Distance (km)",
            log_scale=cb_log, dark=dark_mode,
        )
        st.caption(_h3_caption(gdf_out) + " · " + _scale_caption(gdf_out))
        col1, col2, col3 = st.columns(3)
        col1.metric("Total LINACs", stats["total_machines"])
        col2.metric("Median distance", f"{float(np.nanmedian(dist_km)):.1f} km")
        col3.metric("95th-pct distance", f"{auto_vmax:.1f} km")

    # ---- Radiotherapy Access ----------------------------------------
    else:
        prob = gdf_out["access_probability"].to_numpy(dtype=np.float64)
        cap_prob = gdf_out["capacity_limited_probability"].to_numpy(dtype=np.float64)
        raw_pop = gdf_out["population"].to_numpy(dtype=np.float64)

        # Build tip strings using pandas Series to avoid numpy dtype mismatches
        s_h3 = gdf_out["h3"].astype(str)
        s_prob = (gdf_out["access_probability"] * 100).round(1).astype(str)
        s_cap = (gdf_out["capacity_limited_probability"] * 100).round(1).astype(str)
        s_pop = gdf_out["population"].round(0).astype(int).astype(str)

        if access_display_metric == "Capacity-limited population untreated":
            display_vals = gdf_out["rt_untreated"].to_numpy(dtype=np.float64)
            cb_label_access = "RT patients untreated per year (per hex)"
            auto_vmin_a = 0.0
            auto_vmax_a = float(np.nanmax(display_vals))
            s_vals = pd.Series(display_vals, index=gdf_out.index).round(1).astype(str)
            tip_series = (
                "<b>" + s_h3 + "</b><br/>"
                + "RT patients untreated/yr: " + s_vals + "<br/>"
                + "Cap-limited probability: " + s_cap + "%"
            )
            metric_cmap_fn = _rdylgn_reversed_rgb  # green→red: 0=green, many=red

        elif access_display_metric == "Probability based on distance":
            display_vals = prob
            cb_label_access = "Access probability (distance-based)"
            auto_vmin_a, auto_vmax_a = 0.0, 1.0
            tip_series = (
                "<b>" + s_h3 + "</b><br/>"
                + "Access probability: " + s_prob + "%<br/>"
                + "People per hex: " + s_pop
            )
            metric_cmap_fn = _rdylgn_reversed_rgb  # green→red

        elif access_display_metric == "Capacity-limited probability":
            display_vals = cap_prob
            cb_label_access = "Capacity-limited probability"
            auto_vmin_a, auto_vmax_a = 0.0, 1.0
            tip_series = (
                "<b>" + s_h3 + "</b><br/>"
                + "Cap-limited probability: " + s_cap + "%<br/>"
                + "Distance-based probability: " + s_prob + "%"
            )
            metric_cmap_fn = _rdylgn_reversed_rgb  # green→red

        else:  # Capacity-limited population treated
            display_vals = gdf_out["rt_treated"].to_numpy(dtype=np.float64)
            cb_label_access = "RT patients treated per year (per hex)"
            auto_vmin_a = 0.0
            auto_vmax_a = float(np.nanmax(display_vals))
            s_vals = pd.Series(display_vals, index=gdf_out.index).round(1).astype(str)
            tip_series = (
                "<b>" + s_h3 + "</b><br/>"
                + "RT patients treated/yr: " + s_vals + "<br/>"
                + "Cap-limited probability: " + s_cap + "%"
            )
            metric_cmap_fn = _rdylgn_rgb  # red→green: 0=red, many treated=green

        # Use the metric's natural colormap unless the user has manually picked one
        # (detected by the sidebar selection differing from the map-type default)
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
            layers.extend(_build_linac_columns(facilities_df, h3_res=h3_resolution, country_span_km=country_span_km, height_scale=tower_height_scale, style=linac_tower_style))
        pass  # scale info shown in caption

        if stats["total_rt_demand"] == 0:
            st.warning(
                f"RT demand is zero for **{country}** — this country may not be in the GLOBOCAN dataset. "
                "Capacity allocation cannot be computed; geographic access probability is still valid."
            )

        _render_with_colorbar(
            layers, _make_view(gdf_out, pitch=pitch),
            active_cmap_fn, vmin, vmax, cb_label_access,
            log_scale=cb_log, dark=dark_mode,
        )
        st.caption(_h3_caption(gdf_out) + " · " + _scale_caption(gdf_out))

        if access_model == "exponential":
            model_info = f"Exponential | λ = {lambda_km} km | cut-off = {stats['cutoff_km']:.0f} km"
        elif access_model == "step":
            model_info = f"Step function | max distance = {max_distance_km:.0f} km"
        else:
            model_info = "Uniform (no distance decay)"

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total LINACs", stats["total_machines"])
        col2.metric("Total RT Capacity", f"{int(stats['total_national_capacity']):,}")
        col3.metric("Estimated RT Need", f"{int(stats['total_rt_demand']):,}")
        col4.metric("Total Patients Treated", f"{int(stats['total_rt_treated']):,}")
        col5.metric("Mean geographic access", f"{stats['mean_access_probability']:.1%}")
        col6.metric("Mean capacity-limited access", f"{stats['mean_capacity_limited_probability']:.1%}")
        _demand_info = (
            f"demand: optimal RT utilisations"
            if access_rt_method == "optimal"
            else f"demand: proportional RT ({access_rt_fraction:.0%} of cancer cases)"
        )
        st.caption(
            f"{model_info} | {int(capacity_per_machine_per_year)} pts/machine/yr | "
            f"{_demand_info} | {stats['n_hexagons']:,} hexagons"
        )
