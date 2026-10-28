"""
Map drawing for the simplified dashboard (app.py): the colour ramps, colour
bar, hexagon, border, no-data and linac-tower layers of the full dashboard
(app_full.py), copied unchanged so both apps draw the same maps.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import h3
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import pycountry
import pydeck as pdk
import streamlit as st

from data.cancer import has_globocan_data

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


_LINAC_BLUE = [30, 120, 220, 220]


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


_CARTO_LIGHT_LABELS   = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


_CARTO_LIGHT_NOLABELS = "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json"



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


COLORMAPS = {
    "Purple → Yellow (Viridis)": _viridis_rgb,
    "Red → Green": _rdylgn_rgb,
    "Green → Red": _rdylgn_reversed_rgb,
    "Blue → Red (CB-safe)": _blueorange_rgb,
    "Red → Blue (CB-safe)": _blueorange_reversed_rgb,
    "Cividis (CB-safe)": _cividis_rgb,
}

