"""
RadMaps — Radiotherapy access dashboard (Streamlit)

Run with:
    streamlit run app.py

Capped-3SFCA radiotherapy access in the main configuration of the RadMaps
paper (H3 resolution 5, stored local-08:00 driving times, 2012 CCORE optimal
RTU), with a few configurable inputs: driving-time limit, capacity per linac,
distance-decay model and the number of linacs.

Results come, in order of preference, from
  cache/dashboard_world*.parquet    World and regions (in the repository)
  dashboard_store/{ISO3}.parquet    single countries at resolution 5, downloaded
                                    on first use from the GitHub release below
  compute.compute_members           computed live: edited linac counts, or a
                                    country with no stored file
All three are produced by the same code (scripts/build_dashboard_world.py).

The earlier full research dashboard (other allocation rules, FCA benchmarks,
sensitivity and convergence tabs) is on the full-dashboard branch.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import h3
import numpy as np
import pandas as pd
import pycountry
import plotly.graph_objects as go
import pydeck as pdk
from plotly.subplots import make_subplots
import streamlit as st

from compute import (_country_options, _load_dirac, _region_options, access_profile, centre_loads, compute_members,
                     minutes_to_proxy_km, plan_centres)
from data.cancer import DEFAULT_RTU_SET, DERIVED_CANCER_TYPES, get_cancer_types, get_optimal_rt_fractions
from data.regions import get_region
from map_view import (
    COLORMAPS, _CARTO_LIGHT_NOLABELS, _LINAC_BLUE, _apply_colormap_fixed, _build_borders_layer, _build_hex_layer,
    _build_linac_columns, _build_no_data_fill_layer, _colorbar_fig, _fit_zoom, _rdylgn_reversed_rgb,
    _rdylgn_rgb, _render_colorbar_fig,
)

_ROOT = Path(__file__).resolve().parent
_WORLD_HEXES = _ROOT / "cache" / "dashboard_world.parquet"
_WORLD_COUNTRIES = _ROOT / "cache" / "dashboard_world_countries.parquet"
_STORE_DIR = _ROOT / "dashboard_store"
_STORE_URL = ("https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access/"
              "releases/download/dashboard-store-v1/{iso3}.parquet")

# Configurable inputs: the grid precomputed by scripts/build_dashboard_world.py
TIMES = [30.0, 60.0, 90.0, 120.0]
CAPACITIES = [250.0, 350.0, 450.0, 550.0]
MODELS = {
    "gaussian": "Gaussian",
    "step": "Step",
    "exponential": "Exponential (λ = 60 min)",
    "power": "Power law (β = 1, plateau 10 min)",
}
RTU = DEFAULT_RTU_SET  # 2012 CCORE optimal RTU, as in the paper
LAYERS = ["Access ratio", "Deficit", "Radiotherapy demand"]
SEPARATOR = "──────────"
MAP_HEIGHT = 560

_LOGO = _ROOT / "assets" / "logo" / "radmaps_logo.png"
_FAVICON = _ROOT / "assets" / "logo" / "radmaps_favicon.png"
st.set_page_config(page_title="RadMaps v1.1", page_icon=str(_FAVICON) if _FAVICON.exists() else "🏥", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _world_tables():
    """Precomputed World results (hexes at resolution 3, countries), or None."""
    if not (_WORLD_HEXES.exists() and _WORLD_COUNTRIES.exists()):
        return None
    return pd.read_parquet(_WORLD_HEXES), pd.read_parquet(_WORLD_COUNTRIES)


@st.cache_data(show_spinner=False)
def _global_max_demand(rtu: str, res: int) -> float:
    """Largest hexagon demand worldwide at *res* (scripts/build_dashboard_max_demand.py);
    falls back to the World cache, which holds resolution 3 and coarser."""
    path = _ROOT / "cache" / "dashboard_global_max_demand.json"
    if path.exists():
        return float(json.loads(path.read_text())[rtu][str(res)])
    world = _world_tables()
    hx = world[0][world[0]["rtu"] == rtu].drop_duplicates(["h3", "country"])
    return float(_at_resolution(hx, min(res, 3))["D"].max())


ALLOCATIONS = {"proportional": "Capped 3SFCA", "nearest_first": "Nearest-first"}
_A_PREFIX = {"proportional": "A", "nearest_first": "A_nf"}  # store column prefix per allocation

# The main configuration of the RadMaps paper
PAPER = {"access": "driving", "t_max": 120.0, "capacity": 450.0, "model": "gaussian", "alloc": "proportional",
         "rtu_mode": "Optimal (CCORE 2012)"}

# Access profiles of the paper (Section 3.2), at a nominal 80 % threshold
PROFILES = {"Strong access": [30, 120, 30], "Capacity constrained": [42, 120, 214],
            "Geography constrained": [230, 120, 30], "Doubly constrained": [220, 50, 50]}


_profile = access_profile


def _chip(name: str) -> str:
    r, g_, b = PROFILES[name]
    return (f"<span style='background:rgb({r},{g_},{b});color:white;padding:2px 8px;border-radius:10px;"
            f"font-size:0.9rem;white-space:nowrap;display:inline-block;margin:2px 0'>{name}</span>")


def _select(df: pd.DataFrame, alloc, rtu, model, t_max, capacity) -> pd.DataFrame:
    """Rows of a World table for one configuration (tables without an allocation
    column hold capped 3SFCA only)."""
    a = df["allocation"] == alloc if "allocation" in df else pd.Series(alloc == "proportional", index=df.index)
    return df[a & (df["rtu"] == rtu) & (df["model"] == model) & (df["t_max"] == t_max) & (df["capacity"] == capacity)]


@st.cache_data(show_spinner=False, max_entries=8)
def _country_store(iso3: str) -> Optional[pd.DataFrame]:
    """A country's stored resolution-5 results, downloaded on first use; None if unavailable."""
    path = _STORE_DIR / f"{iso3}.parquet"
    if not path.exists():
        try:
            with urllib.request.urlopen(_STORE_URL.format(iso3=iso3), timeout=60) as r, \
                    tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
                shutil.copyfileobj(r, tmp)
            _STORE_DIR.mkdir(exist_ok=True)
            shutil.move(tmp.name, path)
        except Exception:  # offline, no release yet, or no file for this country
            return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, max_entries=16)
def _computed(alpha2_list: tuple, alloc, rtu, model, t_max, capacity, linacs: tuple, custom_rtu: tuple = (),
              access: str = "driving"):
    return compute_members(list(alpha2_list), model=model, t_max=t_max, capacity=capacity, rtu_set=rtu,
                           linacs=dict(linacs), custom_rtu=custom_rtu, allocation=alloc, access=access)


def _country_result(alpha2: str, alloc, rtu, model, t_max, capacity, linacs: dict, allow_compute: bool = False,
                    custom_rtu: tuple = (), access: str = "driving"):
    """(hexes at resolution 5, one-row country table) for one country, or None
    when no stored result exists and *allow_compute* is False."""
    obj = pycountry.countries.get(alpha_2=alpha2)
    store = None if (alpha2 in linacs or custom_rtu or access != "driving") else _country_store(obj.alpha_3)
    world = _world_tables()
    a_col = f"{_A_PREFIX[alloc]}|{rtu}|{model}|{t_max:.0f}|{capacity:.0f}"
    if store is not None and a_col in store.columns and world is not None:
        hexes = pd.DataFrame({
            "h3": store["h3"], "country": obj.name, "population": store["population"],
            "D": store[f"D|{rtu}"], "A": store[a_col], "A_G": store[f"A_G|{rtu}|{model}|{t_max:.0f}"],
        })
        row = _select(world[1], alloc, rtu, model, t_max, capacity)
        row = row[row["iso3"] == obj.alpha_3]
        if not row.empty:
            return hexes, row
    if not allow_compute:
        return None
    edits = ((alpha2, linacs[alpha2]),) if alpha2 in linacs else ()
    return _computed((alpha2,), alloc, rtu, model, t_max, capacity, edits, custom_rtu, access)


def _region_result(region: str, alloc, rtu, model, t_max, capacity, linacs: dict, allow_compute: bool = False,
                   custom_rtu: tuple = (), access: str = "driving"):
    """(hexes, country table) for a region, from the World cache except for
    countries with edited linacs, which are computed and summed to resolution 3.
    None when that needs a computation and *allow_compute* is False."""
    members = [a for a in get_region(region).member_alpha2 if pycountry.countries.get(alpha_2=a)]
    world = _world_tables()
    if world is not None and (access != "driving" or _select(world[1], alloc, rtu, model, t_max, capacity).empty):
        world = None  # not pre-built for these inputs
    if (world is None or custom_rtu or any(a in linacs for a in members)) and not allow_compute:
        return None
    if world is None or custom_rtu:
        bar = st.progress(0.0, text="Computing each country")
        out = compute_members(members, model=model, t_max=t_max, capacity=capacity, rtu_set=rtu, linacs=linacs,
                              custom_rtu=custom_rtu, allocation=alloc, access=access, progress_callback=lambda i, n: bar.progress(i / n, text=f"Computing: {i} / {n}"))
        bar.empty()
        return out
    edited = tuple(sorted(a for a in members if a in linacs))
    names = {pycountry.countries.get(alpha_2=a).name for a in members if a not in linacs}
    hx, ct = (_select(d, alloc, rtu, model, t_max, capacity) for d in world)
    hx, ct = hx[hx["country"].isin(names)], ct[ct["country"].isin(names)]
    if edited:
        e_hx, e_ct = _computed(edited, alloc, rtu, model, t_max, capacity, tuple((a, linacs[a]) for a in edited),
                               (), access)
        hx = pd.concat([hx, e_hx.assign(h3=[h3.cell_to_parent(c, 3) for c in e_hx["h3"]])], ignore_index=True)
        ct = pd.concat([ct, e_ct], ignore_index=True)
    return hx, ct


def _at_resolution(hexes: pd.DataFrame, res: int) -> pd.DataFrame:
    """Sum population, D, A and A_G into the parent hexagons at *res*; the
    tooltip names the country holding most of each hexagon's demand."""
    df = hexes.copy()
    if h3.get_resolution(df["h3"].iloc[0]) > res:
        df["h3"] = [h3.cell_to_parent(c, res) for c in df["h3"]]
    lead = df.sort_values("D").groupby("h3")["country"].last()
    out = df.groupby("h3", as_index=False)[["population", "D", "A", "A_G"]].sum()
    out["country"] = out["h3"].map(lead)
    return out


# ---------------------------------------------------------------------------
# Map (as in the full dashboard)
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    v = float(v)
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f} M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f} k"
    return f"{v:.1f}"


def _view(lats, lons, pitch: float) -> pdk.ViewState:
    lat0, lat1, lon0, lon1 = float(np.min(lats)), float(np.max(lats)), float(np.min(lons)), float(np.max(lons))
    zoom = _fit_zoom(lon1 - lon0, lat0, lat1, MAP_HEIGHT * 1.5, MAP_HEIGHT)
    return pdk.ViewState(latitude=(lat0 + lat1) / 2, longitude=(lon0 + lon1) / 2,
                         zoom=max(0.2, min(12.0, zoom)), max_zoom=13, pitch=pitch, bearing=0)


# Streamlit HTML-escapes the values substituted into a pydeck tooltip (since 1.6x), so the markup lives
# in the template and each picked object carries plain text: a bold title and a body split by newlines.
_TOOLTIP = {"html": "<b>{tip_title}</b><br/>{tip_body}", "style": {"whiteSpace": "pre-line"}}


def _tip_parts(tip) -> tuple[str, str]:
    """Plain title and body of an HTML tooltip '<b>title</b><br/>line<br/>line...'."""
    s = str(tip)
    m = re.match(r"\s*<b>(.*?)</b>(?:<br\s*/?>)?(.*)", s, re.S)
    title, body = (m.group(1), m.group(2)) if m else (s, "")
    body = re.sub(r"<br\s*/?>|<hr[^>]*>", "\n", body)
    return re.sub(r"<[^>]+>", "", title), re.sub(r"<[^>]+>", "", body).strip("\n")


def _plain_tips(layers: list) -> list:
    """Replace each object's (or GeoJSON feature's) HTML 'tip' by plain tip_title / tip_body."""
    for layer in layers:
        data = getattr(layer, "data", None)
        rows = ([f.setdefault("properties", {}) for f in data.get("features", [])] if isinstance(data, dict)
                else data if isinstance(data, list) else [])
        for r in rows:
            if isinstance(r, dict) and "tip" in r:
                r["tip_title"], r["tip_body"] = _tip_parts(r.pop("tip"))
    return layers


def _render(layers: list, view: pdk.ViewState, cmap_fn, vmin: float, vmax: float, label: str, region: bool,
            log_scale: bool = False) -> None:
    """Map with the colour bar to its right, the full dashboard's layout."""
    if region:
        nd = _build_no_data_fill_layer()
        bl = _build_borders_layer(False)
        layers = ([nd] if nd is not None else []) + layers + ([bl] if bl is not None else [])
    deck = pdk.Deck(layers=_plain_tips(layers), initial_view_state=view, map_style=_CARTO_LIGHT_NOLABELS,
                    tooltip=_TOOLTIP)
    col_map, col_cb = st.columns([7, 1])
    with col_map:
        st.pydeck_chart(deck, width="stretch", height=MAP_HEIGHT, key=f"map_{location}_{res}")
    with col_cb:
        st.markdown("<p style='font-size:12px;font-family:monospace;color:#888;margin:-7px;"
                    "text-align:center;'>© RadMaps 2026</p>", unsafe_allow_html=True)
        _render_colorbar_fig(_colorbar_fig(cmap_fn, vmin, vmax, label, log_scale=log_scale,
                                           height_px=MAP_HEIGHT - 18))


def _dirac_rows(dirac: pd.DataFrame) -> list:
    """Editable centre records from DIRAC; capacity None follows linacs x capacity per linac."""
    return [{"name": r["name"], "city": r.get("city", ""), "lat": float(r["lat"]), "lon": float(r["lon"]),
             "linacs": float(r["n_linacs"]), "capacity": None, "planned": False} for _, r in dirac.iterrows()]


def _is_dirac(rows: list, dirac: pd.DataFrame) -> bool:
    return (len(rows) == len(dirac) and not any(r["planned"] or r["capacity"] is not None for r in rows)
            and np.allclose([r["linacs"] for r in rows], dirac["n_linacs"].to_numpy(float)))


def _cap(r: dict, capacity: float) -> float:
    return float(r["capacity"]) if r["capacity"] is not None else r["linacs"] * capacity


def _linacs_from_edits(edits: dict, capacity: float) -> dict:
    """alpha-2 -> ((lat, lon, linac-equivalents), ...): a centre's capacity divided by the capacity per linac."""
    return {a2: tuple((r["lat"], r["lon"], _cap(r, capacity) / capacity) for r in rows)
            for a2, rows in edits.items()}


@st.cache_data(show_spinner=False)
def _default_rtu_table() -> pd.DataFrame:
    f = {k.lower(): v for k, v in get_optimal_rt_fractions(DEFAULT_RTU_SET).items()}
    sites = [c for c in get_cancer_types() + DERIVED_CANCER_TYPES
             if c.strip().lower() not in {"all cancers", "all cancers excl. nmsc", "nmsc"}]
    return pd.DataFrame({"Cancer site": sites, "RTU (%)": [round(100 * f.get(c.lower(), 0.0), 1) for c in sites]})


LOAD_BANDS = [(0.8, [30, 120, 30], "under 80%"), (1.0, [240, 200, 30], "80–100%"),
              (1.5, [230, 120, 30], "100–150%"), (np.inf, [220, 50, 50], "over 150%")]


def _load_colour(x: float) -> list:
    for hi, rgb, _ in LOAD_BANDS:
        if x < hi:
            return rgb + [240]
    return LOAD_BANDS[-1][1] + [240]


def _facility_layer(fac: pd.DataFrame, capacity: float, by_load: bool = False, rings: bool = True) -> pdk.Layer:
    """Centres sized by their linacs: DIRAC centres in navy, planned centres in purple, or, with
    *by_load*, coloured by demand directed to the centre as a share of its capacity. *rings*
    draws outlines only, so the map stays visible underneath."""
    d = fac[fac["n_linacs"] > 0].copy()
    d["r"] = 3.0 + 2.2 * np.sqrt(d["n_linacs"])
    if by_load and "load" in d:
        d["fill"] = [_load_colour(x) for x in d["load"].fillna(0)]
    else:
        d["fill"] = [[128, 50, 200, 240] if p else [25, 55, 130, 230] for p in d.get("planned", False)]
    cap = d["capacity"] if "capacity" in d else d["n_linacs"] * capacity
    d["tip"] = ("<b>" + d["name"].astype(str) + "</b><br/>" + d["city"].fillna("").astype(str) + "<br/>"
                + d["n_linacs"].round(1).astype(str) + " linac(s), capacity " + cap.round(0).astype(int).astype(str)
                + " patients/yr")
    if "load" in d:
        d["tip"] = d["tip"] + "<br/>Load: " + (100 * d["load"]).round(0).astype("Int64").astype(str) + "% of capacity"
    if rings:
        return pdk.Layer("ScatterplotLayer", id="centres", data=d[["lon", "lat", "r", "fill", "tip"]],
                         get_position="[lon, lat]", get_radius="r", radius_units="pixels", filled=False,
                         stroked=True, get_line_color="fill", line_width_min_pixels=2, pickable=True,
                         parameters={"depthTest": False})
    return pdk.Layer("ScatterplotLayer", id="centres", data=d[["lon", "lat", "r", "fill", "tip"]],
                     get_position="[lon, lat]", get_radius="r", radius_units="pixels", get_fill_color="fill",
                     get_line_color=[255, 255, 255, 255], line_width_min_pixels=1, stroked=True, pickable=True,
                     parameters={"depthTest": False})  # always drawn over the hexagons


def _render_profiles(countries: pd.DataFrame, view: pdk.ViewState, extra: list) -> None:
    """Countries coloured by access profile, with a legend in place of the colour bar."""
    col_map, col_lg = st.columns([7, 1])
    gj = _land_geojson_profiles(tuple(sorted(
        (pycountry.countries.get(alpha_3=r["iso3"]).alpha_2,
         _profile(r["A_C"] / r["D"], r["A_G"] / r["D"], r["A"] / r["D"]), r["country"],
         round(100 * r["A"] / r["D"], 1)) for _, r in countries.iterrows() if r["D"] > 0)))
    layer = pdk.Layer("GeoJsonLayer", id="profiles", data=gj, filled=True, stroked=True,
                      get_fill_color="properties.fill", get_line_color=[255, 255, 255], line_width_min_pixels=0.5,
                      pickable=True)
    nd = _build_no_data_fill_layer()
    deck = pdk.Deck(layers=_plain_tips(([nd] if nd is not None else []) + [layer] + extra), initial_view_state=view,
                    map_style=_CARTO_LIGHT_NOLABELS, tooltip=_TOOLTIP)
    with col_map:
        st.pydeck_chart(deck, width="stretch", height=MAP_HEIGHT, key=f"profiles_{location}")
    with col_lg:
        st.markdown("<br/>".join(_chip(k) for k in PROFILES), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _country_geojson(rows: tuple) -> dict:
    """Country polygons from ((alpha-2, (r, g, b, a), tooltip), ...)."""
    from map_view import _load_country_borders
    by_a2 = {a2: (fill, tip) for a2, fill, tip in rows}
    feats = []
    for f in (_load_country_borders() or {}).get("features", []):
        a2 = f.get("properties", {}).get("iso_a2")
        if a2 in by_a2:
            fill, tip = by_a2[a2]
            feats.append({**f, "properties": {"fill": list(fill), "tip": tip}})
    return {"type": "FeatureCollection", "features": feats}


def _land_geojson_profiles(rows: tuple) -> dict:
    """Country polygons with a fill colour and tooltip per access profile."""
    return _country_geojson(tuple(
        (a2, tuple(PROFILES[prof] + [210]), f"<b>{name}</b><br/>{prof}<br/>RadMaps access {pct}%")
        for a2, prof, name, pct in rows))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_regions = _region_options()
LOCATIONS = _regions + [SEPARATOR] + _country_options()
edits: dict = st.session_state.setdefault("centre_edits", {})  # alpha-2 -> list of centre records
_ver = st.session_state.setdefault("ed_ver", 0)  # bumped to reset the table editors

_URL_KEYS = {"location": str, "res": int, "access": str, "t_max": float, "capacity": float, "model": str,
             "alloc": str, "layer": str}
if not st.session_state.get("_url_read"):
    st.session_state["_url_read"] = True
    for k, typ in _URL_KEYS.items():
        if k in st.query_params:
            try:
                st.session_state[k] = typ(st.query_params[k])
            except ValueError:
                pass
for k, ok in [("location", LOCATIONS), ("access", ["driving", "distance"]), ("t_max", TIMES), ("capacity", CAPACITIES), ("model", list(MODELS)),
              ("alloc", list(ALLOCATIONS))]:
    if st.session_state.get(k) not in ok:
        st.session_state.pop(k, None)
st.session_state.setdefault("location", "World")
for k, v in PAPER.items():
    st.session_state.setdefault(k, v)


def _reset_to_paper():
    for k, v in PAPER.items():
        st.session_state[k] = v
    st.session_state["centre_edits"] = {}
    st.session_state["applied_state"] = {"linacs": {}, "rtu": ()}
    st.session_state["ed_ver"] = st.session_state.get("ed_ver", 0) + 1


with st.sidebar:
    if _LOGO.exists():
        st.image(str(_LOGO), width=190)  # includes the version, v1.1
    else:
        st.title("RadMaps v1.1")
    with st.expander("Country / region", expanded=True):
        location = st.selectbox("Country or region", LOCATIONS, key="location",
                                help="Regions are listed first, then countries.")
        if location == SEPARATOR:
            st.info("Choose a region or a country.")
            st.stop()
        is_region = location in _regions
        res_opts = [1, 2, 3] if is_region else [3, 4, 5]
        # default H3 for regions and H5 for countries whenever the kind of location changes
        if st.session_state.get("res") not in res_opts or st.session_state.get("_was_region") != is_region:
            st.session_state["res"] = res_opts[-1]
        st.session_state["_was_region"] = is_region
        res = st.selectbox(
            "H3 resolution", res_opts, key="res",
            format_func=lambda r: (f"{r} ({math.sqrt(3) * h3.average_hexagon_edge_length(r, 'km'):,.0f} km; "
                                   f"{h3.average_hexagon_area(r, 'km^2'):,.0f} km²)"),
            help="Each resolution is shown with its average centroid-to-centroid distance and cell area. "
                 "Access is always computed at resolution 5 and summed into larger hexagons.")
    alpha2 = None if is_region else pycountry.countries.lookup(location).alpha_2
    _fac = _load_dirac(location)
    dirac = (_fac[1] if _fac[0] is not None
             else pd.DataFrame(columns=["name", "city", "lat", "lon", "n_linacs"])).reset_index(drop=True)
    facilities = dirac

    with st.expander("User inputs", expanded=False):
        st.button("Reset to the paper's configuration", on_click=_reset_to_paper, width="stretch",
                  help="Gaussian decay, 120 min, 450 patients per linac, capped 3SFCA, CCORE 2012 RTU and DIRAC "
                       "centres, as in the RadMaps paper.")
        access = st.selectbox(
            "Access variable", ["driving", "distance"], key="access",
            format_func={"driving": "Driving time", "distance": "Straight-line distance"}.get,
            help="Driving time (default) from TravelTime, leaving at 08:00 local time on a Wednesday; or "
                 "great-circle distance, with each time limit replaced by its calibrated straight-line "
                 "equivalent. Straight-line results are calculated on request.")
        t_max = st.selectbox("Driving-time limit" if access == "driving" else "Distance limit", TIMES, key="t_max",
                             format_func=(lambda t: f"{t:.0f} min") if access == "driving"
                             else (lambda t: f"{minutes_to_proxy_km(t):.1f} km (≈ {t:.0f} min drive)"),
                             help="Longest drive to a radiotherapy centre at which patients are still treated. "
                                  "Driving times leave at 08:00 local time on a Wednesday. Where TravelTime has "
                                  "no coverage (China, Russia, North and South Korea) the calibrated straight-line "
                                  "equivalent is used (30/60/90/120 min ≈ 13/28/42/64 km).")
        capacity = st.selectbox("Capacity per linac", CAPACITIES, key="capacity",
                                format_func=lambda c: f"{c:.0f} patients per year")
        model = st.selectbox("Distance-decay model", list(MODELS), format_func=MODELS.get, key="model",
                             help="How the probability of treatment falls with driving time; each model is cut off "
                                  "at the driving-time limit. The paper uses the Gaussian.")
        alloc = st.selectbox("Allocation method", list(ALLOCATIONS), format_func=ALLOCATIONS.get, key="alloc",
                             help="How patients are shared between the centres they can reach. Capped 3SFCA (the "
                                  "paper's) shares each hexagon's demand between all centres in reach, weighted by "
                                  "distance and size, and no centre treats more than its capacity. Nearest-first "
                                  "fills the nearest centres first, ring by ring of travel time.")

        st.markdown("**Radiotherapy utilisation**")
        rtu_mode = st.radio("Radiotherapy utilisation", ["Optimal (CCORE 2012)", "Custom"], horizontal=True,
                            label_visibility="collapsed", key="rtu_mode")
        custom_rtu: tuple = ()
        if rtu_mode == "Custom":
            st.caption("Share of each cancer site that should receive radiotherapy (CCORE 2012 by default).")
            ed_r = st.data_editor(_default_rtu_table(), hide_index=True, disabled=["Cancer site"], height=240,
                                  key=f"ed_rtu_{_ver}", column_config={
                                      "RTU (%)": st.column_config.NumberColumn(min_value=0, max_value=100, step=1)})
            custom_rtu = tuple(sorted((c, float(v)) for c, v in zip(ed_r["Cancer site"], ed_r["RTU (%)"].fillna(0))))
            if custom_rtu == tuple(sorted(zip(*[_default_rtu_table()[c] for c in ["Cancer site", "RTU (%)"]]))):
                custom_rtu = ()

        st.markdown("**Radiotherapy centres**")
        if is_region:
            st.caption("Edit each country's linacs in the table below the map; its centres keep their shares.")
        elif dirac.empty and alpha2 not in edits:
            st.caption("No radiotherapy centres in DIRAC for this country; add some in the Planning tab.")
        else:
            st.caption("Linacs and capacity of each centre. Capacity follows linacs x capacity per linac "
                       "unless you set it; 0 closes a centre.")
            rows = edits.get(alpha2) or _dirac_rows(dirac)
            shown = [_cap(r, capacity) for r in rows]
            ed = st.data_editor(
                pd.DataFrame({"Centre": [r["name"] for r in rows], "Linacs": [r["linacs"] for r in rows],
                              "Capacity (patients/yr)": shown}),
                hide_index=True, disabled=["Centre"], height=240, key=f"ed_{alpha2}_{_ver}",
                column_config={"Linacs": st.column_config.NumberColumn(min_value=0, step=1),
                               "Capacity (patients/yr)": st.column_config.NumberColumn(min_value=0, step=50,
                                                                                       format="%.0f")})
            new_rows = []
            for r, c0, n_new, c_new in zip(rows, shown, ed["Linacs"].fillna(0), ed["Capacity (patients/yr)"].fillna(0)):
                r2 = dict(r, linacs=float(n_new))
                if abs(float(c_new) - c0) > 1e-6:
                    r2["capacity"] = float(c_new)
                elif abs(float(n_new) - r["linacs"]) > 1e-9:
                    r2["capacity"] = None
                new_rows.append(r2)
            if _is_dirac(new_rows, dirac):
                edits.pop(alpha2, None)
            else:
                edits[alpha2] = new_rows
        if edits:
            st.caption("Edited: " + ", ".join(sorted(pycountry.countries.get(alpha_2=a).name for a in edits)))
            if st.button("Reset centres to DIRAC", width="stretch"):
                edits.clear()
                st.session_state["ed_ver"] = _ver + 1
                st.rerun()
        linacs = _linacs_from_edits(edits, capacity)

    with st.expander("Visualisation settings", expanded=False):
        _layers = LAYERS + (["Access profile"] if is_region else [])
        if st.session_state.get("layer") not in _layers:
            st.session_state["layer"] = _layers[0]
        layer = st.selectbox("Map", _layers, key="layer")
        national = (is_region and layer != "Access profile"
                    and st.checkbox("Show national averages", key="national",
                                    help="Colour each country by its national value instead of each hexagon."))
        show_towers = st.checkbox("Show Facility Locations", value=False)
        fac_style = (st.radio("Facility style", ["Rings", "Points", "Towers"], horizontal=True, key="fac_style",
                              help="Rings keep the map visible underneath; points are filled; towers are 3D.")
                     if show_towers else "Rings")
        colour_by_load = (show_towers and fac_style in ("Rings", "Points") and not is_region
                          and st.checkbox("Colour centres by load", key="colour_by_load",
                                          help="Demand directed to each centre as a share of its capacity "
                                               "(calculate the loads in the Data tab)."))
        _default_cmap = "Red → Green" if layer == "Access ratio" else "Green → Red"
        cmap_name, log_scale, auto_range = _default_cmap, False, True
        if layer != "Access profile":
            cmap_name = st.selectbox("Colour bar", list(COLORMAPS), index=list(COLORMAPS).index(_default_cmap),
                                     key=f"cmap_{layer}")
            log_scale = st.radio("Scale", ["Linear", "Log"], horizontal=True, key=f"scale_{layer}") == "Log"
            auto_range = st.checkbox("Auto range", value=True, key=f"auto_{layer}")
        deficit_range = "Own maximum"
        if auto_range and layer == "Deficit":
            deficit_range = st.radio(
                "Range", ["Own maximum", "Relative to its demand", "Relative to global demand"],
                key="deficit_range",
                help="Own maximum: 0 to the largest deficit shown. Relative to its demand: 1 to the largest "
                     "hexagon demand of this map, so deficit and demand share a scale. Relative to global "
                     "demand: 1 to the largest hexagon demand worldwide at this resolution, so every "
                     "country is on the same scale.")
        if not auto_range:
            _c1, _c2 = st.columns(2)
            vmin_user = _c1.number_input("Min", value=0.0, format="%.4g", key=f"vmin_{layer}")
            vmax_user = _c2.number_input("Max", value=1.0 if layer == "Access ratio" else 100.0,
                                         format="%.4g", key=f"vmax_{layer}")

# live calculations only on request: edited linacs, or a location without stored results
state = {"linacs": linacs, "rtu": custom_rtu}
applied_state: dict = st.session_state.setdefault("applied_state", {"linacs": {}, "rtu": ()})
with st.sidebar:
    if state != applied_state:
        st.warning("Inputs have changed. The map shows the last calculated results.")
        if st.button("Recalculate with changes", type="primary", width="stretch"):
            st.session_state["applied_state"] = applied_state = {"linacs": dict(linacs), "rtu": custom_rtu}
            st.rerun()
applied, applied_rtu = applied_state["linacs"], applied_state["rtu"]
if applied_rtu and location == "World":
    st.sidebar.info("Custom RTUs are applied to a country or a region; the World map uses CCORE 2012.")
    applied_rtu = ()
if access == "distance" and location == "World":
    st.info("Straight-line distance is calculated on request for a country or a region; the World view "
            "is pre-computed for driving time only.")
    st.stop()

_calc_key = ("calc", location, access, alloc, model, t_max, capacity)
with st.spinner(f"Loading results for {location}…"):
    try:
        allow = bool(st.session_state.get(_calc_key)) or bool(applied) or bool(applied_rtu)
        res_ = (_region_result(location, alloc, RTU, model, t_max, capacity, applied, allow, applied_rtu, access)
                if is_region else
                _country_result(alpha2, alloc, RTU, model, t_max, capacity, applied, allow, applied_rtu, access))
    except ValueError as exc:
        st.error(f"No results for {location}: {exc}")
        st.stop()
if res_ is None:
    st.info(f"No pre-built result is available for {location} with these settings.")
    if st.button("Calculate now", type="primary"):
        st.session_state[_calc_key] = True
        st.rerun()
    st.stop()
hexes, countries = res_
if not is_region and alpha2 in applied and alpha2 in edits:
    facilities = pd.DataFrame(edits[alpha2]).rename(columns={"linacs": "n_linacs"})
    facilities["capacity"] = [_cap(r, capacity) for r in edits[alpha2]]
else:
    facilities = dirac.assign(capacity=dirac["n_linacs"] * capacity, planned=False)
facilities = facilities.reset_index(drop=True)
_loads_key = None if is_region else ("loads", alpha2, access, alloc, model, t_max, capacity, applied_rtu,
                                     applied.get(alpha2))
_loads = st.session_state.get("loads", {}).get(_loads_key) if _loads_key else None
if _loads is not None and len(_loads) == len(facilities):
    _q = _loads["demand"].where(_loads["demand"].notna(), _loads["treated"])
    facilities["load"] = (_q / _loads["capacity"].replace(0, np.nan)).to_numpy()
    facilities["demand"], facilities["treated"] = _loads["demand"].to_numpy(), _loads["treated"].to_numpy()

# territories without GLOBOCAN estimates have no demand, and their linacs are not modelled
countries = countries[countries["D"] > 0]
if countries.empty:
    st.warning(f"{location} has no GLOBOCAN cancer estimate, so its radiotherapy access is not modelled.")
    st.stop()
g = _at_resolution(hexes, res)
g = g[g["D"] > 0].reset_index(drop=True)
g["ratio"] = g["A"] / g["D"]
g["deficit"] = (g["D"] - g["A"]).clip(lower=0)
tot = countries[["D", "A_C", "A_G", "A", "n_linacs"]].sum()
proxy = sorted(countries.loc[countries["time_source"] == "proxy", "country"])

tab_map, tab_plan, tab_data, tab_about = st.tabs(["🗺️ Maps", "🔧 Planning", "📊 Data", "📖 About"])

with tab_map:
    st.header(f"{layer} — {location}")
    ll = np.array([h3.cell_to_latlng(c) for c in g["h3"]])
    if layer == "Access ratio":
        vals, cmap_fn, vmin, vmax, label = g["ratio"].to_numpy(), _rdylgn_rgb, 0.0, 1.0, layer
    elif layer == "Deficit":
        vals, cmap_fn, label = g["deficit"].to_numpy(), _rdylgn_reversed_rgb, f"{layer} per hexagon"
        vmin, vmax = 0.0, float(np.nanmax(vals))
    else:
        vals, cmap_fn, label = g["D"].to_numpy(), _rdylgn_reversed_rgb, "RT demand per hexagon"
        vmin, vmax = float(max(vals.min(), 0.001)), float(vals.max())
    cmap_fn = COLORMAPS[cmap_name]
    if layer == "Deficit" and auto_range and deficit_range != "Own maximum":
        vmin = 1.0
        vmax = float(g["D"].max()) if deficit_range == "Relative to its demand" else _global_max_demand(RTU, res)
        label = f"{layer} per hexagon ({deficit_range.lower()})"
    if not auto_range:
        vmin, vmax = float(vmin_user), float(vmax_user)
    if log_scale:
        vmin = max(vmin, 0.001)
        g["color"] = _apply_colormap_fixed(np.log1p(vals), cmap_fn, np.log1p(vmin), np.log1p(vmax))
    else:
        g["color"] = _apply_colormap_fixed(vals, cmap_fn, vmin, vmax)
    g["tip"] = ("<b>" + g["country"] + "</b><br/>"
                + "Access ratio: " + (100 * g["ratio"]).round(1).astype(str) + "%<br/>"
                + "RT demand: " + g["D"].round(1).astype(str) + "<br/>"
                + "RT accessed: " + g["A"].round(1).astype(str) + "<br/>"
                + "RT deficit: " + g["deficit"].round(1).astype(str) + "<br/>"
                + "<hr style='margin:3px 0'/>Population: " + g["population"].map(_fmt))
    layers = [_build_hex_layer(g[["h3", "color", "tip"]], 1.0)]
    if national:
        cn = countries[countries["D"] > 0].copy()
        cn["v"] = {"Access ratio": cn["A"] / cn["D"], "Deficit": cn["D"] - cn["A"]}.get(layer, cn["D"])
        if layer != "Access ratio":
            label = f"National {'deficit' if layer == 'Deficit' else 'RT demand'} (patients per year)"
            if auto_range:
                vmin, vmax = (1.0 if log_scale else 0.0), float(cn["v"].max())
        else:
            label = "National access ratio, A/D"
        v = cn["v"].to_numpy(np.float64)
        cols = (_apply_colormap_fixed(np.log1p(v), cmap_fn, np.log1p(max(vmin, 0.001)), np.log1p(vmax)) if log_scale
                else _apply_colormap_fixed(v, cmap_fn, vmin, vmax))
        fmt = (lambda x: f"{100 * x:.1f}%") if layer == "Access ratio" else (lambda x: f"{x:,.0f} patients/yr")
        gj = _country_geojson(tuple(
            (pycountry.countries.get(alpha_3=r.iso3).alpha_2, tuple(c[:3] + [220]),
             f"<b>{r.country}</b><br/>{layer}: {fmt(r.v)}")
            for r, c in zip(cn.itertuples(), cols)))
        layers = [pdk.Layer("GeoJsonLayer", id="national", data=gj, filled=True, stroked=True,
                            get_fill_color="properties.fill", get_line_color=[255, 255, 255],
                            line_width_min_pixels=0.5, pickable=True)]
    towers = show_towers and not facilities.empty
    if towers:
        if fac_style in ("Rings", "Points"):
            layers.append(_facility_layer(facilities, capacity, by_load=colour_by_load, rings=fac_style == "Rings"))
        else:
            f = facilities[facilities["n_linacs"] > 0]
            span = 111.32 * max(float(np.ptp(ll[:, 0])),
                                float(np.ptp(ll[:, 1])) * math.cos(math.radians(float(ll[:, 0].mean()))))
            layers += _build_linac_columns(f, h3_res=res, country_span_km=span, style="stacked", color=_LINAC_BLUE)
    pitch = 30.0 if towers and fac_style == "Towers" else 0.0
    if layer == "Access profile":
        _render_profiles(countries, _view(ll[:, 0], ll[:, 1], 0.0), layers[1:])
    else:
        _render(layers, _view(ll[:, 0], ll[:, 1], pitch), cmap_fn, vmin, vmax, label, is_region,
                log_scale)
    if colour_by_load and "load" not in facilities:
        st.caption("Centre loads are not yet calculated for these settings: use the Data tab.")
    elif colour_by_load:
        st.caption("Centre load (demand directed to the centre ÷ its capacity): "
                   + " · ".join(f"<span style='color:rgb({c[0]},{c[1]},{c[2]})'>●</span> {t}"
                                for _, c, t in LOAD_BANDS), unsafe_allow_html=True)

    D = float(tot["D"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RT demand", f"{_fmt(D)} /yr", help="Patients per year who would benefit from radiotherapy")
    c2.metric("Capacity-only access", f"{100 * tot['A_C'] / D:.1f}%",
              help="Share of demand the linacs could treat if every patient could reach them")
    c3.metric("Geography-only access", f"{100 * tot['A_G'] / D:.1f}%",
              help="Share of demand within reach of a centre if no linac were ever full")
    c4.metric("RadMaps access", f"{100 * tot['A'] / D:.1f}%",
              help="Share of demand treated: capacity and geography together")
    c5.metric("Deficit ∆", f"{_fmt(D - tot['A'])} /yr", help="Patients per year who would benefit but are not treated")
    if is_region:
        _n = countries.apply(lambda r: _profile(r["A_C"] / r["D"], r["A_G"] / r["D"], r["A"] / r["D"]), axis=1)
        _line = "**Access profiles of its countries:** " + ", ".join(
            f"{_chip(k)} {int((_n == k).sum())}" for k in PROFILES)
    else:
        _line = f"**Access profile:** {_chip(_profile(tot['A_C'] / D, tot['A_G'] / D, tot['A'] / D))}"
    st.markdown(_line, unsafe_allow_html=True,
                help="Strong access: A/D ≥ 80%. Otherwise capacity constrained (A_C < A_G), geography constrained "
                     "(A_G < A_C), or doubly constrained when both A_C/D and A_G/D are below 80% (paper, "
                     "Section 3.2).")
    _diff = [n for n, k, v in [("access variable", "access", access), ("driving-time limit", "t_max", t_max), ("capacity", "capacity", capacity),
                               ("decay model", "model", model), ("allocation", "alloc", alloc)] if PAPER[k] != v]
    _diff += ["custom RTU"] if applied_rtu else []
    _diff += ["edited centres"] if applied else []
    if _diff:
        st.caption("⚠️ Differs from the paper's main configuration in: " + ", ".join(_diff) + ".")
    st.caption(
        f"{int(round(tot['n_linacs'])):,} linacs{' (edited)' if applied else ''}"
        f"{', custom RTU' if applied_rtu else ''}. "
        + (f"Driving-time limit {t_max:.0f} min, " if access == "driving" else
           f"Straight-line distance limit {minutes_to_proxy_km(t_max):.1f} km (calibrated equivalent of "
           f"{t_max:.0f} min), ")
        + f"{MODELS[model]} decay, {capacity:.0f} patients per linac per year, {'custom' if applied_rtu else '2012 CCORE optimal'} RTU, {ALLOCATIONS[alloc]} "
        f"allocation; computed at H3 resolution 5 and shown at resolution {res}."
        + (f" Straight-line equivalent of driving time used for {', '.join(proxy)}." if proxy else "")
    )

    if is_region:  # the region's linac editor: one row per country
        with st.expander("Number of linacs by country", expanded=False):
            ct = countries.reset_index(drop=True)
            ed = st.data_editor(pd.DataFrame({"Country": ct["country"], "Linacs": ct["n_linacs"]}),
                                hide_index=True, disabled=["Country"], key=f"ed_{location}",
                                column_config={"Linacs": st.column_config.NumberColumn(min_value=0, format="%.0f")})
            changed = False
            for iso3, name, old, new in zip(ct["iso3"], ct["country"], ct["n_linacs"], ed["Linacs"].fillna(0)):
                if abs(float(new) - float(old)) > 1e-9:
                    d0 = _load_dirac(name)
                    if d0[0]:  # every centre scaled by the same factor
                        rows = _dirac_rows(d0[1].reset_index(drop=True))
                        n0 = sum(r["linacs"] for r in rows)
                        edits[pycountry.countries.get(alpha_3=iso3).alpha_2] = [
                            dict(r, linacs=r["linacs"] * float(new) / n0) for r in rows]
                        changed = True
            if changed:
                st.rerun()

with tab_plan:
    st.header(f"Radiotherapy planning — {location}")
    if is_region:
        st.info("Planning works on one country at a time: choose a country in the sidebar.")
    else:
        st.caption(
            "Adds linacs one step at a time where they treat the most patients, recomputing access "
            "after each step, with the user inputs in the sidebar. A step adds linacs at the hexagon with the "
            "largest unmet demand (or the most demand out of reach), at the nearest existing centre when one lies "
            "within the chosen driving time, otherwise as a new centre. Driving times to new centres use the "
            "calibrated straight-line equivalent.")
        goal = st.radio("Add", ["A number of linacs", "Linacs until access reaches a target"], horizontal=True,
                        key="plan_goal")
        c1, c2, c3, c4 = st.columns(4)
        n_add, target = None, None
        if goal == "A number of linacs":
            n_add = int(c1.number_input("Linacs to add", min_value=1, value=10, step=1))
        else:
            cur = 100 * float(countries["A"].sum() / countries["D"].sum())
            target = float(c1.number_input("Target RadMaps access (%)", min_value=1.0, max_value=99.0,
                                           value=float(min(99.0, max(80.0, round(cur + 10)))), step=1.0,
                                           help=f"Currently {cur:.1f}%.")) / 100
        per_step = int(c2.number_input("Linacs per step", min_value=1, max_value=20, value=2, step=1,
                                       help="Each step places this many linacs together and recalculates "
                                            "access for the whole country; more per step is faster."))
        strategy = c3.selectbox("Place at the hexagon with", ["Largest unmet demand", "Most demand out of reach"])
        snap = c4.selectbox("Add to an existing centre within", [None, 15.0, 30.0, 60.0], index=2,
                            format_func=lambda v: "Never" if v is None else f"{v:.0f} min")
        compare = st.checkbox("Also run the other placement strategy, for comparison", key="plan_compare")
        MAX_STEPS = 200
        base_rows = edits.get(alpha2) if alpha2 in applied else None
        base_rows = base_rows or _dirac_rows(dirac)
        base_locs = tuple((r["lat"], r["lon"], _cap(r, capacity) / capacity) for r in base_rows)
        if st.button("Run planning", type="primary"):
            bar = st.progress(0.0, text="Planning")
            total_steps = -(-n_add // per_step) if n_add else MAX_STEPS

            def _progress(i, pct, added):
                frac = i / total_steps if n_add else min(1.0, pct / (100 * target))
                bar.progress(min(frac, 1.0), text=f"Planning: {added} linacs added, access {pct:.1f}%")

            before, steps, locs = plan_centres(
                alpha2, model, t_max, capacity, base_locs, n_add, per_step,
                "deficit" if strategy == "Largest unmet demand" else "isolation", snap, applied_rtu,
                target_access=target, max_steps=MAX_STEPS, allocation=alloc, access=access, progress_callback=_progress)
            other = None
            if compare:
                _, other, _ = plan_centres(
                    alpha2, model, t_max, capacity, base_locs, n_add, per_step,
                    "isolation" if strategy == "Largest unmet demand" else "deficit", snap, applied_rtu,
                    target_access=target, max_steps=MAX_STEPS, allocation=alloc, access=access, progress_callback=_progress)
            bar.empty()
            if target and steps and steps[-1]["A/D (%)"] < 100 * target:
                st.warning(f"Stopped after {MAX_STEPS} steps at {steps[-1]['A/D (%)']:.1f}% access; increase the "
                           "linacs per step to go further.")
            st.session_state["plan"] = {"location": location, "before": before, "steps": steps, "locs": locs,
                                        "base_rows": base_rows, "base_locs": base_locs, "capacity": capacity,
                                        "strategy": strategy, "other": other}
        plan = st.session_state.get("plan")
        if plan and plan["location"] == location:
            b0, steps = plan["before"], plan["steps"]
            if not steps:
                st.info("No hexagon has unmet demand, so no linacs were added.")
            else:
                last = steps[-1]
                m0, m1, m2, m3 = st.columns(4)
                m0.metric("Patients treated", f"{_fmt(last['treated'])} /yr", f"{last['treated'] - b0['A']:+,.0f}")
                m1.metric("RadMaps access", f"{last['A/D (%)']:.1f}%",
                          f"{last['A/D (%)'] - 100 * b0['A'] / b0['D']:+.1f} points")
                m2.metric("Deficit ∆", f"{_fmt(last['deficit'])} /yr",
                          f"{last['deficit'] - (b0['D'] - b0['A']):+,.0f}", delta_color="inverse")
                m3.metric("Linacs added", f"{sum(x['linacs_added'] for x in steps)}")
                tbl = pd.DataFrame(steps).assign(per_linac=lambda d: d["additional"] / d["linacs_added"]).round(
                    {"treated": 0, "additional": 0, "per_linac": 0, "deficit": 0}).rename(
                    columns={"step": "Step", "lat": "Latitude", "lon": "Longitude", "where": "Added to",
                             "linacs_added": "Linacs", "treated": "Patients treated",
                             "additional": "Additional patients", "per_linac": "Additional patients per linac",
                             "A/D (%)": "RadMaps access (%)", "deficit": "Deficit"})
                st.dataframe(tbl, hide_index=True, width="stretch",
                             column_config={"RadMaps access (%)": st.column_config.NumberColumn(format="%.1f"),
                                            **{c: st.column_config.NumberColumn(format="localized")
                                               for c in ["Patients treated", "Additional patients",
                                                         "Additional patients per linac", "Deficit"]}})
                st.download_button("⬇️ Download the plan (CSV)", tbl.to_csv(index=False).encode(),
                                   file_name=f"radmaps_plan_{location.replace(' ', '_')}.csv")
                if plan.get("other"):
                    fig = go.Figure()
                    for name, st_ in [(plan["strategy"], steps),
                                      ("Most demand out of reach" if plan["strategy"] == "Largest unmet demand"
                                       else "Largest unmet demand", plan["other"])]:
                        x = [0] + list(np.cumsum([z["linacs_added"] for z in st_]))
                        y = [100 * b0["A"] / b0["D"]] + [z["A/D (%)"] for z in st_]
                        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name))
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                                      xaxis_title="Linacs added", yaxis_title="RadMaps access (%)",
                                      title="Placement strategies compared", legend=dict(orientation="h", y=-0.3))
                    st.plotly_chart(fig, width="stretch")
                if st.button("Apply to the map", help="Adds these linacs to the centres table and recalculates."):
                    rows = [dict(r) for r in plan["base_rows"]]
                    k = sum(r["planned"] for r in rows)
                    for i, (la, lo, n) in enumerate(plan["locs"]):
                        if i < len(rows):
                            add = n - plan["base_locs"][i][2]
                            if add > 1e-9:
                                rows[i]["linacs"] += add
                                if rows[i]["capacity"] is not None:
                                    rows[i]["capacity"] += add * plan["capacity"]
                        else:
                            k += 1
                            rows.append({"name": f"Planned centre {k}", "city": "", "lat": la, "lon": lo,
                                         "linacs": n, "capacity": None, "planned": True})
                    edits[alpha2] = rows
                    st.session_state["ed_ver"] = _ver + 1
                    st.session_state["applied_state"] = {"linacs": _linacs_from_edits(edits, capacity),
                                                         "rtu": custom_rtu}
                    st.session_state.pop("plan", None)
                    st.rerun()

def _rtu_table() -> None:
    """The RTU table, shown below the countries or centres table."""
    st.subheader("Radiotherapy utilisation rates")
    st.caption("Share of each cancer site that should receive external-beam radiotherapy (curative and palliative "
               "first courses): the optimal rates of the 2012 CCORE model (Barton et al. 2014; Wong et al. 2016), "
               "mapped to the GLOBOCAN sites as in Appendix A.5 of the paper.")
    _rt = _default_rtu_table().rename(columns={"RTU (%)": "Optimal RTU, CCORE 2012 (%)"})
    if applied_rtu:
        _rt["Custom RTU in use (%)"] = _rt["Cancer site"].map(dict(applied_rtu))
    st.dataframe(_rt, hide_index=True, width="stretch", height=300)


with tab_data:
    if is_region:
        st.subheader(f"Countries — {location}")
        _prof = [_profile(r["A_C"] / r["D"], r["A_G"] / r["D"], r["A"] / r["D"]) for _, r in countries.iterrows()]
        t = pd.DataFrame({
            "Country": countries["country"], "Access profile": _prof,
            "RT demand": countries["D"], "Linacs": countries["n_linacs"],
            "Capacity-only (%)": 100 * countries["A_C"] / countries["D"],
            "Geography-only (%)": 100 * countries["A_G"] / countries["D"],
            "RadMaps (%)": 100 * countries["A"] / countries["D"],
            "Deficit": countries["D"] - countries["A"],
            "Driving times": countries["time_source"].map(
                {"traveltime": "TravelTime", "proxy": "straight-line equivalent", "none": "no centres"}),
        }).sort_values("Deficit", ascending=False)
        t[["RT demand", "Linacs", "Deficit"]] = t[["RT demand", "Linacs", "Deficit"]].round(0)
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                    ["Capacity-only (%)", "Geography-only (%)", "RadMaps (%)"]}
                     | {c: st.column_config.NumberColumn(format="localized") for c in ["RT demand", "Linacs", "Deficit"]})
        st.download_button("⬇️ Download country table (CSV)", t.to_csv(index=False).encode(),
                           file_name=f"radmaps_{location.replace(' ', '_')}_countries.csv")
        _rtu_table()

        st.subheader("RadMaps access against the single-constraint metrics")
        st.caption("Each country with at least one linac, as in Figure 5 of the paper. Points below the diagonal "
                   "are countries where the combined access is lower than the single metric suggests.")
        _sc = countries[countries["n_linacs"] > 0].assign(
            ac=lambda d: 100 * d["A_C"] / d["D"], ag=lambda d: 100 * d["A_G"] / d["D"], a=lambda d: 100 * d["A"] / d["D"])
        _sc["profile"] = [_profile(r.ac / 100, r.ag / 100, r.a / 100) for r in _sc.itertuples()]
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                            subplot_titles=("Against capacity-only access", "Against geography-only access"))
        for prof, rgb in PROFILES.items():
            d = _sc[_sc["profile"] == prof]
            for col, x in [(1, "ac"), (2, "ag")]:
                fig.add_trace(go.Scatter(
                    x=d[x], y=d["a"], mode="markers", name=prof, legendgroup=prof, showlegend=col == 1,
                    marker=dict(size=8, color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", line=dict(width=0.5, color="white")),
                    text=d["country"], hovertemplate="%{text}<br>%{x:.1f}% → RadMaps %{y:.1f}%<extra></extra>"),
                    row=1, col=col)
                fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color="#bbb", dash="dot"),
                                         showlegend=False, hoverinfo="skip"), row=1, col=col)
        fig.update_xaxes(range=[0, 100], title_text="Share of demand (%)")
        fig.update_yaxes(range=[0, 100], title_text="RadMaps access, A/D (%)", row=1, col=1)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    else:
        st.subheader(f"Radiotherapy centres — {location}")
        if st.button("Calculate centre loads",
                     help="Demand directed to each centre and patients it treats, for the current inputs. "
                          "Takes a few seconds to a minute depending on the country."):
            with st.spinner("Calculating centre loads…"):
                base = tuple((r["lat"], r["lon"], _cap(r, capacity) / capacity)
                             for r in (edits[alpha2] if alpha2 in applied and alpha2 in edits
                                       else _dirac_rows(dirac)))
                st.session_state.setdefault("loads", {})[_loads_key] = centre_loads(
                    alpha2, model, t_max, capacity, base, applied_rtu, alloc, access)
            st.rerun()
        cols = ["name", "city", "n_linacs", "capacity"] + (["demand", "treated", "load"] if "load" in facilities else [])
        st.dataframe(facilities[cols + ["lat", "lon"]].assign(
            **({"load": 100 * facilities["load"]} if "load" in facilities else {})).round(
            {"capacity": 0, "demand": 0, "treated": 0, "load": 0}).rename(
            columns={"name": "Centre", "city": "City", "n_linacs": "Linacs", "capacity": "Capacity (patients/yr)",
                     "demand": "Demand directed (patients/yr)", "treated": "Patients treated", "load": "Load (%)",
                     "lat": "Latitude", "lon": "Longitude"}),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="localized") for c in
                           ["Capacity (patients/yr)", "Demand directed (patients/yr)", "Patients treated", "Load (%)"]})
        if "load" in facilities and alloc == "nearest_first":
            st.caption("Nearest-first fills centres in order of travel time, so the load shown is the share of "
                       "capacity used.")
        _rtu_table()

    # ---- sensitivity of this location to the pre-computed inputs
    st.subheader(f"Sensitivity — {location}")
    st.caption("RadMaps access for every pre-computed combination of inputs (DIRAC centres, CCORE 2012 RTU), "
               "as in Appendix B of the paper; the current inputs are circled.")
    _w = _world_tables()
    if _w is not None:
        _c = _w[1]
        _c = _c[_c["rtu"] == RTU]
        _c = _c[_c["allocation"] == alloc] if "allocation" in _c else (_c if alloc == "proportional" else _c.iloc[:0])
        _c = _c[_c["iso3"].isin(countries["iso3"])]
        if _c.empty:
            st.info("No pre-computed results for this allocation yet.")
        else:
            g_ = _c.groupby(["model", "t_max", "capacity"])[["D", "A"]].sum().reset_index()
            g_["pct"] = 100 * g_["A"] / g_["D"]
            fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                                subplot_titles=(f"Driving-time limit ({capacity:.0f} patients per linac)",
                                                f"Capacity per linac ({t_max:.0f} min)"))
            for m, name in MODELS.items():
                d = g_[(g_["model"] == m) & (g_["capacity"] == capacity)].sort_values("t_max")
                fig.add_trace(go.Scatter(x=d["t_max"], y=d["pct"], mode="lines+markers", name=name.split(" (")[0],
                                         legendgroup=m), row=1, col=1)
                d = g_[(g_["model"] == m) & (g_["t_max"] == t_max)].sort_values("capacity")
                fig.add_trace(go.Scatter(x=d["capacity"], y=d["pct"], mode="lines+markers", name=name, legendgroup=m,
                                         showlegend=False), row=1, col=2)
            cur = g_[(g_["model"] == model) & (g_["t_max"] == t_max) & (g_["capacity"] == capacity)]["pct"]
            if len(cur):
                for col, x in [(1, t_max), (2, capacity)]:
                    fig.add_trace(go.Scatter(x=[x], y=[float(cur.iloc[0])], mode="markers", showlegend=False,
                                             marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(width=2, color="#222")),
                                             hoverinfo="skip"), row=1, col=col)
            fig.update_xaxes(title_text="min", tickvals=TIMES, row=1, col=1)
            fig.update_xaxes(title_text="patients per year", tickvals=CAPACITIES, row=1, col=2)
            fig.update_yaxes(range=[0, 100], title_text="RadMaps access, A/D (%)", row=1, col=1)
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width="stretch")

    hx = g[["h3", "country", "population", "D", "A", "A_G", "deficit"]].rename(
        columns={"D": "rt_demand", "A": "rt_treated", "A_G": "geo_accessible_demand", "deficit": "rt_deficit"})
    st.download_button(f"⬇️ Download hex data at resolution {res} (CSV)", hx.to_csv(index=False).encode(),
                       file_name=f"radmaps_{location.replace(' ', '_')}_res{res}.csv")

with tab_about:
    st.markdown(
        """
The RadMaps methodology is described in
[Wroe *et al.*, "RadMaps: A Geospatial Framework for Simultaneously Modelling Capacity and Geographic
Constraints on Radiotherapy Access", arXiv:2605.23456](https://arxiv.org/abs/2605.23456).

### What the map shows
For every hexagon, RadMaps estimates the cancer patients each year who would benefit from radiotherapy
(the RT demand, *D*) and how many of them are treated (*A*), given where the linacs are, how many patients
each can treat, and how far patients would have to drive.

* **Access ratio** — the share of a hexagon's demand that is treated, *A/D*.
* **Deficit** — the patients per year who would benefit but are not treated, *∆ = D − A*.
* **Radiotherapy demand** — GLOBOCAN 2024 cancer incidence spread over the population (Kontur) and
  multiplied by the optimal radiotherapy utilisation of each cancer site (2012 CCORE model).

### How access is calculated
Each hexagon's patients are shared between the centres they can reach, weighted by distance decay and
centre size. A centre asked to treat more patients than its capacity treats the same fraction of every
share it receives (capped 3SFCA, the paper's method); patients turned away are not sent elsewhere, so
the estimate is conservative. Alternatively, **nearest-first** allocation fills the nearest centres
first, ring by ring of travel time. The results compare this with two single-constraint measures: **capacity-only** access
(linacs against demand, ignoring distance) and **geography-only** access (distance alone, ignoring full
linacs).

### User inputs
* **Driving-time limit** — 30, 60, 90 or 120 minutes (the paper uses 120).
* **Capacity per linac** — 250 to 550 patients per year (the paper uses 450).
* **Distance-decay model** — Gaussian (the paper's choice), step, exponential or power law, each cut off
  at the driving-time limit.
* **Radiotherapy utilisation** — the optimal CCORE 2012 rates (the paper's), or your own rate for each
  cancer site.
* **Radiotherapy centres** — the linacs and capacity of each centre (or, for a region, the linacs in each
  country), from the IAEA DIRAC database by default; capacity follows linacs × capacity per linac
  unless you set it.

Changes to the RTUs or centres are recalculated when you press **Recalculate with changes**; everything
else is read from pre-computed results.

### Radiotherapy planning
For a single country, the Planning tab adds linacs step by step where they treat the most patients:
at the hexagon with the largest unmet demand (or the most demand out of reach), either at the nearest
existing centre within a chosen driving time or as a new centre. Driving times to new centres use the
calibrated straight-line equivalent. The result can be applied to the map.

Driving times are from TravelTime, leaving at 08:00 local time on a Wednesday, from the centre of each
H3 resolution-6 hexagon to each radiotherapy centre. Access is computed at H3 resolution 5 (about
250 km²) and summed into the larger hexagons shown.
        """
    )

# shareable link: the current view in the URL
st.query_params.from_dict({"location": location, "res": res, "access": access, "t_max": t_max, "capacity": capacity,
                           "model": model, "alloc": alloc, "layer": layer})
