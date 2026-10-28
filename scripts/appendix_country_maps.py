"""
Appendix country maps: demand, MVMs, access ratio and deficit for the example
countries, as a 2x2 grid in LaTeX, in the paper's main configuration (capped 3SFCA,
Gaussian 120-min driving catchment from the res-6 store, 450 patients per
linac, CCORE 2012 optimal RTU).

Demand, access ratio and deficit are at H3 resolution 5. Demand and deficit
share one logarithmic colour scale across all countries, from 1 to 50,000
patients per year per hexagon, so the maps compare directly; the access
ratio runs from 0 to 100 %. MVMs are counted per resolution-4 hexagon (finer
cells are too small to see at country scale) with the bins and colours of the
global facilities map, over the populated resolution-5 hexagons in grey.

  figures/appendix_{ISO3}_res5.parquet    h3, D, A per hexagon (the compute step)
  figures/appendix_{ISO3}_{demand,mvm,ratio,deficit}_res{4,5}.png   display resolution 4 or 5

    python scripts/appendix_country_maps.py            # compute (if missing) and plot
    python scripts/appendix_country_maps.py --plot     # plot from the saved parquets only
    python scripts/appendix_country_maps.py --plot --res 4

Lower display resolutions sum demand and access of the resolution-5 hexagons
into their parents, as the global maps do at resolution 3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_FIG = _ROOT / "figures"

from scripts.global_figures import CM_ACCESS, CM_DEFICIT  # noqa: E402
from scripts.results_figures import mvm_colorbar, mvm_scale  # noqa: E402

RES, MVM_RES, CAP, T_MAX = 5, 4, 450.0, 120.0
VMAX = 50_000.0  # demand/deficit scale maximum (largest resolution-4 hexagon: 46,434)
LAND = "#e4e4e1"
_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}

# name, projected CRS, optional lon/lat window (contiguous United States)
COUNTRIES = {
    "USA": ("United States", "EPSG:5070", (-125.0, 24.0, -66.5, 49.5)),
    "JPN": ("Japan", "+proj=lcc +lat_1=30 +lat_2=42 +lat_0=36 +lon_0=136 +datum=WGS84 +units=m", None),
    "MEX": ("Mexico", "EPSG:6372", None),
    "IND": ("India", "EPSG:7755", None),
}
# each map fills half the page width; wide countries get a shorter figure (inches)
WIDE = {"USA", "MEX"}
SIZE_WIDE, SIZE_TALL = (3.3, 2.7), (3.3, 3.6)


def _parquet(iso3):
    return _FIG / f"appendix_{iso3}_res{RES}.parquet"


def compute(iso3):
    from analysis.accessibility import compute_accessibility
    from data.cancer import DERIVED_CANCER_TYPES, apportion_cancer_to_h3, get_cancer_types
    from data.linacs import load_linacs_from_dirac_db
    from data.population import load_population_at_resolution
    from data.tt_store import store_matrix

    name = COUNTRIES[iso3][0]
    gdf = load_population_at_resolution(name, target_resolution=RES)
    cg = apportion_cancer_to_h3(gdf, iso3, get_cancer_types() + DERIVED_CANCER_TYPES,
                                use_actual_rt=False, rtu_set="barton_2014")
    cols = [c for c in cg.columns if c.endswith("_optimal_rt") and c[:-11].strip().lower() not in _AGG]
    dem = cg[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)
    locs, _ = load_linacs_from_dirac_db(name)
    tt = store_matrix(iso3, list(gdf["h3"]), locs)
    if tt is None:
        raise RuntimeError(f"{iso3}: no driving-time store")
    g, s = compute_accessibility(gdf, locs, model="gaussian", max_distance_km=T_MAX,
                                 capacity_per_machine_per_year=CAP, demand=dem, h3_resolution=RES,
                                 travel_time_matrix=tt, allocation="proportional")
    out = pd.DataFrame({"h3": list(gdf["h3"]), "D": dem, "A": g["rt_treated"].to_numpy(np.float64)})
    out.to_parquet(_parquet(iso3), index=False)
    pd.DataFrame([(la, lo, n) for la, lo, n in locs], columns=["lat", "lon", "n_mvm"]).to_parquet(
        _FIG / f"appendix_{iso3}_centres.parquet", index=False)
    print(f"{iso3}: {len(out)} hexagons, A/D = {100 * out.A.sum() / out.D.sum():.1f}%", flush=True)


def _hexes(df, crs, window):
    if window is not None:
        lat_lon = np.array([h3.cell_to_latlng(c) for c in df["h3"]])
        x0, y0, x1, y1 = window
        keep = (lat_lon[:, 1] >= x0) & (lat_lon[:, 1] <= x1) & (lat_lon[:, 0] >= y0) & (lat_lon[:, 0] <= y1)
        df = df[keep]
    polys = [Polygon([(lo, la) for la, lo in h3.cell_to_boundary(c)]) for c in df["h3"]]
    return gpd.GeoDataFrame(df.reset_index(drop=True), geometry=polys, crs="EPSG:4326").to_crs(crs)


def _save(fig, ax, name):
    ax.set_axis_off()
    fig.savefig(_FIG / name, bbox_inches="tight")
    plt.close(fig)


def _map(g, col, cmap, norm, label, name, wide=False):
    """One map at half the page width, sized close to print so the text stays legible.
    On a log scale the lowest decade is labelled "<10^k": smaller values take the lowest colour."""
    fig, ax = plt.subplots(figsize=SIZE_WIDE if wide else SIZE_TALL, dpi=400)
    g.plot(ax=ax, column=col, cmap=cmap, norm=norm, linewidth=0)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                      fraction=0.05, pad=0.02, aspect=25)
    if isinstance(norm, LogNorm):
        lo, hi = int(np.log10(norm.vmin)), int(np.floor(np.log10(norm.vmax)))
        cb.set_ticks([10.0 ** k for k in range(lo, hi + 1)],
                     labels=[f"$<10^{{{lo}}}$"] + [f"$10^{{{k}}}$" for k in range(lo + 1, hi + 1)])
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _save(fig, ax, name)


def _mvm_map(g, centres, crs, name, wide=False):
    """MVMs per resolution-4 hexagon over the populated land in grey."""
    c = centres.assign(h3=[h3.latlng_to_cell(la, lo, MVM_RES) for la, lo in zip(centres["lat"], centres["lon"])])
    c = c.groupby("h3", as_index=False)["n_mvm"].sum()
    polys = [Polygon([(lo, la) for la, lo in h3.cell_to_boundary(x)]) for x in c["h3"]]
    m = gpd.GeoDataFrame(c, geometry=polys, crs="EPSG:4326").to_crs(crs)
    cmap, bnorm, labels = mvm_scale()
    fig, ax = plt.subplots(figsize=SIZE_WIDE if wide else SIZE_TALL, dpi=400)
    g.plot(ax=ax, color=LAND, linewidth=0)
    m.plot(ax=ax, column="n_mvm", cmap=cmap, norm=bnorm, linewidth=0)
    cb = mvm_colorbar(fig, ax, cmap, labels, labelsize=7, orientation="horizontal", fraction=0.05, pad=0.02,
                      aspect=25)
    cb.set_label("Linacs per hexagon", fontsize=8)
    _save(fig, ax, name)


def _at(df, res):
    """Sum demand and access of the resolution-5 hexagons into their parents at *res*."""
    if res == RES:
        return df.copy()
    return (df.assign(h3=[h3.cell_to_parent(c, res) for c in df["h3"]])
              .groupby("h3", as_index=False)[["D", "A"]].sum())


def plot(res=RES):
    """All panels at display resolution *res* (<= 5), aggregated from the resolution-5 results."""
    data = {iso3: _at(pd.read_parquet(_parquet(iso3)), res) for iso3 in COUNTRIES}
    for df in data.values():
        df["deficit"] = (df["D"] - df["A"]).clip(lower=0)
        df["ratio"] = 100 * df["A"] / df["D"].where(df["D"] > 0)
    # one log scale for demand and deficit across every country, 1 to 50,000 patients per year
    vmax = max(float(df["D"].max()) for df in data.values())
    norm = LogNorm(vmin=1.0, vmax=VMAX, clip=True)
    pct = plt.Normalize(0, 100)
    for iso3, df in data.items():
        _, crs, window = COUNTRIES[iso3]
        wide = iso3 in WIDE
        g = _hexes(df[df["D"] > 0], crs, window)
        c = pd.read_parquet(_FIG / f"appendix_{iso3}_centres.parquet")
        if window is not None:
            x0, y0, x1, y1 = window
            c = c[c["lon"].between(x0, x1) & c["lat"].between(y0, y1)]
        out = lambda k: f"appendix_{iso3}_{k}_res{res}.png"  # noqa: E731
        _map(g, "D", CM_DEFICIT, norm, "Demand, $D_h$ (patients per year)", out("demand"), wide)
        _mvm_map(g, c, crs, out("mvm"), wide)
        _map(g, "ratio", CM_ACCESS, pct, "Access ratio, $A_h/D_h$ (%)", out("ratio"), wide)
        _map(g, "deficit", CM_DEFICIT, norm, r"Deficit, $\Delta_h$ (patients per year)", out("deficit"), wide)
    print(f"res {res}: shared log scale 1 to {VMAX:,.0f} (largest hexagon demand {vmax:,.0f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="plot from saved parquets only")
    ap.add_argument("--res", type=int, nargs="+", default=[4, 5], help="display resolutions (default 4 5)")
    args = ap.parse_args()
    if not args.plot:
        for iso3 in COUNTRIES:
            if not _parquet(iso3).exists():
                compute(iso3)
    for res in args.res:
        plot(res)


if __name__ == "__main__":
    main()
