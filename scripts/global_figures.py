"""
Figures for the global results (reads scripts/global_results.py outputs).

  figures/global_choropleth_<cfg>.png   A_C/D, A_G/D and A/D by country
  figures/global_hex_<cfg>.png          A_h/D_h and deficit per res-3 hexagon
  figures/global_scatter_<cfg>.png      A_C/D vs A_G/D, coloured by A/D

<cfg> is gauss_nf (default), gauss_fca, step_nf or step_fca.

    python scripts/global_figures.py [--cfg gauss_nf gauss_fca]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pycountry  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_FIG = _ROOT / "figures"
CRS = "EPSG:8857"  # Equal Earth
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# The RadMaps dashboard's scheme (app.py _rdylgn_rgb, _DISCRETE_PALETTE_RDYLGN, _LINAC_BLUE):
# access red (low) -> green (high); deficit and demand the same ramp reversed.
RDYLGN = ["#dc3232", "#e6781e", "#f0c81e", "#50be3c", "#1e781e"]
BANDS = ["#dc3232", "#e6781e", "#f0c81e", "#3cb43c", "#1e781e"]   # 0-20, ..., 80-100 %
LINAC_BLUE = "#1e78dc"
CM_ACCESS = LinearSegmentedColormap.from_list("access", RDYLGN)
CM_DEFICIT = LinearSegmentedColormap.from_list("deficit", RDYLGN[::-1])
NO_DATA = "#d9d9d6"


def _style(ax):
    ax.set_axis_off()


def _borders():
    g = gpd.read_file(_ROOT / "assets" / "country_borders_50m.geojson")

    def iso3(a2):
        c = pycountry.countries.get(alpha_2=a2) if isinstance(a2, str) else None
        return c.alpha_3 if c else None

    g["iso3"] = g["iso_a2"].map(iso3)
    return g[~g["iso_a2"].isin(["AQ"])].to_crs(CRS)


def choropleth(df, cfg, borders):
    m = borders.merge(df, on="iso3", how="left")
    panels = [("A_C", "Capacity only, $A_\\mathrm{C}/D$"),
              (f"A_G_{cfg.split('_')[0]}", "Geography only, $A_\\mathrm{G}/D$"),
              (f"A_{cfg}", "RadMaps, $A/D$")]
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap, norm = ListedColormap(BANDS), BoundaryNorm([0, 20, 40, 60, 80, 100.01], len(BANDS))
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), dpi=150)
    for ax, (col, title) in zip(axes, panels):
        m["v"] = 100 * m[col] / m["D"]
        m[m["v"].isna()].plot(ax=ax, color=NO_DATA, linewidth=0.2, edgecolor="white")
        m[m["v"].notna()].plot(ax=ax, column="v", cmap=cmap, norm=norm,
                               linewidth=0.2, edgecolor="white")
        ax.set_title(title, loc="left", fontsize=11)
        _style(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.025, pad=0.02, ticks=[0, 20, 40, 60, 80, 100],
                 label="Share of radiotherapy demand met (%)")
    fig.savefig(_FIG / f"global_choropleth_{cfg}.png", bbox_inches="tight")
    plt.close(fig)


def _hex_gdf(hx):
    polys, keep = [], []
    for i, c in enumerate(hx["h3"]):
        b = [(lon, lat) for lat, lon in h3.cell_to_boundary(c)]
        lons = [p[0] for p in b]
        if max(lons) - min(lons) > 180:  # skip cells split by the antimeridian
            continue
        polys.append(Polygon(b)); keep.append(i)
    return gpd.GeoDataFrame(hx.iloc[keep].reset_index(drop=True), geometry=polys, crs="EPSG:4326").to_crs(CRS)


def hexmap(hx, cfg, borders):
    g = _hex_gdf(hx[hx["D"] > 0])
    g["ratio"] = 100 * g[f"A_{cfg}"] / g["D"]
    g["deficit"] = g["D"] - g[f"A_{cfg}"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.4), dpi=150)
    for ax, col, cmap, label in [
        (axes[0], "ratio", CM_ACCESS, "Access ratio, $A_h/D_h$ (%)"),
        (axes[1], "deficit", CM_DEFICIT, "Deficit, $\\Delta_h$ (patients per year)"),
    ]:
        borders.plot(ax=ax, color=NO_DATA, linewidth=0)
        vmax = 100 if col == "ratio" else float(np.nanpercentile(g[col], 99))
        g.plot(ax=ax, column=col, cmap=cmap, vmin=0, vmax=vmax, linewidth=0)
        borders.boundary.plot(ax=ax, color="white", linewidth=0.15)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
        fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.01, label=label)
        _style(ax)
    fig.savefig(_FIG / f"global_hex_{cfg}.png", bbox_inches="tight")
    plt.close(fig)


def scatter(df, cfg):
    d = df[df["D"] > 0].copy()
    x = 100 * d["A_C"] / d["D"]
    y = 100 * d[f"A_G_{cfg.split('_')[0]}"] / d["D"]
    c = 100 * d[f"A_{cfg}"] / d["D"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    sc = ax.scatter(x, y, c=c, cmap=CM_ACCESS, vmin=0, vmax=100, s=np.clip(np.sqrt(d["D"]) / 3, 6, 260),
                    edgecolor="#555", linewidth=0.3)
    ax.plot([0, 100], [0, 100], color="#888", lw=0.8, ls="--")
    for _, r in d.nlargest(12, "D").iterrows():
        ax.annotate(r["iso3"], (100 * r["A_C"] / r["D"], 100 * r[f"A_G_{cfg.split('_')[0]}"] / r["D"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.set_xlabel("Capacity only, $A_\\mathrm{C}/D$ (%)")
    ax.set_ylabel("Geography only, $A_\\mathrm{G}/D$ (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.colorbar(sc, ax=ax, label="RadMaps, $A/D$ (%)")
    fig.tight_layout()
    fig.savefig(_FIG / f"global_scatter_{cfg}.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", nargs="*", default=["gauss_nf", "gauss_fca"])
    args = ap.parse_args()
    df = pd.read_csv(_FIG / "global_results.csv")
    hx = pd.read_parquet(_FIG / "global_hex_res3.parquet")
    hx = hx.groupby("h3", as_index=False).sum(numeric_only=True)  # border cells shared by countries
    borders = _borders()
    for cfg in args.cfg:
        choropleth(df, cfg, borders)
        hexmap(hx, cfg, borders)
        scatter(df, cfg)
        print(f"{cfg}: figures written", flush=True)


if __name__ == "__main__":
    main()
