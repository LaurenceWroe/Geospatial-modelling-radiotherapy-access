"""
United Kingdom case study for the Results: the paper's main configuration
(Gaussian, 120-min driving catchment from the res-6 store, 450 patients per
linac, CCORE 2012 optimal RTU) at H3 resolution 6, both allocation rules, with
resolution 5 for comparison.

  figures/uk_maps.png             access ratio (nearest-first, capped 3SFCA) and deficit
  figures/uk_numbers.json         numbers quoted in the text

    python scripts/uk_case_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_FIG = _ROOT / "figures"

from analysis.accessibility import compute_accessibility  # noqa: E402
from data.cancer import DERIVED_CANCER_TYPES, apportion_cancer_to_h3, get_cancer_types  # noqa: E402
from data.linacs import load_linacs_from_dirac_db  # noqa: E402
from data.population import load_population_at_resolution  # noqa: E402
from data.tt_store import store_matrix  # noqa: E402
from scripts.global_figures import CM_ACCESS, CM_DEFICIT, LINAC_BLUE  # noqa: E402

CAP, T_MAX, CRS = 450.0, 120.0, "EPSG:27700"
_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}


def run(res):
    gdf = load_population_at_resolution("United Kingdom", target_resolution=res)
    cg = apportion_cancer_to_h3(gdf, "GBR", get_cancer_types() + DERIVED_CANCER_TYPES,
                                use_actual_rt=False, rtu_set="barton_2014")
    cols = [c for c in cg.columns if c.endswith("_optimal_rt") and c[:-11].strip().lower() not in _AGG]
    dem = cg[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)
    locs, _ = load_linacs_from_dirac_db("United Kingdom")
    tt = store_matrix("GBR", list(gdf["h3"]), locs)
    out = {}
    for rule in ("nearest_first", "proportional"):
        g, s = compute_accessibility(gdf, locs, model="gaussian", max_distance_km=T_MAX,
                                     capacity_per_machine_per_year=CAP, demand=dem, h3_resolution=res,
                                     travel_time_matrix=tt, allocation=rule)
        out[rule] = (g, s)
    return gdf, dem, locs, out


def hexes(g):
    polys = [Polygon([(lo, la) for la, lo in h3.cell_to_boundary(c)]) for c in g["h3"]]
    return gpd.GeoDataFrame(g.drop(columns="geometry", errors="ignore"), geometry=polys, crs="EPSG:4326").to_crs(CRS)


def main() -> None:
    gdf, dem, locs, out = run(6)
    D = float(dem.sum())
    g_nf, s_nf = out["nearest_first"]
    g_fc, s_fc = out["proportional"]
    n_mac = float(sum(w for *_, w in locs))
    a_g = float((g_nf["access_probability"] * g_nf["rt_demand"]).sum())
    near = g_nf["nearest_linac_min"].to_numpy(float)
    pop = g_nf["population"].to_numpy(float)
    ratio = np.where(dem > 0, g_nf["rt_treated"] / np.maximum(dem, 1e-9), np.nan)
    deficit = g_nf["rt_untreated"].to_numpy(float)
    nums = {
        "res": 6, "n_hex": len(gdf), "population": float(pop.sum()), "D": D,
        "n_facilities": len(locs), "n_machines": n_mac, "capacity": n_mac * CAP,
        "A_C": min(D, n_mac * CAP) / D, "A_G": a_g / D,
        "A_nf": s_nf["total_rt_treated"] / D, "A_fca": s_fc["total_rt_treated"] / D,
        "deficit_nf": D - s_nf["total_rt_treated"], "deficit_fca": D - s_fc["total_rt_treated"],
        "linac_gap_capacity": max(D - n_mac * CAP, 0) / CAP,
        "linac_gap_nf": (D - s_nf["total_rt_treated"]) / CAP,
        "pop_within_60": float(pop[np.nan_to_num(near, nan=1e9) <= 60].sum() / pop.sum()),
        "pop_within_120": float(pop[np.nan_to_num(near, nan=1e9) <= 120].sum() / pop.sum()),
        "median_near_min": float(np.nanmedian(near)),
        "share_deficit_in_ratio_below_50": float(deficit[np.nan_to_num(ratio, nan=1) < 0.5].sum() / deficit.sum()),
        "share_pop_ratio_below_50": float(pop[np.nan_to_num(ratio, nan=1) < 0.5].sum() / pop.sum()),
    }
    ratio_fc = np.where(dem > 0, g_fc["rt_treated"] / np.maximum(dem, 1e-9), np.nan)
    deficit_fc = g_fc["rt_untreated"].to_numpy(float)
    r = np.nan_to_num(ratio_fc, nan=1)
    nums["fca"] = {
        "linac_gap": (D - s_fc["total_rt_treated"]) / CAP,
        "share_deficit_in_ratio_below_50": float(deficit_fc[r < 0.5].sum() / deficit_fc.sum()),
        "share_pop_ratio_below_50": float(pop[r < 0.5].sum() / pop.sum()),
        "share_pop_ratio_65_85": float(pop[(r >= 0.65) & (r <= 0.85)].sum() / pop.sum()),
        "pop_weighted_ratio_p10_p50_p90": [float(np.interp(q, np.cumsum(pop[np.argsort(r)]) / pop.sum(), np.sort(r)))
                                           for q in (0.1, 0.5, 0.9)],
    }
    _, dem5, _, out5 = run(5)
    nums["res5"] = {"A_nf": out5["nearest_first"][1]["total_rt_treated"] / dem5.sum(),
                    "A_fca": out5["proportional"][1]["total_rt_treated"] / dem5.sum(),
                    "A_G": float((out5["nearest_first"][0]["access_probability"] * dem5).sum() / dem5.sum())}

    h = hexes(g_nf)
    h["nf"] = 100 * ratio
    h["fca"] = 100 * np.where(dem > 0, g_fc["rt_treated"].to_numpy() / np.maximum(dem, 1e-9), np.nan)
    h["deficit"] = deficit
    h = h[dem > 0]
    fac = gpd.GeoDataFrame(geometry=gpd.points_from_xy([l[1] for l in locs], [l[0] for l in locs]),
                           crs="EPSG:4326").to_crs(CRS)
    def maps(panels, name, size):
        fig, axes = plt.subplots(1, len(panels), figsize=size, dpi=200)
        for ax, (col, cmap, norm, title) in zip(axes, panels):
            h.plot(ax=ax, column=col, cmap=cmap, norm=norm, linewidth=0)
            fac.plot(ax=ax, markersize=5, color=LINAC_BLUE, marker="^", edgecolor="white", linewidth=0.3)
            fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                         fraction=0.045, pad=0.02).ax.tick_params(labelsize=7)
            ax.set_title(title, fontsize=8.5, loc="left")
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(_FIG / name, bbox_inches="tight")
        plt.close(fig)

    h["deficit_fca"] = deficit_fc[dem > 0]
    pct = plt.Normalize(0, 100)
    # capped 3SFCA in the main text; nearest-first for the appendix
    maps([("fca", CM_ACCESS, pct, "(a) Access ratio, $A_h/D_h$ (%)"),
          ("deficit_fca", CM_DEFICIT, plt.Normalize(0, float(np.nanpercentile(h["deficit_fca"], 99))),
           "(b) Deficit, $\\Delta_h$ (patients/yr)")], "uk_maps.png", (7, 5.4))
    maps([("nf", CM_ACCESS, pct, "(a) Access ratio, $A_h/D_h$ (%)"),
          ("deficit", CM_DEFICIT, plt.Normalize(0, float(np.nanpercentile(h["deficit"], 99))),
           "(b) Deficit, $\\Delta_h$ (patients/yr)")], "uk_maps_nf.png", (7, 5.4))
    (_FIG / "uk_numbers.json").write_text(json.dumps(nums, indent=1))
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()
