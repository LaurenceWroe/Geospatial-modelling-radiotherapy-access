"""
Figures, tables and headline numbers for the revised Results section, from the
outputs of scripts/global_results.py and scripts/verify_wawrzuta.py.

  figures/global_demand_facilities.png   D_h per res-3 hexagon with DIRAC facilities
  figures/global_nf_vs_fca.png           per-country A/D, nearest-first vs capped 3SFCA
  figures/table_income.tex               D, A_C, A_G, A (both rules) by income group
  figures/table_verification.tex         population within 60/120 min vs Wawrzuta et al.
  figures/results_numbers.json           every number quoted in the text

    python scripts/results_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_FIG = _ROOT / "figures"

from scripts.global_figures import CM_ACCESS, CRS, NO_DATA, _borders, _hex_gdf  # noqa: E402
from data.linacs import DIRAC_CSV  # noqa: E402

GROUPS = ["Low income", "Lower middle income", "Upper middle income", "High income"]
SHORT = dict(zip(GROUPS, ["Low", "Lower-middle", "Upper-middle", "High"]))
COLOURS = dict(zip(GROUPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # validated categorical order
CAP = 450.0
SHORT_NAME = {"PRK": "North Korea", "RUS": "Russia", "KOR": "South Korea", "COD": "DR Congo", "COG": "Congo", "LAO": "Laos",
              "IRN": "Iran", "SYR": "Syria", "TZA": "Tanzania", "VEN": "Venezuela", "VNM": "Viet Nam"}
INK, MUTED = "#222222", "#6b6b6b"


def _spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED, labelsize=8)


# DIRAC countries with no GLOBOCAN estimate: their centres are not modelled
NO_GLOBOCAN = {"Taiwan", "Hong Kong", "Macao", "Monaco", "Aruba", "Bermuda"}


def _facilities():
    """Modelled centres: at least one MVM, in a country with GLOBOCAN estimates."""
    fac = pd.read_csv(DIRAC_CSV).rename(columns={"He Photon And Electron Beam Rt": "n_linacs",
                                                 "Latitude": "lat", "Longitude": "lon"})
    fac = fac[(fac["n_linacs"] > 0) & fac["lat"].notna() & ~fac["Country"].isin(NO_GLOBOCAN)]
    return gpd.GeoDataFrame(fac, geometry=gpd.points_from_xy(fac["lon"], fac["lat"]), crs="EPSG:4326").to_crs(CRS)


def _hexmap(g, col, cmap, norm, label, name, borders, below_min=False):
    """One map; *below_min* labels the lowest decade of a log scale "<10^k",
    since values under it are clipped to the lowest colour."""
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    borders.plot(ax=ax, color=NO_DATA, linewidth=0)
    g.plot(ax=ax, column=col, cmap=cmap, norm=norm, linewidth=0)
    borders.boundary.plot(ax=ax, color="white", linewidth=0.15)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation="horizontal",
                      fraction=0.04, pad=0.01, label=label)
    if below_min:
        lo, hi = int(np.log10(norm.vmin)), int(np.floor(np.log10(norm.vmax)))
        cb.set_ticks([10.0 ** k for k in range(lo, hi + 1)],
                     labels=[f"$<10^{{{lo}}}$"] + [f"$10^{{{k}}}$" for k in range(lo + 1, hi + 1)])
    ax.set_axis_off()
    fig.savefig(_FIG / name, bbox_inches="tight")
    plt.close(fig)


# MVMs per hexagon: interval bins and a cyan-blue-purple ramp (no red/green/yellow)
MVM_BOUNDS = [1, 2, 5, 10, 20, 50, 1000]
MVM_RAMP = ["#9ff0ef", "#22b5cc", "#1e78dc", "#4b3bbf", "#6e1f93", "#3b0a52"]


def mvm_scale():
    """(cmap, BoundaryNorm, interval labels) for MVMs-per-hexagon maps."""
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("mvm", MVM_RAMP, N=len(MVM_BOUNDS) - 1)
    # one labelled interval per colour, e.g. "5–9"
    labels = [str(lo) if hi - lo == 1 else f"{lo}–{hi - 1}" for lo, hi in zip(MVM_BOUNDS[:-2], MVM_BOUNDS[1:-1])]
    labels.append(f"{MVM_BOUNDS[-2]}+")
    return cmap, BoundaryNorm(MVM_BOUNDS, cmap.N), labels


def mvm_colorbar(fig, ax, cmap, labels, labelsize=8, **kw):
    """Colour bar with one interval label centred on each colour."""
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, cmap.N)), ax=ax, **kw)
    cb.set_ticks(np.arange(cmap.N) + 0.5, labels=labels)
    cb.ax.tick_params(length=0, labelsize=labelsize)
    return cb


def facilities_map(fac, borders, res, name):
    """MVMs per hexagon at resolution *res*, white where there are none."""
    import h3
    fac = fac.assign(h3=[h3.latlng_to_cell(la, lo, res) for la, lo in zip(fac["lat"], fac["lon"])])
    fh = _hex_gdf(fac.groupby("h3", as_index=False)["n_linacs"].sum())
    cmap, bnorm, labels = mvm_scale()
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
    fh.plot(ax=ax, column="n_linacs", cmap=cmap, norm=bnorm, linewidth=0)
    borders.boundary.plot(ax=ax, color="black", linewidth=0.25)
    mvm_colorbar(fig, ax, cmap, labels, orientation="horizontal", fraction=0.04, pad=0.01, aspect=35,
                 shrink=0.8, label="Number of linacs")
    ax.set_axis_off()
    fig.savefig(_FIG / name, bbox_inches="tight")
    plt.close(fig)


def world_maps(hx, borders):
    """Demand, facilities, and capped-3SFCA access ratio and deficit, per res-3 hexagon."""
    from matplotlib.colors import LogNorm
    from scripts.global_figures import CM_ACCESS, CM_DEFICIT, LINAC_BLUE
    g = _hex_gdf(hx[hx["D"] > 0])
    # demand and deficit share one logarithmic scale
    norm = LogNorm(vmin=10, vmax=float(np.nanpercentile(g["D"], 99.5)), clip=True)
    _hexmap(g, "D", CM_DEFICIT, norm, "Radiotherapy demand per hexagon, $D_h$ (patients per year)",
            "global_demand.png", borders)
    fac = _facilities()
    facilities_map(fac, borders, 2, "global_facilities.png")
    facilities_map(fac, borders, 3, "global_facilities_res3.png")

    # capped 3SFCA access ratio and deficit, one image each (LaTeX subfigures)
    g["r"] = 100 * g["A_gauss_fca"] / g["D"]
    g["deficit"] = (g["D"] - g["A_gauss_fca"]).clip(lower=0)
    _hexmap(g, "r", CM_ACCESS, plt.Normalize(0, 100), "Access ratio, $A_h/D_h$ (%)",
            "global_ratio_fca.png", borders)
    _hexmap(g, "deficit", CM_DEFICIT, norm, "Deficit per hexagon, $\\Delta_h$ (patients per year)",
            "global_deficit_fca.png", borders, below_min=True)
    return int(len(fac)), float(fac["n_linacs"].sum())


def _spread(ys, gap, lo=2.0, hi=98.0):
    """Label heights near *ys* (sorted ascending) at least *gap* apart."""
    out = []
    for y in ys:
        out.append(max(y, out[-1] + gap) if out else max(y, lo))
    over = out[-1] - hi if out else 0
    return [y - max(over, 0) for y in out]


CASE_STUDIES = ["AUS", "IND", "NGA", "OMN", "GBR", "USA"]  # countries mapped in the appendix
LABELLED = CASE_STUDIES + ["CHN", "IDN", "JPN", "BRA", "MEX", "SAU", "RUS"]


def radmaps_vs_single(df, label=LABELLED):
    """RadMaps A/D against capacity-only and geography-only access, per country.

    One image per panel, for side-by-side subfigures in LaTeX.
    """
    from matplotlib.lines import Line2D
    d = df[(df["D"] > 0) & (df["n_machines"] > 0)].copy()
    d["a"] = 100 * d["A_gauss_fca"] / d["D"]
    panels = [("A_C", r"Capacity-only access, $A_\text{C}/D$ (%)", "global_radmaps_vs_capacity.png", True),
              ("A_G_gauss", r"Geography-only access, $A_\text{G}/D$ (%)", "global_radmaps_vs_geography.png", False)]
    for col, xl, name, legend in panels:
        d["x"] = 100 * d[col] / d["D"]
        fig, ax = plt.subplots(figsize=(3.5, 3.4), dpi=300)
        ax.plot([0, 100], [0, 100], color="#999999", lw=0.8, ls="--", zorder=1)
        for grp in GROUPS:
            s = d[d["income_group"] == grp]
            ax.scatter(s["x"], s["a"], s=12, color=COLOURS[grp], edgecolor="white", linewidth=0.5,
                       alpha=0.9, zorder=2)
        s = d[d["income_group"].isna()]
        ax.scatter(s["x"], s["a"], s=12, color="#9a9a9a", edgecolor="white", linewidth=0.5, alpha=0.9, zorder=2)
        # points lie on or below the diagonal, so each label sits in the empty region above it:
        # right-aligned just left of the diagonal at its (spread) height, never off the axis
        top = d[d["iso3"].isin(label)].sort_values("a")
        for (_, r), ly in zip(top.iterrows(), _spread([max(a, 20.0) for a in top["a"]], 5.5)):
            lab = SHORT_NAME.get(r["iso3"], r["name"].split(",")[0].split(" (")[0])
            xr = max(min(r["x"] - 6, ly - 5), 1.7 * len(lab) + 1)
            ax.annotate(lab, (r["x"], r["a"]), xytext=(xr, ly), textcoords="data", fontsize=6, color=INK,
                        ha="right", va="center", arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))
        ax.set_xlim(0, 102); ax.set_ylim(0, 102)
        ax.set_xlabel(xl, color=INK, fontsize=8)
        ax.set_ylabel("RadMaps access, $A/D$ (%)", color=INK, fontsize=8)
        _spines(ax)
        ax.tick_params(labelsize=7)
        if legend:
            handles = [Line2D([], [], marker="o", ls="", color=COLOURS[g], markersize=4, label=SHORT[g])
                       for g in GROUPS]
            ax.legend(handles=handles, title="Income group", frameon=False, fontsize=6.5, title_fontsize=6.5,
                      loc="upper left")
        fig.tight_layout()
        fig.savefig(_FIG / name)
        plt.close(fig)


def nf_vs_fca(df):
    d = df[(df["D"] > 0) & (df["n_machines"] > 0)].copy()
    d["nf"] = 100 * d["A_gauss_nf"] / d["D"]
    d["fca"] = 100 * d["A_gauss_fca"] / d["D"]
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=200)
    ax.plot([0, 100], [0, 100], color="#999999", lw=0.8, ls="--", zorder=1)
    for grp in GROUPS:
        s = d[d["income_group"] == grp]
        ax.scatter(s["nf"], s["fca"], s=np.clip(np.sqrt(s["D"]) / 3, 8, 260), color=COLOURS[grp],
                   edgecolor="white", linewidth=0.6, alpha=0.9, label=SHORT[grp], zorder=2)
    d["diff"] = d["fca"] - d["nf"]
    top = d.reindex(d["diff"].abs().sort_values(ascending=False).index).head(6)
    for k, (_, r) in enumerate(top.sort_values("nf").iterrows()):
        ax.annotate(r["iso3"], (r["nf"], r["fca"]), fontsize=7, color=INK,
                    xytext=(-14, 4 + 7 * (k % 2)), textcoords="offset points", ha="right",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))
    ax.set_xlim(0, 101); ax.set_ylim(0, 101)
    ax.set_xlabel("Nearest-first, $A/D$ (%)", color=INK)
    ax.set_ylabel("Capped 3SFCA, $A/D$ (%)", color=INK)
    _spines(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=COLOURS[g], markersize=6, label=SHORT[g]) for g in GROUPS]
    ax.legend(handles=handles, title="Income group", frameon=False, fontsize=8, title_fontsize=8,
              loc="lower right")
    ax.text(2, 97, "Marker area ∝ national demand $D$", fontsize=7, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(_FIG / "global_nf_vs_fca.png")
    plt.close(fig)
    return d


def _pct(x):
    return f"{100 * x:.1f}"


def _int(x):
    return f"{x:,.0f}".replace(",", "{,}")


def income_table(df):
    rows = []
    for grp in GROUPS + ["World"]:
        d = df if grp == "World" else df[df["income_group"] == grp]
        D = d["D"].sum()
        gap = ((d["D"] - d["A_C"]).clip(lower=0) / CAP).sum()
        rows.append(dict(group="World" if grp == "World" else SHORT[grp], n=len(d), D=D,
                         A_C=d["A_C"].sum() / D, A_G=d["A_G_gauss"].sum() / D,
                         A_nf=d["A_gauss_nf"].sum() / D, A_fca=d["A_gauss_fca"].sum() / D, gap=gap))
    t = pd.DataFrame(rows)
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Income group & Countries & $D$ (k/yr) & $A_\text{C}/D$ (\%) & $A_\text{G}/D$ (\%) & "
             r"$A/D$ NF (\%) & $A/D$ 3SFCA (\%) & Linac gap \\", r"\midrule"]
    for _, r in t.iterrows():
        if r["group"] == "World":
            lines.append(r"\midrule")
        lines.append(f"{r['group']} & {r['n']} & {_int(r['D'] / 1e3)} & {_pct(r['A_C'])} & "
                     f"{_pct(r['A_G'])} & {_pct(r['A_nf'])} & {_pct(r['A_fca'])} & {_int(r['gap'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (_FIG / "table_income.tex").write_text("\n".join(lines) + "\n")
    return t


def verification_table():
    g = pd.read_csv(_FIG / "verification_wawrzuta_groups.csv")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Within 60 min} & \multicolumn{2}{c}{Within 120 min} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Income group & Countries & RadMaps (\%) & Wawrzuta (\%) & RadMaps (\%) & Wawrzuta (\%) \\", r"\midrule"]
    for _, r in g.iterrows():
        name = r["group"] if r["group"] == "World" else SHORT[r["group"]]
        if name == "World":
            lines.append(r"\midrule")
        lines.append(f"{name} & {r['n']} & {r['radmaps_60']:.1f} & {r['wawrzuta_60']:.1f} & "
                     f"{r['radmaps_120']:.1f} & {r['wawrzuta_120']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (_FIG / "table_verification.tex").write_text("\n".join(lines) + "\n")
    return g


# Moraes et al. 2025 (Lancet Oncol 26:239-48), Table 1: patients for radiotherapy in 2022
# and LINACs in 2020-21, per World Bank income group (2022 classification)
MORAES = {"Low income": (236_681, 38), "Lower middle income": (1_526_164, 1_226),
          "Upper middle income": (3_633_239, 3_763), "High income": (3_629_536, 9_765),
          "World": (9_055_621, 14_792)}


def comparison_table(df):
    """Capacity-only (Moraes) and geography-only (Wawrzuta, 120 min) against RadMaps.

    The capacity pair is grouped by the FY2025 World Bank classification, which
    reproduces Moraes et al.'s group sizes (26/49/48/58); everything else uses the
    current classification in data/wb_income_groups.csv.
    """
    w = pd.read_csv(_FIG / "verification_wawrzuta_groups.csv").set_index("group")
    fy25 = df["iso3"].map(pd.read_csv(_ROOT / "data" / "wb_income_groups_fy25.csv").set_index("iso3")["income_group"])
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Capacity only$^*$} & \multicolumn{2}{c}{Geography only} & \multicolumn{3}{c}{RadMaps} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"\makecell[l]{Income\\group} & \makecell[r]{Ref.~\cite{Moraes_2025}\\(\%)} & \makecell[r]{RadMaps\\(\%)} & "
             r"\makecell[r]{Ref.~\cite{Wawrzuta_2025}\\(\%)} & \makecell[r]{RadMaps\\(\%)} & "
             r"\makecell[r]{$A_\text{C}/D$\\(\%)} & \makecell[r]{$A_\text{G}/D$\\(\%)} & \makecell[r]{$A/D$\\(\%)} \\",
             r"\midrule"]
    rows = []
    for grp in GROUPS + ["World"]:
        s = df if grp == "World" else df[df["income_group"] == grp]
        D = s["D"].sum()
        c = df if grp == "World" else df[fy25 == grp]
        pm, nm = MORAES[grp]
        r = dict(group=grp, moraes=100 * CAP * nm / pm, rm_pooled=100 * CAP * c["n_machines"].sum() / c["D"].sum(),
                 wz_120=w.loc[grp, "wawrzuta_120"], rm_120=w.loc[grp, "radmaps_120"],
                 A_C=100 * s["A_C"].sum() / D, A_G=100 * s["A_G_gauss"].sum() / D, A=100 * s["A_gauss_fca"].sum() / D)
        rows.append(r)
        if grp == "World":
            lines.append(r"\midrule")
        lines.append(f"{SHORT.get(grp, grp)} & " + " & ".join(f"{r[k]:.1f}" for k in
                     ("moraes", "rm_pooled", "wz_120", "rm_120", "A_C", "A_G", "A")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (_FIG / "table_comparison.tex").write_text("\n".join(lines) + "\n")
    return pd.DataFrame(rows)


def national_choropleth(df, borders):
    """National access A/D (capped 3SFCA) on the continuous access scale of the hexagon maps."""
    m = borders.merge(df[["iso3", "D", "A_gauss_fca"]], on="iso3", how="left")
    m["v"] = 100 * m["A_gauss_fca"] / m["D"].where(m["D"] > 0)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    m[m["v"].isna()].plot(ax=ax, color=NO_DATA, linewidth=0.15, edgecolor="white")
    m[m["v"].notna()].plot(ax=ax, column="v", cmap=CM_ACCESS, vmin=0, vmax=100, linewidth=0.15, edgecolor="white")
    fig.colorbar(plt.cm.ScalarMappable(cmap=CM_ACCESS, norm=plt.Normalize(0, 100)), ax=ax, orientation="horizontal",
                 fraction=0.04, pad=0.01, label="National access ratio, $A/D$ (%)")
    ax.set_axis_off()
    fig.savefig(_FIG / "global_choropleth_access.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(_FIG / "global_results.csv")
    df = df[df["D"] > 0]
    hx = pd.read_parquet(_FIG / "global_hex_res3.parquet").groupby("h3", as_index=False).sum(numeric_only=True)
    borders = _borders()
    n_fac, n_mac = world_maps(hx, borders)
    national_choropleth(df, borders)
    radmaps_vs_single(df)
    d = nf_vs_fca(df)
    t = income_table(df)
    v = verification_table()
    comparison_table(df)
    vc = pd.read_csv(_FIG / "verification_wawrzuta.csv")

    f = df.copy()
    for c in ("A_C", "A_G_gauss", "A_gauss_nf", "A_gauss_fca"):
        f[c + "_f"] = f[c] / f["D"]
    f["bound"] = np.minimum(f["A_C_f"], f["A_G_gauss_f"])
    f["loss"] = f["bound"] - f["A_gauss_nf_f"]
    D = f["D"].sum()
    from scipy import stats
    nums = {
        "n_countries": int(len(f)), "n_no_facility": int((f["n_machines"] == 0).sum()),
        "D_total": D, "population": float(f["population"].sum()),
        "n_facilities": n_fac, "n_machines": n_mac,
        "A_C": f["A_C"].sum() / D, "A_G": f["A_G_gauss"].sum() / D, "A_G_step": f["A_G_step"].sum() / D,
        "A_nf": f["A_gauss_nf"].sum() / D, "A_fca": f["A_gauss_fca"].sum() / D,
        "A_step_nf": f["A_step_nf"].sum() / D,
        "bound_sum": (f["bound"] * f["D"]).sum() / D, "loss_patients": float((f["loss"] * f["D"]).sum()),
        "deficit_nf": D - f["A_gauss_nf"].sum(), "linac_gap": float(((f["D"] - f["A_C"]).clip(lower=0) / CAP).sum()),
        "no_facility_D": float(f.loc[f["n_machines"] == 0, "D"].sum()),
        "bound_violations": int((f["A_gauss_nf_f"] > f["bound"] + 1e-6).sum()),
        "n_capacity_limited": int((f["A_C_f"] < f["A_G_gauss_f"] - 0.05).sum()),
        "n_geography_limited": int((f["A_G_gauss_f"] < f["A_C_f"] - 0.05).sum()),
        "n_balanced": int(((f["A_C_f"] - f["A_G_gauss_f"]).abs() <= 0.05).sum()),
        "top_loss": f.sort_values("loss", ascending=False)[["name", "A_C_f", "A_G_gauss_f", "A_gauss_nf_f", "loss"]].head(6).round(3).values.tolist(),
        "nf_fca_median_diff_pt": float(d["diff"].median()), "nf_fca_n_over2pt": int((d["diff"].abs() > 2).sum()),
        "nf_fca_top": d.reindex(d["diff"].abs().sort_values(ascending=False).index)[["name", "nf", "fca", "diff"]].head(6).round(1).values.tolist(),
        "nf_fca_spearman": float(stats.spearmanr(d["nf"], d["fca"])[0]),
        "verif_n": int(len(vc)), "verif_r120": float(stats.pearsonr(vc.rm_120, vc.wz_120)[0]),
        "verif_r60": float(stats.pearsonr(vc.dropna(subset=["rm_60", "wz_60"]).rm_60, vc.dropna(subset=["rm_60", "wz_60"]).wz_60)[0]),
        "verif_mad120": float((vc.rm_120 - vc.wz_120).abs().median()),
        "verif_mad60": float((vc.rm_60 - vc.wz_60).abs().median()),
        "verif_within10_120": float(((vc.rm_120 - vc.wz_120).abs() <= 10).mean()),
        "verif_most_negative": vc.sort_values("diff_120")[["name", "rm_120", "wz_120"]].head(6).round(1).values.tolist(),
        "income": t.round(4).to_dict("records"), "verification_groups": v.round(1).to_dict("records"),
    }
    (_FIG / "results_numbers.json").write_text(json.dumps(nums, indent=1, default=float))
    print(json.dumps({k: v for k, v in nums.items() if k not in ("income", "verification_groups")}, indent=1, default=float))
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
