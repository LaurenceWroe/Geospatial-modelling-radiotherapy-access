"""
Verification against Wawrzuta et al. (2025, Radiother Oncol 211:111061):
share of each country's population within 60 and 120 min driving time of its
nearest domestic radiotherapy centre.

Inputs
  figures/global_results.csv            RadMaps (scripts/global_results.py)
  data/wawrzuta_2025_country.csv        their per-country results (GitHub
                                        dwawrzuta/radiotherapyaccess, data_table.rds)
  data/wb_income_groups.csv             World Bank income groups

Outputs
  figures/verification_wawrzuta.csv           per-country comparison
  figures/verification_wawrzuta_groups.csv    by income group and worldwide
  figures/verification_wawrzuta.png           scatter, 60 and 120 min

Their published aggregates (World Bank 2023 groups) are printed alongside.
Income-group aggregates for both models are weighted by Kontur population so
that only the access estimate differs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_FIG = _ROOT / "figures"
GROUPS = ["Low income", "Lower middle income", "Upper middle income", "High income"]
PUBLISHED_120 = {"Low income": 17.1, "Lower middle income": 73.4, "Upper middle income": 85.0,
                 "High income": 93.1, "World": 76.3}
PUBLISHED_60 = {"World": 56.8}


def main() -> None:
    rm = pd.read_csv(_FIG / "global_results.csv")
    wz = pd.read_csv(_ROOT / "data" / "wawrzuta_2025_country.csv").dropna(subset=["iso3"])
    for t in (60, 120):
        rm[f"rm_{t}"] = 100 * rm[f"pop_{t}"] / rm["population"]
    df = rm.merge(wz[["iso3", "pop_60", "pop_120"]].rename(columns={"pop_60": "wz_60", "pop_120": "wz_120"}),
                  on="iso3", how="inner")
    df = df[["iso3", "name", "income_group", "time_source", "population", "n_facilities",
             "rm_60", "wz_60", "rm_120", "wz_120"]]
    df["diff_120"] = df["rm_120"] - df["wz_120"]
    df.sort_values("diff_120").to_csv(_FIG / "verification_wawrzuta.csv", index=False)

    print(f"{len(df)} countries matched ({(df.time_source == 'proxy').sum()} with the distance proxy)")
    for t in (60, 120):
        d = df.dropna(subset=[f"rm_{t}", f"wz_{t}"])
        r = stats.pearsonr(d[f"rm_{t}"], d[f"wz_{t}"])[0]
        rho = stats.spearmanr(d[f"rm_{t}"], d[f"wz_{t}"])[0]
        mad = (d[f"rm_{t}"] - d[f"wz_{t}"]).abs()
        print(f"{t:>3} min: Pearson r = {r:.3f}, Spearman rho = {rho:.3f}, "
              f"median |diff| = {mad.median():.1f} pt, {(mad <= 10).mean():.0%} within 10 pt")

    rows = []
    for grp in GROUPS + ["World"]:
        d = df if grp == "World" else df[df.income_group == grp]
        w = d["population"]
        rows.append({"group": grp, "n": len(d),
                     "radmaps_60": np.average(d.rm_60, weights=w), "wawrzuta_60": np.average(d.wz_60, weights=w),
                     "radmaps_120": np.average(d.rm_120, weights=w), "wawrzuta_120": np.average(d.wz_120, weights=w),
                     "published_120": PUBLISHED_120.get(grp), "published_60": PUBLISHED_60.get(grp)})
    g = pd.DataFrame(rows)
    g.to_csv(_FIG / "verification_wawrzuta_groups.csv", index=False)
    print(g.round(1).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=150, sharey=True)
    colours = dict(zip(GROUPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # validated categorical order
    for ax, t in zip(axes, (60, 120)):
        for grp in GROUPS:
            d = df[df.income_group == grp]
            ax.scatter(d[f"wz_{t}"], d[f"rm_{t}"], s=np.clip(np.sqrt(d.population) / 150, 8, 200),
                       color=colours[grp], alpha=0.75, edgecolor="white", linewidth=0.5, label=grp)
        ax.plot([0, 100], [0, 100], color="#888", lw=1, ls="--")
        ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
        ax.set_xlabel(f"Wawrzuta et al. 2025: population within {t} min (%)")
        ax.set_title(f"{t} min", loc="left", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("RadMaps: population within budget (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(_FIG / "verification_wawrzuta.png")


if __name__ == "__main__":
    main()
