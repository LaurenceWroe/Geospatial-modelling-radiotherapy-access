"""
Figure and table for the global sensitivity analysis, from
figures/sensitivity_global.csv (scripts/sensitivity_global.py).

  figures/sensitivity_global_{capacity,time,allocation,decay}.png\n                                    global A/D against each input, with A_C/D or A_G/D
  figures/table_sensitivity.tex     global and income-group A/D for every configuration

    python scripts/sensitivity_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_FIG = _ROOT / "figures"

from scripts.results_figures import GROUPS, INK, MUTED, SHORT, _spines  # noqa: E402

C_A, C_C, C_G = "#222222", "#2a78d6", "#eb6834"  # combined, capacity-only, geography-only
ORDER = [("Capacity", "Capacity (patients per linac per year)",
          [("cap250", "250"), ("cap350", "350"), ("base", "450"), ("cap550", "550")]),
         ("Time limit", "Gaussian time limit (min)",
          [("t30", "30"), ("t60", "60"), ("t90", "90"), ("base", "120")]),
         ("Allocation", "Allocation rule", [("nf", "Nearest-first"), ("base", "Capped 3SFCA")]),
         ("Decay", "Decay model (120 min cut-off)",
          [("step", "Step"), ("exponential", "Exponential"), ("base", "Gaussian"), ("power", "Power")])]


def _pct(df):
    g = df.groupby("config")[["D", "A_C", "A_G", "A"]].sum()
    return 100 * g[["A_C", "A_G", "A"]].div(g["D"], axis=0)


PANEL = {"Capacity": "capacity", "Time limit": "time", "Allocation": "allocation", "Decay": "decay"}


def figure(df):
    """One image per input (LaTeX subfigures), all on the same 0-100 % axis."""
    p = _pct(df)
    for title, xl, items in ORDER:
        fig, ax = plt.subplots(figsize=(3.3, 2.6), dpi=300)
        x = np.arange(len(items))
        cfg = [c for c, _ in items]
        # ordered inputs are joined by lines; categories (allocation, decay) are points only
        ls = "-" if title in ("Capacity", "Time limit") else "none"
        ax.plot(x, p.loc[cfg, "A"], marker="o", ls=ls, color=C_A, lw=1.4, ms=4, label="$A/D$", zorder=3)
        if title == "Capacity":
            ax.plot(x, p.loc[cfg, "A_C"], marker="s", ls="--", color=C_C, lw=1, ms=3, label=r"$A_\mathrm{C}/D$")
        if title in ("Time limit", "Decay"):
            ax.plot(x, p.loc[cfg, "A_G"], marker="^", ls="--" if ls == "-" else "none", color=C_G, lw=1, ms=3,
                    label=r"$A_\mathrm{G}/D$")
        b = [i for i, c in enumerate(cfg) if c == "base"][0]
        ax.plot([b], [p.loc["base", "A"]], "o", ms=8, mfc="none", mec=C_A, mew=1.0, zorder=4)
        for xi, c in zip(x, cfg):
            ax.annotate(f"{p.loc[c, 'A']:.1f}", (xi, p.loc[c, "A"]), xytext=(0, -10), textcoords="offset points",
                        ha="center", fontsize=6.5, color=INK)
        ax.set_xticks(x, [l for _, l in items], fontsize=7)  # horizontal, so all four panels align
        ax.set_xlim(-0.5, len(items) - 0.5)
        ax.set_ylim(0, 100)
        ax.set_xlabel(xl, fontsize=7.5, color=MUTED)
        ax.set_ylabel("Share of global demand (%)", fontsize=7.5, color=INK)
        _spines(ax)
        ax.tick_params(labelsize=7)
        if title != "Allocation":
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(_FIG / f"sensitivity_global_{PANEL[title]}.png")
        plt.close(fig)


def table(df, wb):
    df = df.merge(wb, on="iso3", how="left")
    cols = [("World", None)] + [(SHORT[g], g) for g in GROUPS]
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Input & Value & World & " + " & ".join(c for c, _ in cols[1:]) + r" \\", r"\midrule"]
    for k, (title, _, items) in enumerate(ORDER):
        for j, (c, lab) in enumerate(items):
            vals = []
            for _, g in cols:
                s = df[df["config"] == c] if g is None else df[(df["config"] == c) & (df["income_group"] == g)]
                vals.append(100 * s["A"].sum() / s["D"].sum())
            cell = [f"\\textbf{{{v:.1f}}}" if c == "base" else f"{v:.1f}" for v in vals]
            lines.append(f"{title if j == 0 else ''} & {lab} & " + " & ".join(cell) + r" \\")
        if k < len(ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (_FIG / "table_sensitivity.tex").write_text("\n".join(lines) + "\n")


def resolution_figure():
    """Global A/D and A_G/D against the H3 resolution of the calculation
    (figures/sensitivity_resolution.csv, scripts/sensitivity_resolution.py).
    Resolution 6 is not shown: China and the United States are too large to
    compute there, so it has no global value (the printout gives the rest)."""
    d = pd.read_csv(_FIG / "sensitivity_resolution.csv")
    skip = set(d.loc[d["skipped"], "iso3"])

    def pct(df):
        g = df.groupby("res")[["D", "A_G", "A"]].sum(min_count=1)
        return 100 * g[["A_G", "A"]].div(g["D"], axis=0)

    full = pct(d[d["res"] < 6])
    part = pct(d[~d["iso3"].isin(skip)])
    fig, ax = plt.subplots(figsize=(3.3, 2.6), dpi=300)
    ax.plot(full.index, full["A"], marker="o", color=C_A, lw=1.4, ms=4, label="$A/D$", zorder=3)
    ax.plot(full.index, full["A_G"], marker="^", ls="--", color=C_G, lw=1, ms=3, label=r"$A_\mathrm{G}/D$")
    ax.plot([5], [full.loc[5, "A"]], "o", ms=8, mfc="none", mec=C_A, mew=1.0, zorder=4)
    for r, v in full["A"].items():
        ax.annotate(f"{v:.1f}", (r, v), xytext=(0, -10), textcoords="offset points", ha="center",
                    fontsize=6.5, color=INK)
    ax.set_xticks(list(full.index), [str(r) for r in full.index], fontsize=7)
    ax.set_xlim(full.index.min() - 0.5, full.index.max() + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("H3 resolution", fontsize=7.5, color=MUTED)
    ax.set_ylabel("Share of global demand (%)", fontsize=7.5, color=INK)
    _spines(ax)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(_FIG / "sensitivity_global_resolution.png")
    plt.close(fig)
    print("resolution, all countries:\n", full.round(1).to_string(), "\nexcl.", sorted(skip), ":\n",
          part.round(1).to_string())


def main() -> None:
    df = pd.read_csv(_FIG / "sensitivity_global.csv")
    wb = pd.read_csv(_ROOT / "inputs" / "wb_income_groups.csv")[["iso3", "income_group"]]
    figure(df)
    if (_FIG / "sensitivity_resolution.csv").exists():
        resolution_figure()
    table(df, wb)
    print((_pct(df)).round(1).to_string())


if __name__ == "__main__":
    main()
