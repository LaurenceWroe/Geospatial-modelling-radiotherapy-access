"""
Publication figure: supply vs. access bottleneck for national radiotherapy.

Two panels, all quantities normalised by national RT demand D so countries of
very different size are comparable:

  (a) Scatter  A_C/D (supply adequacy)  vs  A_G/D (geographic reach, 2 h drive).
      Marker colour = A_RM/D (realised treatment). Because A_RM <= min(A_C, A_G),
      position shows the *binding constraint* and colour the *realised outcome*.

  (b) Demand-normalised bullet bars: the two ceilings (A_C, A_G) and the realised
      A_RM. Where the A_RM bar stops short of the nearer ceiling, the red arrow is
      the SPATIAL MISMATCH — supply and reachable demand located in different
      places (A_RM < min(A_C, A_G)). India is the exemplar.

Data: figures/table3_metrics.csv (per-country D/A_C/A_G/A_RM in thousands of
patients/yr, H3 resolution 5, 2 h step-function driving time). Those values are
produced by scripts/table3_percountry.py from the corrected DIRAC database
(Database_DIRAC_fixed.csv) and the cached TravelTime matrices.

Usage:  python figures/plot_bottleneck.py
Outputs: figures/bottleneck.png, figures/bottleneck.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = Path(__file__).resolve().parent
MISMATCH_TOL = 0.03  # min(A_C,A_G) - A_RM above this is flagged as spatial mismatch

# Per-point label offsets (points offset) to avoid overlap in panel (a).
_LABEL_OFFSET = {
    "USA": (8, 6), "Australia": (-9, -15), "UK": (-6, 9),
    "India": (9, 4), "Nigeria": (9, 2), "Oman": (-10, 8),
}
_LABEL_HA = {"UK": "right", "Oman": "right", "Australia": "right"}

C_SUPPLY = "#d95f02"   # A_C ceiling
C_ACCESS = "#1b9e77"   # A_G ceiling
C_REAL = "#2c7fb8"     # A_RM realised


def main() -> None:
    df = pd.read_csv(_HERE / "table3_metrics.csv")
    df["c"] = df["A_C"] / df["D"]
    df["g"] = df["A_G"] / df["D"]
    df["rm"] = df["A_RM"] / df["D"]
    df["mismatch"] = np.minimum(df["c"], df["g"]) - df["rm"]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.6))

    # ---- Panel (a): supply vs access, colour = realised treatment ----
    sc = ax.scatter(df["c"], df["g"], c=df["rm"], cmap="viridis", vmin=0, vmax=1,
                    s=280, edgecolor="k", linewidth=1.2, zorder=3)
    for _, r in df.iterrows():
        dx, dy = _LABEL_OFFSET.get(r["short"], (7, 7))
        ax.annotate(r["short"], (r["c"], r["g"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10.5, weight="bold",
                    ha=_LABEL_HA.get(r["short"], "left"))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, zorder=1)
    ax.axhline(1, c="lightgrey", lw=.8, zorder=0)
    ax.axvline(1, c="lightgrey", lw=.8, zorder=0)
    ax.text(.97, .04, "supply-limited", ha="right", fontsize=9.5, style="italic", color="#555")
    ax.text(.04, .97, "access-limited", ha="left", va="top", fontsize=9.5, style="italic", color="#555")
    ax.set_xlabel("$A_C/D$  —  supply adequacy")
    ax.set_ylabel("$A_G/D$  —  geographic reach (2 h drive)")
    ax.set_xlim(-.02, 1.10)
    ax.set_ylim(-.02, 1.08)
    ax.set_title("(a) Where is the bottleneck?")
    cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.04)
    cb.set_label("$A_{RM}/D$  —  realised treatment")

    # ---- Panel (b): demand-normalised bullet bars ----
    order = df.sort_values("rm").reset_index(drop=True)
    for i, r in order.iterrows():
        c, g, rm, mn = r["c"], r["g"], r["rm"], min(r["c"], r["g"])
        bx.barh(i, 1.0, color="#eee", zorder=1)                     # demand D
        bx.barh(i, rm, color=C_REAL, height=.55, zorder=3)          # realised A_RM
        bx.plot([c, c], [i - .42, i + .42], color=C_SUPPLY, lw=2.6, zorder=4, solid_capstyle="round")
        bx.plot([g, g], [i - .42, i + .42], color=C_ACCESS, lw=2.6, zorder=4, solid_capstyle="round")
        if r["mismatch"] > MISMATCH_TOL:
            bx.annotate("", xy=(mn, i), xytext=(rm, i),
                        arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5))
            bx.text((rm + mn) / 2, i + .30, f"mismatch\n{r['mismatch']*100:.0f} pp",
                    color="crimson", fontsize=8, ha="center", va="bottom", linespacing=.9)
    bx.set_yticks(range(len(order)))
    bx.set_yticklabels(order["short"])
    bx.set_xlim(0, 1.05)
    bx.set_xlabel("fraction of national RT demand $D$")
    bx.set_title("(b) Demand, ceilings & realised treatment")
    bx.legend(handles=[
        Patch(fc=C_REAL, label="$A_{RM}$  realised treatment"),
        Line2D([0], [0], color=C_SUPPLY, lw=2.6, label="$A_C$  supply ceiling"),
        Line2D([0], [0], color=C_ACCESS, lw=2.6, label="$A_G$  access ceiling"),
    ], loc="lower right", fontsize=8.5, framealpha=.95)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(_HERE / f"bottleneck.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", _HERE / "bottleneck.png", "and .pdf")


if __name__ == "__main__":
    main()
