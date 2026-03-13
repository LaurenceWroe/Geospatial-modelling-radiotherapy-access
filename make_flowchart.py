"""
Generate dataflow_newnew.png — radiotherapy access model pipeline flowchart.
Self-contained: only matplotlib and numpy required.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_INPUT   = "#6a3d9a"
C_DATA    = "#1b3a6b"
C_PROC    = "#1565a8"
C_OUTPUT  = "#276221"
C_METRICS = "#8b5e00"
C_ARROW   = "#444444"
C_BG      = "#f5f5f5"

# ---------------------------------------------------------------------------
# Column x-centres
# ---------------------------------------------------------------------------
XA = 2.0    # User Inputs
XB = 6.5    # Data Sources
XC = 12.0   # Processing
XD = 18.0   # Output Maps

FIG_W, FIG_H = 22, 14

# Box geometry
BOX_W_A = 3.2
BOX_W_B = 3.6
BOX_W_C = 4.0
BOX_W_D = 3.6
BOX_H   = 0.52

# ---------------------------------------------------------------------------
# Helper — draw a rounded box and return its centre
# ---------------------------------------------------------------------------

def draw_box(ax, cx, cy, text, color, box_w, box_h=BOX_H, fontsize=8.5):
    x0 = cx - box_w / 2
    y0 = cy - box_h / 2
    patch = FancyBboxPatch(
        (x0, y0), box_w, box_h,
        boxstyle="round,pad=0.12",
        linewidth=0,
        facecolor=color,
        zorder=3,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        cx, cy, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold",
        color="white", zorder=4,
        wrap=False, clip_on=False,
        multialignment="center",
    )
    return (cx, cy)

# ---------------------------------------------------------------------------
# Helper — draw an arrow between two box centres
# ---------------------------------------------------------------------------

def arrow(ax, src, dst, label=None, color=C_ARROW, lw=1.5,
          connectionstyle="arc3,rad=0.0", src_side=None, dst_side=None):
    """Draw annotated arrow.  src/dst are (cx, cy) of boxes."""
    sx, sy = src
    dx, dy = dst

    # By default connect from right/left edges or top/bottom
    if src_side == "right":
        sx += 0  # will be adjusted below using edge
    if dst_side == "left":
        dx += 0

    ax.annotate(
        "",
        xy=(dx, dy),
        xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            connectionstyle=connectionstyle,
        ),
        zorder=2,
        annotation_clip=False,
    )
    if label:
        mx = (sx + dx) / 2
        my = (sy + dy) / 2
        ax.text(
            mx, my, label,
            fontsize=7, fontstyle="italic",
            ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1.5),
            zorder=5,
            clip_on=False,
        )


def arrow_edges(ax, src_box, dst_box, src_cx, src_cy, dst_cx, dst_cy,
                src_w, dst_w, label=None, color=C_ARROW, lw=1.5,
                connectionstyle="arc3,rad=0.0"):
    """Arrow that starts from the right/left edge of a box."""
    # Determine direction: if dst is to the right, start from right edge
    if dst_cx > src_cx + 0.1:
        sx = src_cx + src_w / 2
        sy = src_cy
        dx = dst_cx - dst_w / 2
        dy = dst_cy
    elif dst_cx < src_cx - 0.1:
        sx = src_cx - src_w / 2
        sy = src_cy
        dx = dst_cx + dst_w / 2
        dy = dst_cy
    else:
        # Same column — vertical
        if dst_cy < src_cy:
            sx = src_cx
            sy = src_cy - BOX_H / 2
            dx = dst_cx
            dy = dst_cy + BOX_H / 2
        else:
            sx = src_cx
            sy = src_cy + BOX_H / 2
            dx = dst_cx
            dy = dst_cy - BOX_H / 2

    ax.annotate(
        "",
        xy=(dx, dy),
        xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            connectionstyle=connectionstyle,
        ),
        zorder=2,
        annotation_clip=False,
    )
    if label:
        mx = (sx + dx) / 2
        my = (sy + dy) / 2
        ax.text(
            mx, my, label,
            fontsize=7, fontstyle="italic",
            ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1.5),
            zorder=5,
            clip_on=False,
        )


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

ax.set_xlim(0, 21)
ax.set_ylim(0, 14)
ax.set_aspect("equal")
ax.axis("off")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(
    10.5, 13.55,
    "Radiotherapy Access Model — Data Flow",
    ha="center", va="center",
    fontsize=13, fontweight="bold", color="#222222",
    zorder=6,
)

# ---------------------------------------------------------------------------
# Column headers
# ---------------------------------------------------------------------------
HEADER_Y = 13.1
for x, label in [
    (XA, "User Inputs"),
    (XB, "Data Sources"),
    (XC, "Processing"),
    (XD, "Output Maps"),
]:
    ax.text(
        x, HEADER_Y, label,
        ha="center", va="center",
        fontsize=10, fontweight="bold", color="#333333",
        zorder=6,
    )

# ---------------------------------------------------------------------------
# COLUMN A — User Inputs
# ---------------------------------------------------------------------------
# 7 inputs, spread vertically
A_labels = [
    "Country",
    "H3 Resolution\n(levels 5–9)",
    "Cancer Type(s)",
    "RT Method\n(Optimal / Proportional)",
    "Probability Model\n(Exponential / Step / Uniform)",
    "λ or cut-off distance (km)",
    "LINAC Capacity\n(patients / yr / machine)",
]
N_A = len(A_labels)
A_TOP = 12.4
A_BOT = 1.2
A_ys = np.linspace(A_TOP, A_BOT, N_A)
A_centres = {}
for i, lbl in enumerate(A_labels):
    cy = A_ys[i]
    c = draw_box(ax, XA, cy, lbl, C_INPUT, BOX_W_A)
    A_centres[lbl] = c

# ---------------------------------------------------------------------------
# COLUMN B — Data Sources
# ---------------------------------------------------------------------------
B_labels = [
    "Kontur H3 Population\nDataset 2023",
    "GLOBOCAN 2022\n(incidence by cancer & ISO3)",
    "Optimal RT Utilisations\n(Delaney et al. 2005)",
    "Actual RT Fractions\n(per-country CSV, optional)",
    "IAEA DIRAC Database\n(LINAC locations, 2025)",
]
N_B = len(B_labels)
B_TOP = 12.0
B_BOT = 2.4
B_ys = np.linspace(B_TOP, B_BOT, N_B)
B_centres = {}
for i, lbl in enumerate(B_labels):
    cy = B_ys[i]
    c = draw_box(ax, XB, cy, lbl, C_DATA, BOX_W_B)
    B_centres[lbl] = c

# ---------------------------------------------------------------------------
# COLUMN C — Processing
# ---------------------------------------------------------------------------
C_labels = [
    "H3 Hexagonal Grid\n(resample to target resolution)",
    "Cancer Incidence per Hex\n(GLOBOCAN × population share)",
    "RT Demand per Hex\n(incidence × RT fraction)",
    "LINAC Locations\n(geocoded from DIRAC)",
    "Access Probability per Hex\nP = 1 − ∏(1 − exp(−d/λ))^w",
    "Greedy Nearest-First\nCapacity Allocation",
]
N_C = len(C_labels)
C_TOP = 12.2
C_BOT = 1.8
C_ys = np.linspace(C_TOP, C_BOT, N_C)
C_centres = {}
for i, lbl in enumerate(C_labels):
    cy = C_ys[i]
    c = draw_box(ax, XC, cy, lbl, C_PROC, BOX_W_C)
    C_centres[lbl] = c

# ---------------------------------------------------------------------------
# COLUMN D — Output Maps
# ---------------------------------------------------------------------------
D_labels = [
    "Population Density Map",
    "Cancer Incidence Map",
    "RT Treatment Map\n(Optimal / Actual)",
    "Nearest LINAC Distance Map",
    "Radiotherapy Access Map\n(P_access)",
    "Capacity-Limited Access Map",
    "Patients Treated Map",
    "Patients Untreated Map",
]
N_D = len(D_labels)
D_TOP = 12.4
D_BOT = 2.2
D_ys = np.linspace(D_TOP, D_BOT, N_D)
D_centres = {}
for i, lbl in enumerate(D_labels):
    cy = D_ys[i]
    c = draw_box(ax, XD, cy, lbl, C_OUTPUT, BOX_W_D)
    D_centres[lbl] = c

# Summary metrics box (amber) — below output maps
metrics_cy = 1.15
metrics_c = draw_box(
    ax, XD, metrics_cy,
    "Summary Metrics\nTotal LINACs · Modelled RT Need\nPatients Treated / Untreated",
    C_METRICS, BOX_W_D, box_h=0.76,
)

# ---------------------------------------------------------------------------
# Convenience: edge helpers
# ---------------------------------------------------------------------------
def right_edge(cx, cy, w):
    return (cx + w / 2, cy)

def left_edge(cx, cy, w):
    return (cx - w / 2, cy)

def top_edge(cx, cy):
    return (cx, cy + BOX_H / 2)

def bot_edge(cx, cy):
    return (cx, cy - BOX_H / 2)


def arr(src, dst, label=None, cs="arc3,rad=0.0", lw=1.5, color=C_ARROW):
    ax.annotate(
        "",
        xy=dst,
        xytext=src,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            connectionstyle=cs,
        ),
        zorder=2,
        annotation_clip=False,
    )
    if label:
        mx = (src[0] + dst[0]) / 2
        my = (src[1] + dst[1]) / 2
        ax.text(
            mx, my, label,
            fontsize=7, fontstyle="italic",
            ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1.5),
            zorder=5,
            clip_on=False,
        )


# Short aliases for centres
def ac(k): return A_centres[k]
def bc(k): return B_centres[k]
def cc(k): return C_centres[k]
def dc(k): return D_centres[k]


# ---------------------------------------------------------------------------
# ARROWS  A → B  (User inputs to Data Sources)
# ---------------------------------------------------------------------------
# Country → Kontur
arr(
    right_edge(*ac("Country"), BOX_W_A),
    left_edge(*bc("Kontur H3 Population\nDataset 2023"), BOX_W_B),
)

# ---------------------------------------------------------------------------
# ARROWS  A → C  (User inputs skipping or cross-column)
# ---------------------------------------------------------------------------
# H3 Resolution → H3 Hex Grid
arr(
    right_edge(*ac("H3 Resolution\n(levels 5–9)"), BOX_W_A),
    left_edge(*cc("H3 Hexagonal Grid\n(resample to target resolution)"), BOX_W_C),
    cs="arc3,rad=-0.15",
)

# Cancer Type(s) → Cancer Incidence per Hex (via GLOBOCAN label)
arr(
    right_edge(*ac("Cancer Type(s)"), BOX_W_A),
    left_edge(*cc("Cancer Incidence per Hex\n(GLOBOCAN × population share)"), BOX_W_C),
    label="via GLOBOCAN",
    cs="arc3,rad=-0.1",
)

# RT Method → RT Demand per Hex (via Delaney/Actual label)
arr(
    right_edge(*ac("RT Method\n(Optimal / Proportional)"), BOX_W_A),
    left_edge(*cc("RT Demand per Hex\n(incidence × RT fraction)"), BOX_W_C),
    cs="arc3,rad=0.0",
)

# Probability Model → Access Probability per Hex
arr(
    right_edge(*ac("Probability Model\n(Exponential / Step / Uniform)"), BOX_W_A),
    left_edge(*cc("Access Probability per Hex\nP = 1 − ∏(1 − exp(−d/λ))^w"), BOX_W_C),
    cs="arc3,rad=0.1",
)

# λ/cut-off → Access Probability per Hex
arr(
    right_edge(*ac("λ or cut-off distance (km)"), BOX_W_A),
    left_edge(*cc("Access Probability per Hex\nP = 1 − ∏(1 − exp(−d/λ))^w"), BOX_W_C),
    cs="arc3,rad=0.15",
)

# LINAC Capacity → Greedy Capacity Allocation
arr(
    right_edge(*ac("LINAC Capacity\n(patients / yr / machine)"), BOX_W_A),
    left_edge(*cc("Greedy Nearest-First\nCapacity Allocation"), BOX_W_C),
    cs="arc3,rad=0.2",
)

# ---------------------------------------------------------------------------
# ARROWS  B → C  (Data Sources → Processing)
# ---------------------------------------------------------------------------
# Kontur → H3 Hex Grid
arr(
    right_edge(*bc("Kontur H3 Population\nDataset 2023"), BOX_W_B),
    left_edge(*cc("H3 Hexagonal Grid\n(resample to target resolution)"), BOX_W_C),
)

# GLOBOCAN → Cancer Incidence per Hex
arr(
    right_edge(*bc("GLOBOCAN 2022\n(incidence by cancer & ISO3)"), BOX_W_B),
    left_edge(*cc("Cancer Incidence per Hex\n(GLOBOCAN × population share)"), BOX_W_C),
)

# Optimal RT → RT Demand per Hex
arr(
    right_edge(*bc("Optimal RT Utilisations\n(Delaney et al. 2005)"), BOX_W_B),
    left_edge(*cc("RT Demand per Hex\n(incidence × RT fraction)"), BOX_W_C),
)

# Actual RT → RT Demand per Hex
arr(
    right_edge(*bc("Actual RT Fractions\n(per-country CSV, optional)"), BOX_W_B),
    left_edge(*cc("RT Demand per Hex\n(incidence × RT fraction)"), BOX_W_C),
)

# DIRAC → LINAC Locations
arr(
    right_edge(*bc("IAEA DIRAC Database\n(LINAC locations, 2025)"), BOX_W_B),
    left_edge(*cc("LINAC Locations\n(geocoded from DIRAC)"), BOX_W_C),
)

# ---------------------------------------------------------------------------
# ARROWS  within C (Processing chain)
# ---------------------------------------------------------------------------
# H3 Hex Grid → Cancer Incidence
arr(
    bot_edge(*cc("H3 Hexagonal Grid\n(resample to target resolution)")),
    top_edge(*cc("Cancer Incidence per Hex\n(GLOBOCAN × population share)")),
)

# Cancer Incidence → RT Demand
arr(
    bot_edge(*cc("Cancer Incidence per Hex\n(GLOBOCAN × population share)")),
    top_edge(*cc("RT Demand per Hex\n(incidence × RT fraction)")),
)

# LINAC Locations → Access Probability (distances label)
arr(
    bot_edge(*cc("LINAC Locations\n(geocoded from DIRAC)")),
    top_edge(*cc("Access Probability per Hex\nP = 1 − ∏(1 − exp(−d/λ))^w")),
    label="distances",
)

# LINAC Locations → Greedy Capacity Allocation (curved left side)
linac_cx, linac_cy = cc("LINAC Locations\n(geocoded from DIRAC)")
greedy_cx, greedy_cy = cc("Greedy Nearest-First\nCapacity Allocation")
arr(
    (linac_cx - BOX_W_C / 2 - 0.1, linac_cy),
    (greedy_cx - BOX_W_C / 2 - 0.1, greedy_cy),
    cs="arc3,rad=0.0",
)

# RT Demand → Access Probability (vertical)
arr(
    bot_edge(*cc("RT Demand per Hex\n(incidence × RT fraction)")),
    top_edge(*cc("Access Probability per Hex\nP = 1 − ∏(1 − exp(−d/λ))^w")),
)

# RT Demand → Greedy Capacity Allocation (curved, offset left)
rtd_cx, rtd_cy = cc("RT Demand per Hex\n(incidence × RT fraction)")
arr(
    (rtd_cx - BOX_W_C / 2 - 0.4, rtd_cy),
    (greedy_cx - BOX_W_C / 2 - 0.4, greedy_cy),
    cs="arc3,rad=0.0",
)

# Access Probability → Greedy Capacity Allocation
arr(
    bot_edge(*cc("Access Probability per Hex\nP = 1 − ∏(1 − exp(−d/λ))^w")),
    top_edge(*cc("Greedy Nearest-First\nCapacity Allocation")),
)

# ---------------------------------------------------------------------------
# ARROWS  C → D  (Processing → Outputs)
# ---------------------------------------------------------------------------
# H3 Hex Grid → Population Density Map
arr(
    right_edge(*cc("H3 Hexagonal Grid\n(resample to target resolution)"), BOX_W_C),
    left_edge(*dc("Population Density Map"), BOX_W_D),
)

# Cancer Incidence → Cancer Incidence Map
arr(
    right_edge(*cc("Cancer Incidence per Hex\n(GLOBOCAN × population share)"), BOX_W_C),
    left_edge(*dc("Cancer Incidence Map"), BOX_W_D),
)

# RT Demand → RT Treatment Map
arr(
    right_edge(*cc("RT Demand per Hex\n(incidence × RT fraction)"), BOX_W_C),
    left_edge(*dc("RT Treatment Map\n(Optimal / Actual)"), BOX_W_D),
)

# LINAC Locations → Nearest LINAC Distance Map
arr(
    right_edge(*cc("LINAC Locations\n(geocoded from DIRAC)"), BOX_W_C),
    left_edge(*dc("Nearest LINAC Distance Map"), BOX_W_D),
)

# Access Probability → Radiotherapy Access Map
arr(
    right_edge(*cc("Access Probability per Hex\nP = 1 − ∏(1 − exp(−d/λ))^w"), BOX_W_C),
    left_edge(*dc("Radiotherapy Access Map\n(P_access)"), BOX_W_D),
)

# Capacity Allocation → Capacity-Limited Access Map
arr(
    right_edge(*cc("Greedy Nearest-First\nCapacity Allocation"), BOX_W_C),
    left_edge(*dc("Capacity-Limited Access Map"), BOX_W_D),
)

# Capacity Allocation → Patients Treated Map
arr(
    right_edge(*cc("Greedy Nearest-First\nCapacity Allocation"), BOX_W_C),
    left_edge(*dc("Patients Treated Map"), BOX_W_D),
    cs="arc3,rad=0.1",
)

# Capacity Allocation → Patients Untreated Map
arr(
    right_edge(*cc("Greedy Nearest-First\nCapacity Allocation"), BOX_W_C),
    left_edge(*dc("Patients Untreated Map"), BOX_W_D),
    cs="arc3,rad=0.2",
)

# ---------------------------------------------------------------------------
# ARROWS  D → Summary Metrics
# ---------------------------------------------------------------------------
# Patients Treated → Summary
arr(
    bot_edge(*dc("Patients Treated Map")),
    (XD, metrics_cy + 0.76 / 2),
    cs="arc3,rad=0.0",
)
# Patients Untreated → Summary
arr(
    bot_edge(*dc("Patients Untreated Map")),
    (XD, metrics_cy + 0.76 / 2),
    cs="arc3,rad=0.1",
)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_y = 0.45
legend_items = [
    (C_INPUT,   "User Inputs"),
    (C_DATA,    "Data Sources"),
    (C_PROC,    "Processing"),
    (C_OUTPUT,  "Output Maps"),
    (C_METRICS, "Summary Metrics"),
]
lx = 5.0
for col, lbl in legend_items:
    patch = mpatches.FancyBboxPatch(
        (lx - 0.25, legend_y - 0.18), 0.5, 0.36,
        boxstyle="round,pad=0.05",
        facecolor=col, linewidth=0, zorder=6,
    )
    ax.add_patch(patch)
    ax.text(lx + 0.4, legend_y, lbl, fontsize=8,
            va="center", ha="left", color="#333", zorder=7)
    lx += 2.8

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plt.tight_layout(pad=0)
plt.savefig(
    "/Users/wroe/Geospatial-modelling-radiotherapy-access/dataflow_newnew.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=C_BG,
)
print("Saved dataflow_newnew.png")
