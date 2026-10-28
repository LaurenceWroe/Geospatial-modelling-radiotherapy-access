"""
One-at-a-time sensitivity analysis of the RadMaps metrics for the United Kingdom.

Baseline = the paper's Table 3 configuration: H3 resolution 5, step function with
a 2 h driving-time cut-off, 450 patients per linac per year, Delaney 2005
optimal RTU (as published), nearest-first allocation. Each axis below is varied
on its own with everything else held at baseline:

  resolution      H3 3, 4, 5, 6 (2 h driving; cached 5 h matrices thresholded)
  drive_cutoff    30 … 300 min driving (res-5 5 h matrix)
  transit_cutoff  30 … 180 min public transport (res-5 3 h matrix)
  distance_cutoff 25 … 300 km straight-line step
  capacity        250 … 800 patients per linac per year
  decay           step / Gaussian / Weibull k=1,2,4 / exponential / uniform, in driving time
  rtu_set         Delaney 2005 as published / corrected / CCORE 2012
  allocation      nearest-first / proportional / optimal

Outputs: figures/sensitivity_uk.csv and figures/sensitivity_uk.{png,pdf}
(2 x 4 panels of A_C, A_G, A_RM as % of demand with the baseline marked).

Memory: one UK grid at a time (res 6 ≈ 10k hexes). Run on the server:
    scripts/remote_run.sh python scripts/sensitivity_uk.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "travel_time_cache"
_FIG = _ROOT / "figures"

from data.population import load_population_at_resolution
from data.linacs import load_linacs_from_dirac_db
from data.cancer import apportion_cancer_to_h3, get_cancer_types, DERIVED_CANCER_TYPES
from analysis.accessibility import compute_accessibility

COUNTRY = "United Kingdom"
_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}
BASE = dict(res=5, cutoff_min=120, capacity=450.0, rtu="delaney_2005", allocation="nearest_first")
RTU_LABELS = {"delaney_2005": "2005 (as published)", "delaney_2005_corrected": "2005 (corrected)", "barton_2014": "CCORE 2012"}
ALLOC_LABELS = {"nearest_first": "nearest-first", "proportional": "proportional", "optimal": "optimal"}


def _linac_hash(locs):
    ll = [(lat, lon) for lat, lon, _ in locs]
    return hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]


def _matrix(iso3, res, hours, mode, locs):
    p = _CACHE / f"{iso3}_res{res}_{hours}h_{_linac_hash(locs)}_{mode}.npz"
    if not p.exists():
        raise FileNotFoundError(p.name)
    return np.load(p)["matrix"]


def _demand(gdf, iso3, rtu):
    c = apportion_cancer_to_h3(gdf, iso3, get_cancer_types() + DERIVED_CANCER_TYPES,
                               use_actual_rt=False, rtu_set=rtu)
    cols = [x for x in c.columns if x.endswith("_optimal_rt") and x[:-len("_optimal_rt")].strip().lower() not in _AGG]
    return c[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)


def run(gdf, locs, dem, *, res, tt=None, model="step", threshold=120.0, scale=60.0, shape=2.0,
        capacity=450.0, allocation="nearest_first", cutoff=None):
    g, s = compute_accessibility(
        gdf, locs, model=model, lambda_km=scale, weibull_k=shape, max_distance_km=threshold,
        cutoff_km=cutoff, capacity_per_machine_per_year=capacity, demand=dem, h3_resolution=res,
        travel_time_matrix=tt, allocation=allocation,
    )
    D = s["total_rt_demand"]
    return {
        "D": D,
        "A_C": 100 * min(D, s["total_machines"] * capacity) / D,
        "A_G": 100 * float((g["access_probability"] * g["rt_demand"]).sum()) / D,
        "A_RM": 100 * s["total_rt_treated"] / D,
        "unused_capacity": s.get("unused_capacity", 0.0),
    }


def main() -> None:
    obj = pycountry.countries.lookup(COUNTRY)
    iso3 = obj.alpha_3
    locs, _ = load_linacs_from_dirac_db(obj.name)
    rows = []

    def add(axis, value, label, r):
        rows.append({"axis": axis, "value": value, "label": label, **r})
        print(f"{axis:16} {str(label):22} A_C {r['A_C']:5.1f}  A_G {r['A_G']:5.1f}  A_RM {r['A_RM']:5.1f}", flush=True)

    # --- res-5 grid used by most axes -------------------------------------
    gdf5 = load_population_at_resolution(obj.name, target_resolution=5)
    dem5 = {k: _demand(gdf5, iso3, k) for k in RTU_LABELS}
    tt5_5h = _matrix(iso3, 5, 5, "driving", locs)
    tt5_3h_pt = _matrix(iso3, 5, 3, "public_transport", locs)
    base_dem = dem5[BASE["rtu"]]

    base = run(gdf5, locs, base_dem, res=5, tt=tt5_5h, threshold=120)
    add("baseline", 120, "baseline", base)

    for m in [30, 60, 90, 120, 150, 180, 240, 300]:
        add("drive_cutoff", m, f"{m} min", run(gdf5, locs, base_dem, res=5, tt=tt5_5h, threshold=m))
    for m in [30, 60, 90, 120, 150, 180]:
        add("transit_cutoff", m, f"{m} min", run(gdf5, locs, base_dem, res=5, tt=tt5_3h_pt, threshold=m))
    for km in [25, 50, 75, 100, 150, 200, 300]:
        add("distance_cutoff", km, f"{km} km", run(gdf5, locs, base_dem, res=5, tt=None, threshold=km))
    for cap in [250, 300, 350, 400, 450, 500, 600, 700, 800]:
        add("capacity", cap, f"{cap}", run(gdf5, locs, base_dem, res=5, tt=tt5_5h, threshold=120, capacity=cap))
    decays = [
        ("step 120 min", dict(model="step", threshold=120)),
        ("Gaussian 120 min", dict(model="gaussian", threshold=120)),
        ("Weibull λ=120, k=4", dict(model="weibull", scale=120, shape=4, cutoff=300)),
        ("Weibull λ=120, k=2", dict(model="weibull", scale=120, shape=2, cutoff=300)),
        ("exponential λ=120", dict(model="exponential", scale=120, cutoff=300)),
        ("exponential λ=60", dict(model="exponential", scale=60, cutoff=300)),
        ("uniform", dict(model="uniform")),
    ]
    for i, (lab, kw) in enumerate(decays):
        add("decay", i, lab, run(gdf5, locs, base_dem, res=5, tt=tt5_5h, **kw))
    for i, k in enumerate(RTU_LABELS):
        add("rtu_set", i, RTU_LABELS[k], run(gdf5, locs, dem5[k], res=5, tt=tt5_5h, threshold=120))
    for i, a in enumerate(ALLOC_LABELS):
        add("allocation", i, ALLOC_LABELS[a], run(gdf5, locs, base_dem, res=5, tt=tt5_5h, threshold=120, allocation=a))
    del gdf5, dem5, tt5_5h, tt5_3h_pt

    # --- resolution axis (one grid at a time) ------------------------------
    for res in [3, 4, 5, 6]:
        gdf = load_population_at_resolution(obj.name, target_resolution=res)
        dem = _demand(gdf, iso3, BASE["rtu"])
        tt = _matrix(iso3, res, 5 if res != 5 else 5, "driving", locs)
        add("resolution", res, f"res {res}", run(gdf, locs, dem, res=res, tt=tt, threshold=120))
        del gdf, dem, tt

    df = pd.DataFrame(rows)
    _FIG.mkdir(exist_ok=True)
    df.to_csv(_FIG / "sensitivity_uk.csv", index=False)

    # --- figure ------------------------------------------------------------
    panels = [
        ("resolution", "H3 resolution", 5, None),
        ("drive_cutoff", "Driving-time cut-off (min)", 120, None),
        ("transit_cutoff", "Public-transport cut-off (min)", None, None),
        ("distance_cutoff", "Straight-line cut-off (km)", None, None),
        ("capacity", "Capacity (patients / linac / yr)", 450, None),
        ("decay", "Decay model (driving time)", 0, "cat"),
        ("rtu_set", "Optimal RTU set", 0, "cat"),
        ("allocation", "Allocation rule", 0, "cat"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    colours = {"A_C": "#1f77b4", "A_G": "#9ecae1", "A_RM": "#ff7f0e"}
    for ax, (axis, xlabel, base_x, kind) in zip(axes.flat, panels):
        d = df[df["axis"] == axis]
        x = d["value"].to_numpy() if kind != "cat" else np.arange(len(d))
        for metric in ("A_C", "A_G", "A_RM"):
            ax.plot(x, d[metric], marker="o", ms=4, lw=1.6, color=colours[metric],
                    label={"A_C": r"$A_\mathrm{C}$", "A_G": r"$A_\mathrm{G}$", "A_RM": r"$A_\mathrm{RM}$"}[metric])
        if base_x is not None:
            ax.axvline(base_x if kind != "cat" else 0, color="grey", lw=0.8, ls="--")
        if kind == "cat":
            ax.set_xticks(x); ax.set_xticklabels(d["label"], rotation=35, ha="right", fontsize=8)
        ax.set_xlabel(xlabel); ax.set_ylim(0, 105); ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Access (% of RT demand)")
    axes[0, 0].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"United Kingdom — one-at-a-time sensitivity around the Table 3 baseline "
                 f"(res 5, 2 h driving step, 450 pts/linac, Delaney 2005 RTU, nearest-first)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(_FIG / "sensitivity_uk.png", dpi=200)
    fig.savefig(_FIG / "sensitivity_uk.pdf")
    print(f"\nwritten {_FIG / 'sensitivity_uk.csv'} and sensitivity_uk.png/pdf")


if __name__ == "__main__":
    main()
