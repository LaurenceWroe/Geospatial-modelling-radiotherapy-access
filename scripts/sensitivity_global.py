"""
Global sensitivity of RadMaps access to the configured inputs, one input at a
time around the main configuration (capped 3SFCA, Gaussian decay with a
120-min driving-time limit, 450 patients per linac per year, CCORE 2012
optimal RTU, H3 resolution 5):

  capacity      250, 350, 450, 550 patients per linac per year
  time limit    30, 60, 90, 120 min (Gaussian x_max)
  allocation    nearest-first vs capped 3SFCA
  decay model   step (120 min), exponential (lambda = 60 min), Gaussian
                (x_max = 120 min), power law (beta = 1, plateau x0 = 10 min),
                all cut off at 120 min

Countries TravelTime does not serve use the calibrated straight-line
equivalent of every time parameter (compute.minutes_to_proxy_km), as in
scripts/global_results.py.

  figures/sensitivity_global.csv     one row per country and configuration:
                                     D, A_C, A_G, A (patients per year)

    python scripts/sensitivity_global.py          # run on the server
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_FIG = _ROOT / "figures"

from analysis.accessibility import compute_accessibility  # noqa: E402
from compute import minutes_to_proxy_km  # noqa: E402
from data.cancer import DERIVED_CANCER_TYPES, apportion_cancer_to_h3, get_cancer_types  # noqa: E402
from data.linacs import load_linacs_from_dirac_db  # noqa: E402
from data.population import load_population_at_resolution  # noqa: E402
from data.regions import get_region  # noqa: E402
from data.tt_store import store_matrix  # noqa: E402
from scripts.global_results import _demand  # noqa: E402

RES = 5
BASE = dict(model="gaussian", rule="proportional", cap=450.0, t_max=120.0, scale=None, shape=None)
# (label, input varied, overrides of BASE)
CONFIGS = [("base", "baseline", {})]
CONFIGS += [(f"cap{c}", "capacity", {"cap": float(c)}) for c in (250, 350, 550)]
CONFIGS += [(f"t{t}", "time limit", {"t_max": float(t)}) for t in (30, 60, 90)]
CONFIGS += [("nf", "allocation", {"rule": "nearest_first"})]
CONFIGS += [("step", "decay", {"model": "step"}),
            ("exponential", "decay", {"model": "exponential", "scale": 60.0}),
            ("power", "decay", {"model": "power", "scale": 10.0, "shape": 1.0})]


def _run(gdf, locs, dem, tt, cfg):
    """(A_G, A) in patients per year for one configuration."""
    if tt is None:  # distance proxy: every time parameter in calibrated km
        thr = minutes_to_proxy_km(cfg["t_max"])
        scale = minutes_to_proxy_km(cfg["scale"]) if cfg["scale"] is not None else None
    else:
        thr, scale = cfg["t_max"], cfg["scale"]
    kw = dict(model=cfg["model"], max_distance_km=thr, capacity_per_machine_per_year=cfg["cap"],
              demand=dem, h3_resolution=RES, travel_time_matrix=tt, allocation=cfg["rule"])
    if cfg["model"] in ("exponential", "power"):
        kw.update(lambda_km=scale, cutoff_km=thr)
        if cfg["shape"] is not None:
            kw["weibull_k"] = cfg["shape"]
    g, s = compute_accessibility(gdf, locs, **kw)
    a_g = float((g["access_probability"] * g["rt_demand"]).sum())
    return a_g, float(s["total_rt_treated"])


def main() -> None:
    cancers = get_cancer_types() + DERIVED_CANCER_TYPES
    objs = [pycountry.countries.get(alpha_2=a2) for a2 in get_region("World").member_alpha2]
    rows = []
    for i, obj in enumerate(o for o in objs if o is not None):
        name, iso3 = obj.name, obj.alpha_3
        try:
            gdf = load_population_at_resolution(name, target_resolution=RES)
            cgdf = apportion_cancer_to_h3(gdf, iso3, cancers, use_actual_rt=False, rtu_set="barton_2014")
        except Exception as exc:  # noqa: BLE001
            print(f"{iso3}: skipped ({str(exc)[:80]})", flush=True)
            continue
        dem = _demand(cgdf)
        if dem is None or dem.sum() <= 0:
            continue
        try:
            locs, _ = load_linacs_from_dirac_db(name)
        except Exception:  # noqa: BLE001
            locs = []
        D = float(dem.sum())
        n_mac = float(sum(w for *_, w in locs)) if locs else 0.0
        tt = store_matrix(iso3, list(gdf["h3"]), locs) if locs else None
        for lab, varied, over in CONFIGS:
            cfg = {**BASE, **over}
            a_g, a = _run(gdf, locs, dem, tt, cfg) if locs else (0.0, 0.0)
            rows.append({"iso3": iso3, "config": lab, "varied": varied, "D": D,
                         "A_C": min(D, n_mac * cfg["cap"]), "A_G": a_g, "A": a,
                         "time_source": "none" if not locs else ("traveltime" if tt is not None else "proxy")})
        base = next(r for r in rows[::-1] if r["iso3"] == iso3 and r["config"] == "base")
        print(f"[{i + 1:>3}] {iso3} {name[:22]:22} D={D:>9.0f}  A={base['A'] / D:5.0%}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(_FIG / "sensitivity_global.csv", index=False)
    g = out.groupby(["config", "varied"])[["D", "A_C", "A_G", "A"]].sum()
    print("\nGlobal (% of D):")
    print((100 * g[["A_C", "A_G", "A"]].div(g["D"], axis=0)).round(1).to_string())


if __name__ == "__main__":
    main()
