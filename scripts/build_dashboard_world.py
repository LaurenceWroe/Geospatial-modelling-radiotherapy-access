"""
Precompute the dashboard's results for every configuration it offers, so no
map has to be computed in the app:

  RTU             CCORE 2012 (barton_2014), as in the paper
  allocation      capped 3SFCA ("proportional"), nearest-first
  decay model     step, exponential, Gaussian, power law
  time limit      30, 60, 90, 120 min
  capacity        250, 350, 450, 550 patients per linac per year

Each configuration runs compute.compute_members for every World member at
H3 resolution 5 (as in the paper), with DIRAC machine counts. Outputs:

  cache/dashboard_world.parquet            allocation, rtu, model, t_max, capacity, h3, country,
                                           population, D, A, A_G, summed into
                                           resolution-3 parents, one row per parent
                                           and country so regions filter from it
  cache/dashboard_world_countries.parquet  allocation, rtu, model, t_max, capacity and the
                                           per-country table of compute_members
  dashboard_store/{ISO3}.parquet           one country at resolution 5: h3,
                                           population, then float32 columns
                                           D|rtu, A_G|rtu|model|t, A|rtu|model|t|cap (capped
                                           3SFCA) and A_nf|rtu|model|t|cap (nearest-first)
                                           (uploaded as a GitHub release; see app.py)

    scripts/remote_run.sh python scripts/build_dashboard_world.py
"""
from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent

from compute import _compute_access_one_country_for_region, compute_members  # noqa: E402
from data.regions import get_region  # noqa: E402

RTUS = ["barton_2014"]
ALLOCATIONS = {"proportional": "A", "nearest_first": "A_nf"}  # allocation -> store column prefix
MODELS = ["step", "exponential", "gaussian", "power"]
TIMES = [30.0, 60.0, 90.0, 120.0]
CAPACITIES = [250.0, 350.0, 450.0, 550.0]
STORE_RES = 3
COUNTRY_DIR = _ROOT / "dashboard_store"


def main() -> None:
    members = get_region("World").member_alpha2
    configs = list(product(ALLOCATIONS, RTUS, MODELS, TIMES, CAPACITIES))
    COUNTRY_DIR.mkdir(exist_ok=True)
    hex_parts, country_parts = [], []
    # countries outer, configurations inner, so each population grid loads once
    for i, a2 in enumerate(members):
        t0 = time.time()
        cols, parent = {}, None
        for alloc, rtu, model, t_max, cap in configs:
            try:
                hexes, countries = compute_members([a2], model=model, t_max=t_max, capacity=cap, rtu_set=rtu,
                                                   allocation=alloc)
            except ValueError:  # no population or GLOBOCAN data
                break
            if parent is None:
                parent = [h3.cell_to_parent(c, STORE_RES) for c in hexes["h3"]]
                cols["h3"] = hexes["h3"].to_numpy()
                cols["population"] = hexes["population"].to_numpy(np.float32)
            cols[f"D|{rtu}"] = hexes["D"].to_numpy(np.float32)
            cols[f"A_G|{rtu}|{model}|{t_max:.0f}"] = hexes["A_G"].to_numpy(np.float32)
            cols[f"{ALLOCATIONS[alloc]}|{rtu}|{model}|{t_max:.0f}|{cap:.0f}"] = hexes["A"].to_numpy(np.float32)
            g = hexes.assign(h3=parent).groupby(["h3", "country"], as_index=False)[
                ["population", "D", "A", "A_G"]].sum()
            tag = dict(allocation=alloc, rtu=rtu, model=model, t_max=t_max, capacity=cap)
            hex_parts.append(g.assign(**tag))
            country_parts.append(countries.assign(**tag))
        _compute_access_one_country_for_region.clear()  # each result is used once; keep memory flat
        if cols:
            pd.DataFrame(cols).to_parquet(COUNTRY_DIR / f"{pycountry.countries.get(alpha_2=a2).alpha_3}.parquet",
                                          index=False, compression="zstd")
        print(f"[{i + 1:>3}/{len(members)}] {a2}  ({time.time() - t0:.0f} s)", flush=True)
    hx = pd.concat(hex_parts, ignore_index=True)
    ct = pd.concat(country_parts, ignore_index=True)
    s = ct.groupby(["allocation", "rtu", "model", "t_max", "capacity"])[["D", "A_C", "A_G", "A"]].sum()
    print((100 * s[["A_C", "A_G", "A"]].div(s["D"], axis=0)).round(1).to_string())
    (_ROOT / "cache").mkdir(exist_ok=True)
    hx.to_parquet(_ROOT / "cache" / "dashboard_world.parquet", index=False, compression="zstd")
    ct.to_parquet(_ROOT / "cache" / "dashboard_world_countries.parquet", index=False)


if __name__ == "__main__":
    main()
