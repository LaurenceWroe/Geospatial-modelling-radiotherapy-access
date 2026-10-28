"""
Sensitivity of nearest-first allocation to the order in which facilities are
processed within a ring (analysis.accessibility._allocate_nearest_first sorts
them by coordinates). For every country with at least one centre, in the main
configuration (Gaussian, 120 min, 450 patients per linac, CCORE 2012, res 5),
national access is computed with the canonical order and with random orders.
Capped 3SFCA is computed with the canonical and one random order as a check
that it does not depend on any ordering.

  figures/nf_order_sensitivity.csv   iso3, rule, seed, D, A

    scripts/remote_run.sh python scripts/nf_order_sensitivity.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_FIG = Path(__file__).resolve().parent.parent / "figures"

import analysis.accessibility as acc  # noqa: E402
from compute import minutes_to_proxy_km  # noqa: E402
from data.cancer import DERIVED_CANCER_TYPES, apportion_cancer_to_h3, get_cancer_types  # noqa: E402
from data.linacs import load_linacs_from_dirac_db  # noqa: E402
from data.population import load_population_at_resolution  # noqa: E402
from data.regions import get_region  # noqa: E402
from data.tt_store import store_matrix  # noqa: E402
from scripts.global_results import _demand  # noqa: E402

SEEDS = {"nearest_first": 10, "proportional": 2}  # seed 0 is the canonical order


def _order(seed):
    """Replacement for the facility sort: canonical for seed 0, else a random permutation."""
    if seed == 0:
        return sorted
    rng = random.Random(seed)
    return lambda it, key=None: rng.sample(list(it), len(list(it)))


def main() -> None:
    cancers = get_cancer_types() + DERIVED_CANCER_TYPES
    rows = []
    for a2 in get_region("World").member_alpha2:
        obj = pycountry.countries.get(alpha_2=a2)
        if obj is None:
            continue
        try:
            locs, _ = load_linacs_from_dirac_db(obj.name)
            gdf = load_population_at_resolution(obj.name, target_resolution=5)
            dem = _demand(apportion_cancer_to_h3(gdf, obj.alpha_3, cancers, use_actual_rt=False,
                                                 rtu_set="barton_2014"))
        except Exception:  # noqa: BLE001  no centres, population or GLOBOCAN data
            continue
        if not locs or dem is None or dem.sum() <= 0:
            continue
        tt = store_matrix(obj.alpha_3, list(gdf["h3"]), locs)
        thr = 120.0 if tt is not None else minutes_to_proxy_km(120.0)
        for rule, n in SEEDS.items():
            for seed in range(n):
                acc.sorted = _order(seed)
                _, s = acc.compute_accessibility(gdf, locs, model="gaussian", max_distance_km=thr,
                                                 capacity_per_machine_per_year=450.0, demand=dem,
                                                 h3_resolution=5, travel_time_matrix=tt, allocation=rule)
                rows.append({"iso3": obj.alpha_3, "rule": rule, "seed": seed, "D": float(dem.sum()),
                             "A": float(s["total_rt_treated"]), "n_facilities": len(locs)})
        acc.sorted = sorted
        print(f"{obj.alpha_3} done ({len(locs)} centres)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(_FIG / "nf_order_sensitivity.csv", index=False)
    for rule in SEEDS:
        d = df[df["rule"] == rule]
        g = d.groupby("iso3").agg(D=("D", "first"), lo=("A", "min"), hi=("A", "max"))
        spread = 100 * (g["hi"] - g["lo"]) / g["D"]
        w = (spread * g["D"]).sum() / g["D"].sum()
        glob = d.groupby("seed")["A"].sum() / g["D"].sum() * 100
        print(f"\n{rule}: {len(g)} countries; spread in national A/D (points): "
              f"max {spread.max():.2f} ({spread.idxmax()}), mean {spread.mean():.3f}, "
              f"demand-weighted mean {w:.3f}; global A/D {glob.min():.3f}-{glob.max():.3f}%")
        print(spread.sort_values(ascending=False).head(5).round(2).to_string())


if __name__ == "__main__":
    main()
