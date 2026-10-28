"""
Sensitivity of global access to the H3 resolution at which it is computed, in
the main configuration (capped 3SFCA, Gaussian 120-min driving-time limit, 450
patients per linac per year, CCORE 2012 optimal RTU). Driving times below
resolution 6 are aggregated from the resolution-6 store (data.tt_store);
countries TravelTime does not serve use the calibrated straight-line proxy.

At resolution 6 a country is skipped when its dense hexagon x facility matrix
would exceed MAX_PAIRS entries (memory); skipped countries are recorded so the
comparison can be restricted to the countries computed at every resolution.

  figures/sensitivity_resolution.csv   iso3, res, D, A_C, A_G, A, n_hex, skipped

    scripts/remote_run.sh python scripts/sensitivity_resolution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_FIG = Path(__file__).resolve().parent.parent / "figures"

from analysis.accessibility import compute_accessibility  # noqa: E402
from compute import minutes_to_proxy_km  # noqa: E402
from data.cancer import DERIVED_CANCER_TYPES, apportion_cancer_to_h3, get_cancer_types  # noqa: E402
from data.linacs import load_linacs_from_dirac_db  # noqa: E402
from data.population import load_population_at_resolution  # noqa: E402
from data.regions import get_region  # noqa: E402
from data.tt_store import store_matrix  # noqa: E402
from scripts.global_results import _demand  # noqa: E402

RESOLUTIONS = [3, 4, 5, 6]
CAP, T_MAX = 450.0, 120.0
MAX_PAIRS = 1.5e8


def main() -> None:
    cancers = get_cancer_types() + DERIVED_CANCER_TYPES
    rows = []
    for a2 in get_region("World").member_alpha2:
        obj = pycountry.countries.get(alpha_2=a2)
        if obj is None:
            continue
        try:
            locs, _ = load_linacs_from_dirac_db(obj.name)
        except Exception:  # noqa: BLE001
            locs = []
        for res in RESOLUTIONS:
            try:
                gdf = load_population_at_resolution(obj.name, target_resolution=res)
                dem = _demand(apportion_cancer_to_h3(gdf, obj.alpha_3, cancers, use_actual_rt=False,
                                                     rtu_set="barton_2014"))
            except Exception:  # noqa: BLE001  no population or GLOBOCAN data
                break
            if dem is None or dem.sum() <= 0:
                break
            D = float(dem.sum())
            row = {"iso3": obj.alpha_3, "res": res, "D": D, "n_hex": len(gdf), "skipped": False}
            n_mac = float(sum(n for *_, n in locs)) if locs else 0.0
            row["A_C"] = min(D, n_mac * CAP)
            if not locs:
                row.update(A_G=0.0, A=0.0)
            elif len(gdf) * len(locs) > MAX_PAIRS:
                row.update(A_G=float("nan"), A=float("nan"), skipped=True)
            else:
                tt = store_matrix(obj.alpha_3, list(gdf["h3"]), locs)
                thr = T_MAX if tt is not None else minutes_to_proxy_km(T_MAX)
                g, s = compute_accessibility(gdf, locs, model="gaussian", max_distance_km=thr,
                                             capacity_per_machine_per_year=CAP, demand=dem, h3_resolution=res,
                                             travel_time_matrix=tt, allocation="proportional")
                row.update(A_G=float((g["access_probability"] * g["rt_demand"]).sum()),
                           A=float(s["total_rt_treated"]))
            rows.append(row)
        print(f"{obj.alpha_3} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(_FIG / "sensitivity_resolution.csv", index=False)
    skipped = sorted(df.loc[df["skipped"], "iso3"])
    print("\nskipped at resolution 6:", skipped)
    for label, d in [("all countries", df), ("countries computed at every resolution",
                                            df[~df["iso3"].isin(skipped)])]:
        g = d.groupby("res")[["D", "A_C", "A_G", "A"]].sum(min_count=1)
        print(f"\n{label} (% of D):")
        print((100 * g[["A_C", "A_G", "A"]].div(g["D"], axis=0)).round(1).to_string())


if __name__ == "__main__":
    main()
