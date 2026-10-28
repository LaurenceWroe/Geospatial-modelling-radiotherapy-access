"""
Number of populated H3 cells per country at the resolution of the global run
(scripts/global_results.py), from the same population loader.

  figures/res5_cells.csv    iso3, name, n_cells, population

    python scripts/count_res5.py [--res 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent

from data.population import load_population_at_resolution  # noqa: E402
from data.regions import get_region  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=5)
    res = ap.parse_args().res
    rows = []
    for a2 in get_region("World").member_alpha2:
        obj = pycountry.countries.get(alpha_2=a2)
        if obj is None:
            continue
        try:
            gdf = load_population_at_resolution(obj.name, target_resolution=res)
        except Exception as exc:  # noqa: BLE001
            print(f"{obj.alpha_3}: {exc}", flush=True)
            continue
        rows.append({"iso3": obj.alpha_3, "name": obj.name, "n_cells": len(gdf),
                     "population": float(gdf["population"].sum())})
        print(obj.alpha_3, len(gdf), flush=True)
    pd.DataFrame(rows).to_csv(_ROOT / "figures" / f"res{res}_cells.csv", index=False)


if __name__ == "__main__":
    main()
