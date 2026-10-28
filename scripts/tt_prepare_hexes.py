"""
Export, for every World member country, the H3 hex ids and populations at
resolutions 6, 5, 4 and 3 in the exact order the app uses, so that travel-time
matrices can be fetched and aggregated without loading the population grids
again (run on the server; the fetch itself runs where the API key lives).

    python scripts/tt_prepare_hexes.py            -> tt_jobs/{ISO3}_res{R}.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.population import load_population_at_resolution  # noqa: E402
from data.regions import get_region  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tt_jobs"
RESOLUTIONS = (6, 5, 4, 3)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for a2 in get_region("World").member_alpha2:
        obj = pycountry.countries.get(alpha_2=a2)
        if obj is None:
            continue
        for res in RESOLUTIONS:
            p = OUT / f"{obj.alpha_3}_res{res}.parquet"
            if p.exists():
                continue
            try:
                gdf = load_population_at_resolution(obj.name, target_resolution=res)
            except Exception as e:
                print(f"{obj.alpha_3} res{res}: failed ({e})", flush=True)
                break
            pd.DataFrame({"h3": gdf["h3"].astype(str), "population": gdf["population"].astype(float)}).to_parquet(p)
            print(f"{obj.alpha_3} res{res}: {len(gdf)} hexes", flush=True)


if __name__ == "__main__":
    main()
