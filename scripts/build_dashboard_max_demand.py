"""
Largest radiotherapy demand of any hexagon worldwide at each H3 resolution
(1-5), for the dashboard's "relative to global demand" colour range. Read from
the stored per-country results (dashboard_store/, scripts/build_dashboard_world.py);
demand of hexagons that straddle a border is summed across countries.

  cache/dashboard_global_max_demand.json   {rtu: {res: max D_h}}

    python scripts/build_dashboard_max_demand.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h3
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
RES = [1, 2, 3, 4, 5]


def main() -> None:
    parts = []
    for f in sorted((_ROOT / "dashboard_store").glob("*.parquet")):
        d = pd.read_parquet(f, columns=None)
        dcols = [c for c in d.columns if c.startswith("D|")]
        parts.append(d[["h3"] + dcols])
    allh = pd.concat(parts, ignore_index=True)
    out = {}
    for col in [c for c in allh.columns if c.startswith("D|")]:
        rtu = col.split("|", 1)[1]
        out[rtu] = {}
        for r in RES:
            key = allh["h3"] if r == 5 else allh["h3"].map(lambda c, r=r: h3.cell_to_parent(c, r))
            out[rtu][str(r)] = float(allh.groupby(key)[col].sum().max())
        print(rtu, {k: round(v) for k, v in out[rtu].items()})
    (_ROOT / "cache" / "dashboard_global_max_demand.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
