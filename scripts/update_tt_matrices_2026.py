"""
Build travel-time matrices for the 2026 DIRAC facility list from the cached
2025 matrices, fetching only the facilities that are new or moved.

For every cached matrix {ISO3}_res{R}_{H}h_{hash2025}_{mode}.npz of a study
country, the 2026 facility list is matched to the 2025 columns by coordinate
(carried-over centres have identical coordinates, so their travel times are
unchanged). Columns for the remaining facilities are fetched from the
TravelTime /v4/h3 endpoint with the same resolution, cut-off and mode, and the
result is saved under the 2026 coordinate hash so compute.py picks it up
unchanged. Credentials come from .streamlit/secrets.toml.

    python scripts/update_tt_matrices_2026.py [--countries "United Kingdom" India ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "travel_time_cache"
_NEW_CSV = _ROOT / "linac_data" / "dirac_2026" / "Database_DIRAC_2026_fixed.csv"

import os
os.environ["RADMAPS_DIRAC_CSV"] = str(_NEW_CSV)  # make data.linacs read the 2026 file
from data import linacs as _linacs_2026  # noqa: E402
from data.population import load_population_at_resolution  # noqa: E402
from data.travel_time import aggregate_tt_matrix, compute_travel_time_matrix  # noqa: E402

STUDY = ["Australia", "India", "Nigeria", "Oman", "United Kingdom", "United States"]
TOL_KM = 1.0
_PAT = re.compile(r"^(?P<iso3>[A-Z]{3})_res(?P<res>\d)_(?P<h>\d+)h_(?P<hash>[0-9a-f]{8})_(?P<mode>driving|public_transport)\.npz$")


def _hash(ll):
    return hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]


def _load_2025_locs(name):
    """The 2025 facility set exactly as the app loaded it when the cached matrices were built."""
    locs, _ = _linacs_2026.load_linacs_from_dirac_db(name, csv_path=_linacs_2026.DIRAC_CSV_2025)
    return [(lat, lon) for lat, lon, _ in locs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*", default=STUDY)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sec = tomllib.loads((_ROOT / ".streamlit" / "secrets.toml").read_text())["traveltime"]

    for name in args.countries:
        obj = pycountry.countries.lookup(name)
        iso3 = obj.alpha_3
        locs_new, _ = _linacs_2026.load_linacs_from_dirac_db(obj.name)
        ll_new = [(lat, lon) for lat, lon, _ in locs_new]
        ll_old = _load_2025_locs(obj.name)
        h_old, h_new = _hash(ll_old), _hash(ll_new)
        # a 2026 facility reuses a 2025 column if it lies within TOL_KM of that facility
        # (geocoder differences of a few hundred metres do not change res-5 travel times)
        old_arr = np.radians(np.array(ll_old, dtype=float)) if ll_old else np.zeros((0, 2))
        carried = []
        for lat, lon in ll_new:
            if old_arr.size == 0:
                carried.append(None); continue
            la, lo = np.radians(lat), np.radians(lon)
            a = np.sin((old_arr[:, 0] - la) / 2) ** 2 + np.cos(la) * np.cos(old_arr[:, 0]) * np.sin((old_arr[:, 1] - lo) / 2) ** 2
            d = 2 * 6371.0088 * np.arcsin(np.sqrt(a))
            k = int(np.argmin(d))
            carried.append(k if d[k] <= TOL_KM else None)
        new_cols = [j for j, c in enumerate(carried) if c is None]
        print(f"\n{name}: 2025 {len(ll_old)} facilities ({h_old}) -> 2026 {len(ll_new)} ({h_new}); "
              f"{len(ll_new) - len(new_cols)} carried, {len(new_cols)} to fetch")
        files = [p for p in _CACHE.glob(f"{iso3}_res*_{h_old}_*.npz")]
        if not files:
            print("  no cached 2025 matrices for this country"); continue
        # Resolution 5 first: the API only serves resolution >= 5, so coarser
        # matrices are population-weighted aggregates of the new resolution-5 one
        # (exactly as the app builds them).
        for p in sorted(files, key=lambda q: (-int(_PAT.match(q.name)["res"]), q.name)):
            m = _PAT.match(p.name)
            res, hours, mode = int(m["res"]), int(m["h"]), m["mode"]
            target = _CACHE / f"{iso3}_res{res}_{hours}h_{h_new}_{mode}.npz"
            if target.exists():
                print(f"  {target.name}: exists"); continue
            if res < 5:
                src = _CACHE / f"{iso3}_res5_{hours}h_{h_new}_{mode}.npz"
                if not src.exists():
                    print(f"  {target.name}: no new resolution-5 matrix for {hours} h {mode}; skipped"); continue
                if args.dry_run:
                    print(f"  {target.name}: would aggregate from {src.name}"); continue
                gdf5 = load_population_at_resolution(obj.name, target_resolution=5)
                gdf = load_population_at_resolution(obj.name, target_resolution=res)
                agg = aggregate_tt_matrix(
                    np.load(src)["matrix"], list(gdf5["h3"]), gdf5["population"].to_numpy(dtype=np.float64),
                    list(gdf["h3"]), target_resolution=res, unreachable_fill_min=hours * 60.0,
                )
                np.savez_compressed(target, matrix=agg)
                print(f"  {target.name}: saved (aggregated from resolution 5)"); continue
            old = np.load(p)["matrix"]
            gdf = load_population_at_resolution(obj.name, target_resolution=res)
            hex_ids = list(gdf["h3"])
            if old.shape[0] != len(hex_ids):
                print(f"  {p.name}: hex count {old.shape[0]} != grid {len(hex_ids)}; skipped"); continue
            new = np.full((len(hex_ids), len(ll_new)), np.inf, dtype=np.float32)
            for j, c in enumerate(carried):
                if c is not None:
                    new[:, j] = old[:, c]
            if new_cols:
                if args.dry_run:
                    print(f"  {target.name}: would fetch {len(new_cols)} facilities ({hours} h, {mode})"); continue
                fetched, errors = compute_travel_time_matrix(
                    hex_ids, [ll_new[j] for j in new_cols], res, mode, sec["app_id"], sec["api_key"],
                    cache_key=f"_tmp_{iso3}_res{res}_{hours}h_{h_new}", max_travel_time_sec=hours * 3600,
                )
                if errors:
                    print(f"  {target.name}: API errors {errors}; not saved"); continue
                for k, j in enumerate(new_cols):
                    new[:, j] = fetched[:, k]
                tmp = _CACHE / f"_tmp_{iso3}_res{res}_{hours}h_{h_new}_{mode}.npz"
                if tmp.exists():
                    tmp.unlink()
            np.savez_compressed(target, matrix=new)
            print(f"  {target.name}: saved ({len(new_cols)} fetched)")


if __name__ == "__main__":
    main()
