"""
Fetch 120-min driving travel times for every country at H3 resolution 6
against the default (October 2026) DIRAC facility list, store them sparsely,
and build the dense resolution-5, 4 and 3 matrices the engine reads by
population-weighted aggregation.

Resolution 6 costs the same number of API requests as resolution 5 (one per
10 facilities) but resolves road access within each resolution-5 cell. A
dense resolution-6 matrix is too large for big countries (USA ~2 GB), so only
reachable (hex, facility, minutes) triples are kept:

    travel_time_cache/res6_sparse/{ISO3}_res6_2h_{hash}_driving.npz
        rows (int32), cols (int32), minutes (float32), n_hex, n_fac

and the engine's usual dense files are written next to the other caches:

    travel_time_cache/{ISO3}_res{5,4,3}_2h_{hash}_driving.npz

Aggregation matches data.travel_time.aggregate_tt_matrix: each child weighs
max(population, 1); unreachable children count at the 120-min cut-off; a
parent stays unreachable from a facility only if none of its children reach it.
An existing dense file of the same name (e.g. a native resolution-5 fetch) is
moved to travel_time_cache/native/ first.

Each fetch is also written to the self-describing store of data/tt_store.py
(travel_time_store/{ISO3}_driving.npz), which is what the analysis reads;
scripts/build_tt_store.py converted the fetches made before it existed.

Hex ids come from tt_jobs/{ISO3}_res{R}.parquet (scripts/tt_prepare_hexes.py,
run on the server), so this script never loads population grids. Shorter
budgets (30, 60, 90 min) are thresholds of the same matrices.

    python scripts/fetch_tt_res6.py --dry-run           # count API requests
    python scripts/fetch_tt_res6.py [--countries PRT KEN ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import tomllib
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "travel_time_cache"
_SPARSE = _CACHE / "res6_sparse"
_NATIVE = _CACHE / "native"
_JOBS = _ROOT / "tt_jobs"

from data.linacs import load_linacs_from_dirac_db  # noqa: E402
from data.tt_store import save_store  # noqa: E402
from data.travel_time import (  # noqa: E402
    MAX_SNAP_THRESHOLD_M,
    _MAX_SEARCHES_PER_REQUEST,
    _call_h3,
    next_wednesday_8am_local,
)

RES, HOURS, MODE = 6, 2, "driving"
CUTOFF_MIN = HOURS * 60.0
_RETRIES, _BACKOFF = 10, (2, 5, 10)


def _hash(ll):
    return hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]


def _locs(name):
    try:
        locs, _ = load_linacs_from_dirac_db(name)
    except Exception:
        return []
    return [(lat, lon) for lat, lon, _ in locs]


def fetch_sparse(hex_ids, ll, app_id, api_key):
    """Reachable (row, col, minutes) triples for every facility; raises on a failed batch."""
    idx = {c: i for i, c in enumerate(hex_ids)}
    snap = min(int(h3.average_hexagon_edge_length(RES, unit="m") * 3), MAX_SNAP_THRESHOLD_M)
    dep = [next_wednesday_8am_local(lat, lon) for lat, lon in ll]  # each facility's local 08:00
    R, C, T = [], [], []
    bad: list[int] = []

    def call(sub, ids):
        for attempt in range(_RETRIES):
            try:
                return _call_h3(sub, ids, RES, MODE, app_id, api_key, [dep[int(i[1:])] for i in ids], int(CUTOFF_MIN * 60), snap)
            except Exception as exc:  # noqa: BLE001
                if getattr(getattr(exc, "response", None), "status_code", None) == 422:
                    raise  # unprocessable start point: retrying will not help
                if attempt == _RETRIES - 1:
                    raise
                time.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])

    def add(res):
        for lid, cells in res.items():
            j = int(lid[1:])
            pairs = [(idx[c], t) for c, t in cells.items() if c in idx]
            if pairs:
                rr, tt = zip(*pairs)
                R.append(np.asarray(rr, np.int32)); T.append(np.asarray(tt, np.float32))
                C.append(np.full(len(rr), j, np.int32))

    for s in range(0, len(ll), _MAX_SEARCHES_PER_REQUEST):
        e = min(s + _MAX_SEARCHES_PER_REQUEST, len(ll))
        try:
            add(call(ll[s:e], [f"l{j}" for j in range(s, e)]))
        except Exception as exc:  # noqa: BLE001
            if getattr(getattr(exc, "response", None), "status_code", None) != 422:
                raise RuntimeError(f"facilities {s}-{e - 1}: {exc}") from exc
            # one facility cannot start a search: fetch the batch one by one
            for j in range(s, e):
                try:
                    add(call([ll[j]], [f"l{j}"]))
                except Exception:  # noqa: BLE001
                    bad.append(j)
            if s == 0 and len(bad) == e:
                raise RuntimeError(f"no facility in the first batch is routable (country not served?): {exc}")
    fetch_sparse.unroutable = bad
    if not R:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    rows, cols, mins = np.concatenate(R), np.concatenate(C), np.concatenate(T)
    # keep the minimum time per (hex, facility) pair
    order = np.lexsort((mins, cols, rows))
    rows, cols, mins = rows[order], cols[order], mins[order]
    first = np.ones(rows.size, bool)
    first[1:] = (np.diff(rows) != 0) | (np.diff(cols) != 0)
    return rows[first], cols[first], mins[first]


def aggregate(rows, cols, mins, pop6, hex6, hex_target, res, n_fac):
    """Dense population-weighted aggregate of the sparse res-6 matrix to `res`."""
    t_idx = {c: i for i, c in enumerate(hex_target)}
    parent = np.array([t_idx.get(h3.cell_to_parent(c, res), -1) for c in hex6], dtype=np.int64)
    w = np.maximum(pop6, 1.0)
    w_all = np.bincount(parent[parent >= 0], weights=w[parent >= 0], minlength=len(hex_target))
    p = parent[rows]
    keep = p >= 0
    p, c, t, wr = p[keep], cols[keep], mins[keep].astype(np.float64), w[rows[keep]]
    key = p * n_fac + c
    uk, inv = np.unique(key, return_inverse=True)
    s_wt = np.bincount(inv, weights=wr * t)
    s_w = np.bincount(inv, weights=wr)
    pi, ci = uk // n_fac, uk % n_fac
    out = np.full((len(hex_target), n_fac), np.inf, dtype=np.float32)
    out[pi, ci] = ((s_wt + (w_all[pi] - s_w) * CUTOFF_MIN) / w_all[pi]).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*", help="ISO3 codes (default: every country in tt_jobs/)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sec = tomllib.loads((_ROOT / ".streamlit" / "secrets.toml").read_text())["traveltime"]
    _SPARSE.mkdir(parents=True, exist_ok=True)
    iso3s = args.countries or sorted(p.name.split("_")[0] for p in _JOBS.glob("*_res6.parquet"))
    total_req, failed, t0 = 0, [], time.time()

    for iso3 in iso3s:
        obj = pycountry.countries.get(alpha_3=iso3)
        jobs6 = _JOBS / f"{iso3}_res6.parquet"
        if obj is None or not jobs6.exists():
            continue
        ll = _locs(obj.name)
        if not ll:
            continue  # no facilities: nothing to fetch
        h = _hash(ll)
        sp = _SPARSE / f"{iso3}_res{RES}_{HOURS}h_{h}_{MODE}.npz"
        g6 = pd.read_parquet(jobs6)
        hex6 = list(g6["h3"])
        n_req = -(-len(ll) // _MAX_SEARCHES_PER_REQUEST)

        if sp.exists():
            z = np.load(sp)
            rows, cols, mins = z["rows"], z["cols"], z["minutes"]
            status = "cached"
        else:
            total_req += n_req
            print(f"{iso3} {obj.name[:28]:28} {len(hex6):>8} res-6 hexes  {len(ll):>4} facilities  "
                  f"{n_req:>3} requests", flush=True)
            if args.dry_run:
                continue
            try:
                rows, cols, mins = fetch_sparse(hex6, ll, sec["app_id"], sec["api_key"])
            except RuntimeError as exc:
                failed.append((iso3, str(exc)))
                print(f"    failed, not saved: {str(exc)[:140]}", flush=True)
                continue
            bad = getattr(fetch_sparse, "unroutable", [])
            if bad:
                print(f"    unroutable facilities (column left unreachable): "
                      f"{[(j, ll[j]) for j in bad]}", flush=True)
            np.savez_compressed(sp, rows=rows, cols=cols, minutes=mins,
                                n_hex=len(hex6), n_fac=len(ll), unroutable=np.array(bad, np.int32))
            save_store(iso3, MODE, hex6, g6["population"].to_numpy(), ll, rows, cols, mins,
                       unroutable=bad, cutoff_min=CUTOFF_MIN, departure="Wednesday 08:00 local time at each facility",
                       fetched=time.strftime("%Y-%m-%d"),
                       source="TravelTime /v4/h3 (one-to-many), scripts/fetch_tt_res6.py",
                       facilities="IAEA DIRAC, linac_data/Database_DIRAC_2026_fixed.csv")
            status = "fetched"
        if args.dry_run:
            continue

        pop6 = g6["population"].to_numpy(np.float64)
        for res in (5, 4, 3):
            jobs = _JOBS / f"{iso3}_res{res}.parquet"
            if not jobs.exists():
                continue
            out = _CACHE / f"{iso3}_res{res}_{HOURS}h_{h}_{MODE}.npz"
            if out.exists():
                marker = _CACHE / ".from_res6" / out.name
                if marker.exists():
                    continue  # already built from this res-6 fetch
                _NATIVE.mkdir(exist_ok=True)
                shutil.move(str(out), _NATIVE / out.name)
            dense = aggregate(rows, cols, mins, pop6, hex6, list(pd.read_parquet(jobs)["h3"]), res, len(ll))
            np.savez_compressed(out, matrix=dense)
            (_CACHE / ".from_res6").mkdir(exist_ok=True)
            (_CACHE / ".from_res6" / out.name).touch()
        reach = len(np.unique(rows)) / max(len(hex6), 1)
        print(f"    {status}: {len(mins):,} reachable pairs; {reach:.0%} of res-6 hexes reach a facility "
              f"within {CUTOFF_MIN:.0f} min  [{(time.time() - t0) / 60:.0f} min elapsed]", flush=True)

    print(f"\nTotal API requests {'needed' if args.dry_run else 'made'}: {total_req}")
    if failed:
        print(f"Failed ({len(failed)}):", "; ".join(f"{i}: {e[:80]}" for i, e in failed))


if __name__ == "__main__":
    main()
