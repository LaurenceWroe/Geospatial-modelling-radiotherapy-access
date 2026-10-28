"""
Convert the resolution-6 sparse fetch (travel_time_cache/res6_sparse/, whose
rows and columns only make sense alongside tt_jobs/ and the DIRAC loader's
facility order) into the self-describing store of data/tt_store.py, then check
that the store reproduces the dense resolution-5/4/3 matrices built at fetch
time.

    python scripts/build_tt_store.py [--countries PRT KEN ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "travel_time_cache"
_SPARSE = _CACHE / "res6_sparse"
_JOBS = _ROOT / "tt_jobs"

from data.tt_store import load_store, save_store  # noqa: E402
from scripts.fetch_tt_res6 import CUTOFF_MIN, MODE, _hash, _locs  # noqa: E402

META = {
    "cutoff_min": CUTOFF_MIN,
    "departure": "Wednesday 08:00 UTC",
    "fetched": "2026-10-24",
    "source": "TravelTime /v4/h3 (one-to-many), scripts/fetch_tt_res6.py",
    "facilities": "IAEA DIRAC October 2026 export, inputs/dirac/Database_DIRAC_2026_fixed.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*")
    args = ap.parse_args()
    files = sorted(_SPARSE.glob(f"*_res6_2h_*_{MODE}.npz"))
    if args.countries:
        files = [f for f in files if f.name[:3] in args.countries]
    worst = 0.0
    for f in files:
        iso3, h = f.name[:3], f.name.split("_")[3]
        ll = _locs(pycountry.countries.get(alpha_3=iso3).name)
        if _hash(ll) != h:
            print(f"{iso3}: facility list no longer matches {f.name}; skipped")
            continue
        z = np.load(f)
        g6 = pd.read_parquet(_JOBS / f"{iso3}_res6.parquet")
        assert int(z["n_hex"]) == len(g6) and int(z["n_fac"]) == len(ll)
        bad = z["unroutable"] if "unroutable" in z.files else []
        save_store(iso3, MODE, list(g6["h3"]), g6["population"].to_numpy(), ll,
                   z["rows"], z["cols"], z["minutes"], unroutable=bad, **META)
        store = load_store(iso3, MODE)
        # the store must reproduce every dense matrix built at fetch time
        for res in (5, 4, 3):
            dense = _CACHE / f"{iso3}_res{res}_2h_{h}_{MODE}.npz"
            if not dense.exists():
                continue
            ref = np.load(dense)["matrix"]
            got = store.matrix(list(pd.read_parquet(_JOBS / f"{iso3}_res{res}.parquet")["h3"]), ll)
            same_inf = np.array_equal(np.isinf(ref), np.isinf(got))
            fin = np.isfinite(ref)
            err = float(np.abs(ref[fin] - got[fin]).max()) if fin.any() else 0.0
            worst = max(worst, err)
            if not same_inf or err > 1e-3:
                print(f"{iso3} res{res}: MISMATCH (reachability equal: {same_inf}, max error {err:.4f} min)")
        print(f"{iso3}: {len(z['minutes']):,} pairs, {len(g6):,} cells, {len(ll)} facilities", flush=True)
    print(f"\nlargest difference from the fetch-time dense matrices: {worst:.2e} min")


if __name__ == "__main__":
    main()
