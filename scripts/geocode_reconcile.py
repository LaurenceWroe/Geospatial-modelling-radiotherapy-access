"""
Final step of the 2026 DIRAC geocoding: choose each centre's coordinate.

Every centre was geocoded afresh (scripts/geocode_dirac.py --regeocode-all,
then scripts/geocode_disambiguate.py). For centres that already had a curated
coordinate in the 2025 file, the fresh geocode is used as a *check* on that
coordinate rather than as a replacement: an operator-level hit is at least as
precise as the curated point, but a city-level hit is a town centroid, which
is typically a few kilometres from the hospital. The rule is therefore:

  1. operator-level hit (facility found by name)         -> new coordinate
  2. city-level hit within CONFIRM_KM of the curated point -> curated point kept
     (confirmed: same town)
  3. city-level hit further away, or fallback to the previous coordinate
     (new geocode failed / ambiguous)                     -> curated point kept,
     flagged for review
  4. no curated point                                    -> new coordinate
  5. manual fix                                          -> the fixed coordinate

The log gains final_method / final_lat / final_lon; the fixed CSV is written
from them. Light: no network, runs anywhere.

    python scripts/geocode_reconcile.py [--dir inputs/dirac]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent

CONFIRM_KM = 25.0


def _hav(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(_ROOT / "inputs" / "dirac"))
    args = ap.parse_args()
    d = Path(args.dir)
    log = pd.read_csv(d / "geocode_log_2026.csv")
    out = pd.read_csv(d / "Database_DIRAC_2026_fixed.csv", dtype=str)
    assert len(log) == len(out)

    m = log["method"].astype(str)
    lat, lon = pd.to_numeric(log["lat"], errors="coerce"), pd.to_numeric(log["lon"], errors="coerce")
    plat, plon = pd.to_numeric(log["prev_lat"], errors="coerce"), pd.to_numeric(log["prev_lon"], errors="coerce")
    has_prev = plat.notna() & plon.notna()
    has_new = lat.notna() & lon.notna() & (log["status"].astype(str) != "failed")   # failed rows carry a rejected coordinate
    shift = _hav(lat, lon, plat, plon)

    final_method = pd.Series("", index=log.index, dtype=object)
    flat, flon = lat.copy(), lon.copy()

    manual = m.str.startswith("manual")
    operator = m.str.startswith("operator") & ~manual
    city = m.str.startswith("city+country") & ~manual
    prev_fb = m.str.startswith("previous coordinate") & ~manual

    final_method[manual & has_new] = "manual fix"
    final_method[~has_new] = "unplaced"
    sel = operator & has_new
    final_method[sel] = "new geocode: facility found by name"
    sel = city & has_new & ~has_prev
    final_method[sel] = "new geocode: town centroid (no previous coordinate)"
    sel = city & has_new & has_prev & (shift <= CONFIRM_KM)
    final_method[sel] = f"previous coordinate, confirmed by town-level geocode (≤{CONFIRM_KM:.0f} km)"
    flat[sel], flon[sel] = plat[sel], plon[sel]
    sel = city & has_new & has_prev & (shift > CONFIRM_KM)
    final_method[sel] = "previous coordinate kept, town-level geocode disagrees (review)"
    flat[sel], flon[sel] = plat[sel], plon[sel]
    sel = prev_fb & has_new
    final_method[sel] = "previous coordinate kept, new geocode failed or ambiguous (review)"
    # rows the first stage already filled from the previous coordinate carry it in lat/lon
    unlabelled = final_method == ""
    if unlabelled.any():
        print("WARNING: unlabelled rows\n", log.loc[unlabelled, ["Country", "City", "method"]].head(20).to_string())
        final_method[unlabelled & has_new] = "new geocode: other"

    log["final_method"], log["final_lat"], log["final_lon"] = final_method, flat, flon
    log["final_shift_km_vs_previous"] = np.where(has_prev, _hav(flat, flon, plat, plon), np.nan)
    out["Latitude"], out["Longitude"] = flat.values, flon.values
    out.to_csv(d / "Database_DIRAC_2026_fixed.csv", index=False)
    log.to_csv(d / "geocode_log_2026.csv", index=False)
    rev = log[final_method.str.contains("review", regex=False)]
    rev.to_csv(d / "geocode_review_2026.csv", index=False)

    counts = final_method.value_counts()
    moved = log.loc[has_prev & has_new, "final_shift_km_vs_previous"]
    summary = (
        f"\n## Final coordinate choice ({pd.Timestamp.today():%Y-%m-%d})\n\n"
        f"{counts.to_string()}\n\n"
        f"Centres with a previous coordinate: {int(has_prev.sum())}; final coordinate differs from it by "
        f"> 1 km for {int((moved > 1).sum())}, > 25 km for {int((moved > 25).sum())}.\n"
        f"Rows for review: {len(rev)} (geocode_review_2026.csv). Unplaced: {int((final_method == 'unplaced').sum())}.\n"
    )
    print(summary)
    with open(d / "GEOCODING_SUMMARY.md", "a") as fh:
        fh.write(summary)


if __name__ == "__main__":
    main()
