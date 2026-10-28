"""
Second-stage geocoding pass: resolve ambiguous city-level placements.

scripts/geocode_dirac.py places most centres from a single "City, Country"
query (Nominatim, limit 1). Where the city name is shared by several places
(Hiram GA vs Hiram ME; Jinan vs Jin'an district) that top hit can be the
wrong one, and occasionally it is a non-settlement feature (a stadium, a
road, a river). This pass re-queries the rows where that is plausible and
records exactly what was done, so the method can be described per centre.

Scope (rows whose current method is a city-level hit):
  * the placement is > SHIFT_KM from the coordinate in the previous export, or
  * there is no previous coordinate to compare with, or
  * the result label names a non-settlement feature.

Resolution, in order:
  A. "Operator name, Country" (limit 5): accept the first hospital/clinic-type
     result inside the country that is within NEAR_KM of the previous
     coordinate (or any such result when there is no previous coordinate).
  B. "City, Country" (limit 10): keep only settlement-type candidates inside
     the country; take the one nearest the previous coordinate if it is within
     NEAR_KM, or the top-ranked one when there is no previous coordinate.
  C. Otherwise keep the previous coordinate (it passed the polygon check) and
     mark the row for review; rows with no previous coordinate keep the
     first-stage placement and are marked for review.

The first-stage placement is preserved in the log (first_method / first_lat /
first_lon) and the audit columns are updated in place. Run on the server:

    python scripts/geocode_disambiguate.py --email you@example.org
    python scripts/geocode_disambiguate.py --limit 12   # dry run, prints decisions
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent

from scripts.geocode_dirac import NOMINATIM, RATE_SECONDS, CountryPolygons, iso2  # noqa: E402

SHIFT_KM = 25.0      # city-level hit further than this from the previous coordinate is re-examined
NEAR_KM = 50.0       # a candidate within this distance of the previous coordinate is taken as confirmed
SETTLEMENT_TYPES = {"city", "town", "village", "municipality", "hamlet", "suburb", "city_district",
                    "district", "borough", "quarter", "neighbourhood", "locality", "county", "township"}
FACILITY_TYPES = {"hospital", "clinic", "doctors", "healthcare", "nursing_home", "medical", "university",
                  "college", "research_institute", "institute"}
NON_SETTLEMENT = re.compile(
    r"\b(Stadium|Road|Street|Drive|Avenue|Boulevard|River|Rio |Río |Lake|Creek|Mountain|Airport|"
    r"Park|Bridge|Railway|Station|Reservoir|Canal|Island|Beach|Bay)\b", re.I)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class NominatimMulti:
    def __init__(self, email: str):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = f"RadMaps-geocoder/1.0 ({email})"
        self._last = 0.0

    def query(self, q: str, countrycode: str | None, limit: int):
        wait = RATE_SECONDS - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {"q": q, "format": "jsonv2", "limit": limit, "addressdetails": 1}
        if countrycode:
            params["countrycodes"] = countrycode.lower()
        for attempt in range(3):
            try:
                r = self.s.get(NOMINATIM, params=params, timeout=30)
                self._last = time.time()
                if r.status_code == 200:
                    return r.json() or []
                time.sleep(5 * (attempt + 1))
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        return []


def _is_settlement(c: dict) -> bool:
    at = (c.get("addresstype") or c.get("type") or "").lower()
    cls = (c.get("class") or "").lower()
    if cls == "place" and at not in {"house", "postcode"}:
        return True
    if cls == "boundary" and (c.get("type") == "administrative"):
        return True
    return at in SETTLEMENT_TYPES


def _is_facility(c: dict) -> bool:
    cls = (c.get("class") or "").lower()
    typ = (c.get("type") or "").lower()
    at = (c.get("addresstype") or "").lower()
    if cls in {"amenity", "healthcare", "office"} and (typ in FACILITY_TYPES or at in FACILITY_TYPES):
        return True
    return cls == "building" and typ in FACILITY_TYPES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(_ROOT / "inputs" / "dirac"))
    ap.add_argument("--polygons", default=str(_ROOT / "assets" / "country_borders_50m.geojson"))
    ap.add_argument("--limit", type=int, default=None, help="dry run on the first N rows in scope; writes nothing")
    ap.add_argument("--email", default="radmaps@generality.org")
    ap.add_argument("--where", choices=["far", "noprev", "odd"], default=None,
                    help="dry-run helper: restrict the scope to one trigger")
    args = ap.parse_args()
    d = Path(args.dir)

    log = pd.read_csv(d / "geocode_log_2026.csv")
    out = pd.read_csv(d / "Database_DIRAC_2026_fixed.csv", dtype=str)
    assert len(log) == len(out), "log and fixed file are out of step"
    for col, default in (("first_method", ""), ("first_lat", np.nan), ("first_lon", np.nan), ("review", "")):
        if col not in log.columns:
            log[col] = default
    log["shift_km_vs_previous"] = pd.to_numeric(log["shift_km_vs_previous"], errors="coerce")

    city_hit = log["method"].astype(str).str.startswith("city+country")
    has_prev = log["prev_lat"].notna() & log["prev_lon"].notna()
    far = log["shift_km_vs_previous"] > SHIFT_KM
    odd_label = log["result_label"].fillna("").map(lambda s: bool(NON_SETTLEMENT.search(s)))
    scope = (log["status"] == "geocoded") & city_hit & (far | ~has_prev | odd_label)
    if args.where:
        scope &= {"far": far, "noprev": ~has_prev, "odd": odd_label}[args.where]
    rows = log.index[scope].tolist()
    print(f"rows in scope: {len(rows)}  (far {int((scope & far).sum())}, no previous {int((scope & ~has_prev).sum())}, "
          f"odd label {int((scope & odd_label).sum())})", flush=True)
    if args.limit:
        rows = rows[: args.limit]

    polys = CountryPolygons(Path(args.polygons))
    geo = NominatimMulti(args.email)
    counts: dict[str, int] = {}

    def _in_country(code, c) -> bool:
        lat, lon = float(c["lat"]), float(c["lon"])
        ok = polys.contains(code, lat, lon)
        cc = ((c.get("address") or {}).get("country_code") or "").upper()
        if ok is False:
            return False
        if ok is None and code and cc and cc != code:
            return False
        return True

    for k, i in enumerate(rows, 1):
        r = log.loc[i]
        code = iso2(r["Country"])
        prev = (float(r["prev_lat"]), float(r["prev_lon"])) if has_prev[i] else None
        if not r["first_method"]:
            log.loc[i, ["first_method", "first_lat", "first_lon"]] = (r["method"], r["lat"], r["lon"])
        decision = None

        # A. operator name within the country
        q_a = f"{r['Operator Name']}, {r['Country']}"
        for c in geo.query(q_a, code, 5):
            if not _is_facility(c) or not _in_country(code, c):
                continue
            lat, lon = float(c["lat"]), float(c["lon"])
            if prev is None or haversine_km(lat, lon, *prev) <= NEAR_KM:
                decision = ("operator+country" + ("" if prev is None else " (confirmed by previous coordinate)"),
                            q_a, c, lat, lon)
                break

        # B. settlement candidates for the city
        if decision is None:
            q_b = f"{r['City']}, {r['Country']}"
            cands = [c for c in geo.query(q_b, code, 10) if _is_settlement(c) and _in_country(code, c)]
            if cands:
                if prev is not None:
                    best = min(cands, key=lambda c: haversine_km(float(c["lat"]), float(c["lon"]), *prev))
                    dist = haversine_km(float(best["lat"]), float(best["lon"]), *prev)
                    if dist <= NEAR_KM:
                        decision = ("city+country (nearest settlement to previous coordinate)", q_b, best,
                                    float(best["lat"]), float(best["lon"]))
                else:
                    best = cands[0]
                    decision = ("city+country (settlement filter)", q_b, best, float(best["lat"]), float(best["lon"]))

        # C. fall back
        if decision is None:
            if prev is not None:
                method = "previous coordinate (new geocode ambiguous)"
                log.loc[i, ["method", "pass", "query", "result_label", "lat", "lon", "in_country", "review"]] = (
                    method, 12, "", "", prev[0], prev[1], True, "ambiguous: kept previous coordinate")
            else:
                method = log.loc[i, "method"] + " [unverified]"
                log.loc[i, ["method", "review"]] = (method, "no previous coordinate; first-stage placement kept")
        else:
            method, q, c, lat, lon = decision
            log.loc[i, ["method", "pass", "query", "result_label", "lat", "lon", "in_country", "review"]] = (
                method, 11, q, (c.get("display_name") or "")[:160], lat, lon, True, "")
        counts[method] = counts.get(method, 0) + 1
        if prev is not None:
            log.loc[i, "shift_km_vs_previous"] = haversine_km(float(log.loc[i, "lat"]), float(log.loc[i, "lon"]), *prev)
        if args.limit:
            print(f"  [{i}] {r['Country']} | {r['City']} | {r['Operator Name'][:50]} :: {log.loc[i, 'first_method']} "
                  f"({r['shift_km_vs_previous']:.0f} km) -> {method} ({log.loc[i, 'shift_km_vs_previous']:.1f} km) "
                  f"{str(log.loc[i, 'result_label'])[:70]}")
        if k % 25 == 0 or k == len(rows):
            print(f"  {k}/{len(rows)}  {counts}", flush=True)

    if args.limit:
        print("dry run: nothing written")
        return

    # write back
    out.loc[log.index, "Latitude"] = log["lat"].values
    out.loc[log.index, "Longitude"] = log["lon"].values
    out.to_csv(d / "Database_DIRAC_2026_fixed.csv", index=False)
    log.to_csv(d / "geocode_log_2026.csv", index=False)
    log[log["review"] != ""].to_csv(d / "geocode_review_2026.csv", index=False)

    both = log["prev_lat"].notna() & log["lat"].notna() & (log["status"] == "geocoded")
    dist = log.loc[both, "shift_km_vs_previous"].astype(float)
    summary = (
        f"\n## Disambiguation pass ({pd.Timestamp.today():%Y-%m-%d})\n\n"
        f"Rows re-examined: {len(rows)} (city-level hits > {SHIFT_KM:.0f} km from the previous coordinate, "
        f"with no previous coordinate, or with a non-settlement result label).\n"
        f"Outcomes: {counts}\n\n"
        f"Final method counts:\n{log['method'].value_counts().to_string()}\n\n"
        f"Agreement with previous coordinates ({int(both.sum())} centres): median shift {dist.median():.1f} km; "
        f"within 1 km {100 * (dist <= 1).mean():.1f}%; within 5 km {100 * (dist <= 5).mean():.1f}%; "
        f"within 25 km {100 * (dist <= 25).mean():.1f}%; more than 50 km {int((dist > 50).sum())} centres.\n"
        f"Rows marked for review: {int((log['review'] != '').sum())} (geocode_review_2026.csv).\n"
    )
    print(summary)
    with open(d / "GEOCODING_SUMMARY.md", "a") as fh:
        fh.write(summary)


if __name__ == "__main__":
    main()
