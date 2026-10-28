"""
Resolve the centres that scripts/geocode_dirac.py could not place automatically.

Each fix is one of:
  query   — a corrected search string (DIRAC city spelling was wrong or non-standard),
            sent to Nominatim and validated by the returned country code;
  reuse   — the coordinate from the 2025 file was correct but failed the polygon
            check only because the territory (Puerto Rico, Kosovo) is a separate
            polygon from the DIRAC country label.

Every fix is appended to geocode_log_2026.csv with method "manual: <reason>"
so the audit trail stays complete. Light (≤ 20 requests); runs locally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_DIR = _ROOT / "inputs" / "dirac"

from scripts.geocode_dirac import Nominatim, iso2  # noqa: E402

# key: (Country, City, Operator Name) as in the cleaned file
FIXES = {
    ("Egypt", "Ezbet El-Nozha", "Specialized Universal Network of Oncology (SUN Oncology Center)"):
        ("query", "Ezbet El Nakhl, Cairo, Egypt", "city spelling (Cairo district, city-centroid)"),
    ("Egypt", "Shebin El Koum", "M.O.C Medical Center Shebin El Koum Private Center"):
        ("query", "Shibin El Kom, Egypt", "city spelling (city-centroid)"),
    ("Germany", "Hildensheim", "Gemeinschaftspraxis fur Strahlentherapie"):
        ("query", "Hildesheim, Germany", "city spelling"),
    ("India", "Sri Gangangagar", "Cancer Treatment Services Hyderabad Private limited"):
        ("query", "Sri Ganganagar, Rajasthan, India", "city spelling"),
    ("Japan", "Chuto-gun", "University of the Ryukyu Hospital"):
        ("query", "University of the Ryukyus Hospital, Nishihara, Okinawa, Japan", "district name"),
    ("Japan", "Shikanuma-shi", "JA Kamitsuga General Hospital"):
        ("query", "Kamitsuga General Hospital, Kanuma, Tochigi, Japan", "city spelling"),
    ("Japan", "Suntou-gun", "National Hospital Organization Shizuoka Medical Center"):
        ("query", "Shizuoka Medical Center, Shimizu, Sunto District, Shizuoka, Japan", "district name"),
    ("Japan", "Tuchiura-shi", "JA Tsuchiura Kyodo General Hospital"):
        ("query", "Tsuchiura, Ibaraki, Japan", "city spelling (city-centroid)"),
    ("Morocco", "El J adida", "Centre International d'Oncologie El Jadida (AKDITAL Group)"):
        ("query", "El Jadida, Morocco", "city spelling"),
    ("Romania", "Loc. Ovidiu, Oras Ovidiu", "Ovidius Clinical Hospital Centre of Oncology and Radiotherapy"):
        ("query", "Ovidiu, Constanta, Romania", "city spelling"),
    ("Serbia", "Pristhina", "University Clinical Centre of Kosovo (UCCK)"):
        ("reuse", None, "Pristina lies in the Kosovo polygon; DIRAC lists it under Serbia"),
    ("State of Palestine", "Nablus", "An-Najah National University Hospital An-Najah Cancer Center"):
        ("query", "Nablus, Palestine", "no polygon for PS; city-centroid, validated by country code"),
    ("State of Palestine", "Ramallah", "Istishari Cancer Center"):
        ("query", "Ramallah, Palestine", "no polygon for PS; city-centroid, validated by country code"),
    ("Ukraine", "Chernivtsy", "Bukovinian clinical oncology center"):
        ("query", "Chernivtsi, Ukraine", "city spelling (city-centroid)"),
    ("USA", "Mayaguez", "RTCI Mayaguez"):
        ("reuse", None, "Puerto Rico is a separate polygon from the US"),
}


def main() -> None:
    fixed = pd.read_csv(_DIR / "Database_DIRAC_2026_fixed.csv", dtype=str)
    log = pd.read_csv(_DIR / "geocode_log_2026.csv")
    prev = pd.read_csv(_ROOT / "inputs" / "dirac" / "Database_DIRAC_2025_fixed.csv", dtype=str)
    geo = Nominatim("laurence.wroe@generality.org")
    failed = log[log["status"] == "failed"]
    print(f"{len(failed)} rows to resolve")
    done = 0
    for _, f in failed.iterrows():
        key = (f["Country"], f["City"], f["Operator Name"])
        fix = FIXES.get(key)
        # tolerate minor whitespace / truncation differences in the key
        if fix is None:
            for k, v in FIXES.items():
                if k[0] == key[0] and k[1] == key[1] and str(key[2]).startswith(k[2][:20]):
                    fix = v; break
        if fix is None:
            print("  NO FIX DEFINED:", key); continue
        kind, q, reason = fix
        i = int(f["row"])
        if kind == "reuse":
            m = prev[(prev["Country"].str.strip() == key[0]) & (prev["City"].str.strip() == key[1])]
            if m.empty:
                print("  no 2025 row for", key); continue
            lat, lon, label = float(m.iloc[0]["Latitude"]), float(m.iloc[0]["Longitude"]), "2025 coordinate"
        else:
            res = geo.query(q, None)
            if not res:
                print("  query failed:", q); continue
            cc = (res.get("address", {}) or {}).get("country_code", "").upper()
            want = iso2(key[0])
            if want and cc and cc != want and not (want == "RS" and cc == "XK") and not (want == "US" and cc == "PR"):
                print(f"  country mismatch for {key}: got {cc}, want {want}"); continue
            lat, lon, label = float(res["lat"]), float(res["lon"]), res.get("display_name", "")[:160]
        fixed.loc[i, ["Latitude", "Longitude"]] = (lat, lon)
        log.loc[i, ["status", "method", "pass", "query", "result_label", "lat", "lon", "in_country"]] = (
            "geocoded", f"manual: {reason}", 4, q or "", label, lat, lon, True)
        done += 1
        print(f"  ok  {key[0]:18} {key[1]:26} -> {lat:.4f}, {lon:.4f}  ({reason})")
    fixed.to_csv(_DIR / "Database_DIRAC_2026_fixed.csv", index=False)
    log.to_csv(_DIR / "geocode_log_2026.csv", index=False)
    n = len(fixed); have = int(pd.to_numeric(fixed["Latitude"], errors="coerce").notna().sum())
    print(f"resolved {done}; coordinates now {have}/{n}")
    with open(_DIR / "GEOCODING_SUMMARY.md", "a") as fh:
        fh.write(f"\nManual resolution (scripts/geocode_manual_fixes.py): {done} rows — corrected spellings "
                 f"re-queried, Puerto Rico and Kosovo coordinates reused from the 2025 file. "
                 f"Final: {have}/{n} rows with coordinates.\n")


if __name__ == "__main__":
    main()
