"""
fix_dirac_coordinates.py
------------------------
Repair missing and suspect coordinates in Database_DIRAC.csv using the
Nominatim geocoder (OpenStreetMap).

What it does
------------
1. Flags rows with missing (NaN) coordinates.
2. Flags rows whose coordinates fall outside the country's bounding box
   (likely copy-paste errors / wrong facility matched).
3. For each flagged row, attempts geocoding in three passes:
     a) "Operator Name, City, Country"
     b) "City, Country"
     c) Country centroid as last-resort (coord marked as low-confidence)
4. Validates the geocoded result is inside the country bbox before accepting it.
5. Writes a corrected CSV and a separate report CSV showing what was changed.

Usage
-----
    python fix_dirac_coordinates.py [--dry-run] [--country COUNTRY]

    --dry-run    Print changes without saving.
    --country    Limit processing to a single country (useful for testing).

Output
------
    c_probability_of_access/linac/Database_DIRAC_fixed.csv   — corrected data
    c_probability_of_access/linac/fix_report.csv             — what changed
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_CSV  = Path("c_probability_of_access/linac/Database_DIRAC.csv")
OUTPUT_CSV = Path("c_probability_of_access/linac/Database_DIRAC_fixed.csv")
REPORT_CSV = Path("c_probability_of_access/linac/fix_report.csv")

# Nominatim requires a meaningful user-agent string
GEOCODER = Nominatim(user_agent="dirac_coord_fixer/1.0")

# Seconds between geocoding requests — Nominatim rate-limits to 1 req/s
REQUEST_DELAY = 1.1

# ---------------------------------------------------------------------------
# Country bounding boxes  {country_name_in_csv: (lat_min, lat_max, lon_min, lon_max)}
# Covers every country with ≥1 suspect row found in initial audit.
# ---------------------------------------------------------------------------

_COUNTRY_BBOX: dict[str, tuple[float, float, float, float]] = {
    # lat_min, lat_max, lon_min, lon_max
    "Afghanistan":         (29.4,  38.5,  60.5,  74.9),
    "Albania":             (39.6,  42.7,  19.3,  21.1),
    "Algeria":             (18.9,  37.1,  -8.7,   9.0),
    "Angola":              (-18.0,  -4.4,  11.7,  24.1),
    "Argentina":           (-55.0, -21.8, -73.6, -53.6),
    "Armenia":             (38.8,  41.3,  43.5,  46.6),
    "Australia":           (-43.7, -10.7, 113.3, 153.6),
    "Austria":             (46.4,  49.0,   9.5,  17.2),
    "Azerbaijan":          (38.3,  41.9,  44.8,  51.0),
    "Bahrain":             (25.8,  26.3,  50.4,  50.8),
    "Bangladesh":          (20.7,  26.6,  88.0,  92.7),
    "Belarus":             (51.3,  56.2,  23.2,  32.8),
    "Belgium":             (49.5,  51.5,   2.5,   6.4),
    "Bolivia":             (-22.9, -9.7,  -69.6, -57.5),
    "Bosnia and Herzegovina": (42.6, 45.3, 15.7, 19.7),
    "Brazil":              (-33.8,   5.3, -73.9, -34.8),
    "Bulgaria":            (41.2,  44.2,  22.4,  28.6),
    "Cambodia":            (10.4,  14.7, 102.3, 107.6),
    "Cameroon":            ( 1.7,  13.1,   8.5,  16.2),
    "Canada":              (41.7,  83.1, -141.0, -52.6),
    "Chile":               (-55.9, -17.5, -75.7, -66.4),
    "China":               (18.2,  53.6,  73.5, 134.8),
    "Colombia":            (-4.2,  12.5, -79.0, -66.8),
    "Croatia":             (42.4,  46.6,  13.5,  19.5),
    "Cuba":                (19.8,  23.3, -85.0, -74.1),
    "Cyprus":              (34.6,  35.7,  32.3,  34.6),
    "Czech Republic":      (48.6,  51.1,  12.1,  18.9),
    "Democratic Republic of the Congo": (-13.5, 5.4, 12.2, 31.3),
    "Denmark":             (54.6,  57.8,   8.1,  15.2),
    "Ecuador":             (-5.0,   1.5, -80.9, -75.2),
    "Egypt":               (22.0,  31.7,  24.7,  37.1),
    "Ethiopia":            ( 3.4,  15.0,  33.0,  48.0),
    "Finland":             (59.8,  70.1,  20.0,  31.6),
    "France":              (41.3,  51.1,  -5.1,   9.6),
    "Georgia":             (41.1,  43.6,  40.0,  46.7),
    "Germany":             (47.3,  55.1,   5.9,  15.0),
    "Ghana":               ( 4.7,  11.2,  -3.3,   1.2),
    "Greece":              (35.0,  41.8,  19.4,  28.3),
    "Guatemala":           (13.7,  17.8, -92.2, -88.2),
    "Hungary":             (45.7,  48.6,  16.1,  22.9),
    "India":               ( 6.5,  35.7,  67.0,  97.4),
    "Indonesia":           (-11.1,   5.9,  95.0, 141.0),
    "Iran":                (25.1,  39.8,  44.1,  63.3),
    "Iraq":                (29.1,  37.4,  38.8,  48.6),
    "Ireland":             (51.4,  55.4, -10.5,  -6.0),
    "Israel":              (29.5,  33.3,  34.2,  35.9),
    "Italy":               (36.6,  47.1,   6.6,  18.5),
    "Japan":               (24.0,  45.6, 122.9, 153.0),
    "Jordan":              (29.2,  33.4,  35.0,  39.3),
    "Kazakhstan":          (40.6,  55.5,  50.3,  87.4),
    "Kenya":               (-4.7,   5.0,  33.9,  42.0),
    "North Korea":         (37.7,  42.7, 124.2, 130.7),
    "South Korea":         (33.1,  38.6, 124.6, 129.6),
    "Kuwait":              (28.5,  30.1,  46.5,  48.4),
    "Lebanon":             (33.1,  34.7,  35.1,  36.6),
    "Libya":               (19.5,  33.2,   9.3,  25.2),
    "Malaysia":            ( 0.9,   7.4, 100.1, 119.3),
    "Mexico":              (14.5,  32.7, -117.1, -86.7),
    "Moldova":             (45.5,  48.5,  26.6,  30.2),
    "Morocco":             (27.7,  35.9, -13.2,  -1.0),
    "Mozambique":          (-26.9, -10.5,  30.2,  40.8),
    "Myanmar":             (10.0,  28.5,  92.2, 101.2),
    "Nepal":               (26.4,  30.4,  80.1,  88.2),
    "Netherlands":         (50.8,  53.6,   3.4,   7.2),
    "New Zealand":         (-47.3, -34.4, 166.4, 178.6),
    "Nigeria":             ( 4.3,  13.9,   2.7,  14.7),
    "Norway":              (57.8,  71.2,   4.6,  31.2),
    "Oman":                (16.6,  26.4,  51.8,  59.8),
    "Pakistan":            (23.7,  37.1,  60.9,  77.1),
    "Palestine":           (31.2,  32.6,  34.2,  35.7),
    "Panama":              ( 7.2,   9.7, -83.0, -77.2),
    "Paraguay":            (-27.6, -19.3, -62.7, -54.3),
    "Peru":                (-18.4,  -0.0, -81.3, -68.7),
    "Philippines":         ( 4.6,  21.1, 116.9, 126.6),
    "Poland":              (49.0,  54.9,  14.1,  24.2),
    "Portugal":            (30.0,  42.2, -31.3,  -6.2),  # includes Azores & Madeira
    "Qatar":               (24.5,  26.2,  50.8,  51.6),
    "Republic of Ireland": (51.4,  55.4, -10.5,  -6.0),
    "Romania":             (43.6,  48.3,  20.3,  30.0),
    "Russia":              (41.2,  82.0,  19.6, 190.0),
    "Saudi Arabia":        (16.4,  32.2,  34.6,  55.7),
    "Senegal":             (12.3,  16.7, -17.5, -11.4),
    "Serbia":              (41.9,  46.2,  18.8,  23.0),
    "Singapore":           ( 1.2,   1.5, 103.6, 104.0),
    "Slovakia":            (47.7,  49.6,  16.8,  22.6),
    "Slovenia":            (45.4,  46.9,  13.4,  16.6),
    "South Africa":        (-34.8, -22.1,  16.5,  33.0),
    "Spain":               (27.6,  43.8, -18.2,   4.3),  # includes Canary Islands
    "Sri Lanka":           ( 5.9,   9.8,  79.7,  81.9),
    "Sudan":               (10.0,  22.2,  24.0,  38.6),
    "Sweden":              (55.3,  69.1,  11.1,  24.2),
    "Switzerland":         (45.8,  47.8,   5.9,  10.5),
    "Syria":               (32.3,  37.3,  35.7,  42.4),
    "Taiwan":              (21.9,  25.3, 120.0, 122.1),
    "Tanzania":            (-11.7,  -1.0,  29.3,  40.4),
    "Thailand":            ( 5.6,  20.5,  97.3, 105.7),
    "Tunisia":             (30.2,  37.5,   7.5,  11.6),
    "Turkey":              (36.0,  42.1,  26.0,  44.8),
    "Uganda":              (-1.5,   4.2,  29.6,  35.0),
    "Ukraine":             (44.4,  52.4,  22.1,  40.2),
    "United Arab Emirates":(22.6,  26.1,  51.6,  56.4),
    "United Kingdom":      (49.9,  61.0,  -8.2,   2.0),
    "USA":                 (18.9,  71.4, -179.2, -66.9),  # includes Hawaii & Puerto Rico
    "Uruguay":             (-35.0, -30.1, -58.4, -53.2),
    "Uzbekistan":          (37.2,  45.6,  55.9,  73.2),
    "Venezuela":           ( 0.6,  12.2, -73.4, -59.8),
    "Vietnam":             ( 8.3,  23.4, 102.2, 109.5),
    "Yemen":               (12.1,  19.0,  42.5,  54.5),
    "Zambia":              (-18.1,  -8.2,  22.0,  33.7),
    "Zimbabwe":            (-22.4, -15.6,  25.2,  33.1),
}

# Countries to skip entirely (no known bbox or not worth processing)
_SKIP_COUNTRIES: set[str] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _geocode(query: str) -> tuple[float, float] | None:
    """Return (lat, lon) or None; respects rate limit."""
    try:
        time.sleep(REQUEST_DELAY)
        loc = GEOCODER.geocode(query, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass
    return None


def _best_coords(
    name: str,
    city: str,
    country: str,
    bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, str] | None:
    """Try geocoding strategies; return (lat, lon, method) or None."""
    strategies = [
        (f"{name}, {city}, {country}", "name+city+country"),
        (f"{city}, {country}", "city+country"),
    ]
    for query, method in strategies:
        result = _geocode(query)
        if result is None:
            continue
        lat, lon = result
        if bbox is not None and not _in_bbox(lat, lon, bbox):
            # Nominatim returned something outside expected country — discard
            continue
        return lat, lon, method
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False, filter_country: str | None = None) -> None:
    df = pd.read_csv(INPUT_CSV)
    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    report_rows: list[dict] = []
    changed = 0

    for idx, row in df.iterrows():
        country = str(row.get("Country", "")).strip()
        if filter_country and country != filter_country:
            continue

        lat = row["Latitude"]
        lon = row["Longitude"]
        bbox = _COUNTRY_BBOX.get(country)
        name = str(row.get("Operator Name", "")).strip()
        city = str(row.get("City", "")).strip()

        needs_fix = False
        reason = ""

        if pd.isna(lat) or pd.isna(lon):
            needs_fix = True
            reason = "missing"
        elif bbox is not None and not _in_bbox(lat, lon, bbox):
            needs_fix = True
            reason = f"outside bbox (was {lat:.4f},{lon:.4f})"

        if not needs_fix:
            continue

        print(f"[{idx:5d}] {country} | {city} | {name[:50]} — {reason}")
        result = _best_coords(name, city, country, bbox)

        if result is None:
            print(f"         -> no fix found")
            report_rows.append({
                "row": idx, "country": country, "city": city,
                "operator": name, "reason": reason,
                "old_lat": lat, "old_lon": lon,
                "new_lat": None, "new_lon": None, "method": "FAILED",
            })
            continue

        new_lat, new_lon, method = result
        print(f"         -> ({new_lat:.4f}, {new_lon:.4f}) via '{method}'")
        report_rows.append({
            "row": idx, "country": country, "city": city,
            "operator": name, "reason": reason,
            "old_lat": lat, "old_lon": lon,
            "new_lat": new_lat, "new_lon": new_lon, "method": method,
        })

        if not dry_run:
            df.at[idx, "Latitude"]  = round(new_lat, 7)
            df.at[idx, "Longitude"] = round(new_lon, 7)
        changed += 1

    report_df = pd.DataFrame(report_rows)
    total = len(report_rows)
    fixed = len(report_df[report_df["method"] != "FAILED"]) if total else 0
    print(f"\nDone. {total} rows flagged, {fixed} fixed, {total-fixed} unresolved.")

    if not dry_run:
        df.to_csv(OUTPUT_CSV, index=False)
        report_df.to_csv(REPORT_CSV, index=False)
        print(f"Saved corrected CSV → {OUTPUT_CSV}")
        print(f"Saved fix report    → {REPORT_CSV}")
    else:
        print("(dry-run — nothing saved)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix DIRAC coordinate errors using geocoding.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without saving")
    parser.add_argument("--country", default=None, help="Limit to one country (e.g. 'India')")
    args = parser.parse_args()
    main(dry_run=args.dry_run, filter_country=args.country)
