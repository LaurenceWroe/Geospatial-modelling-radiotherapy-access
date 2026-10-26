# DIRAC October 2026 export — geocoding overview

Pipeline (all scripted, one audit row per centre in `geocode_log_2026.csv`):

1. `scripts/geocode_dirac.py --regeocode-all` — every one of the 8,625 cleaned centres geocoded afresh with
   Nominatim (operator + city + country; then city + country), each result validated against the Natural Earth
   country polygon. The coordinates of the 2025 file (7,937 centres after discarding 11 outside their country)
   were kept only for comparison.
2. `scripts/geocode_disambiguate.py` — 2,101 town-level placements re-examined (> 25 km from the 2025 point,
   no 2025 point, or a non-settlement result): operator name within the country, then the settlement candidate
   nearest the 2025 point.
3. `scripts/geocode_manual_fixes.py` — 15 centres with misspelt or non-standard town names resolved by hand.
4. `scripts/geocode_reconcile.py` — final coordinate per centre. The fresh geocode is used to confirm or
   correct the curated 2025 coordinate rather than replace it, because facility-level hits agree with the
   curated points to a median of 0.1 km whereas town centroids are typically a few km from the hospital.

Final coordinate source (8,625 centres, all placed):

| Source | Centres |
|---|---|
| 2025 coordinate, confirmed by a town-level geocode within 25 km | 5,212 |
| New geocode: facility found by name (median 0.1 km from the 2025 point) | 1,758 |
| 2025 coordinate kept: new geocode failed or ambiguous (flagged for review) | 993 |
| New geocode: town centroid (centre not in the 2025 file) | 547 |
| 2025 coordinate kept: town-level geocode disagrees (flagged for review) | 100 |
| Manual fix | 15 |

Relative to the 2025 file the final coordinate moved by more than 1 km for 282 centres and more than 25 km
for 55. `geocode_review_2026.csv` lists the 1,093 flagged rows. First-stage results, before disambiguation,
are preserved in the log as first_method / first_lat / first_lon.

---

# DIRAC 2026 geocoding summary (2026-10-22)

Input rows: 8630
Dropped — DIRAC-flagged duplicates ("duplicated / do not use"): 2
Dropped — exact duplicate rows (all fields identical): 3
Rows after cleaning: 8625  (MVMs: 17226; blank MVM count: 0; zero MVM: 124)
Countries not resolvable to an ISO code (geocoded without country filter): 2

Coordinates carried over from the 2025 file: 7948
  of which outside their country polygon (discarded, re-geocoded): 11
  of which not checkable (no polygon for the country code): 5
Rows geocoded in this run: 8625
  operator + city + country: 1758
  city + country (city-centroid placement): 6711
  city + country without country filter: 34
  previous coordinate kept because the new geocode failed: 107
  failed — manual placement required: 15
Rows with coordinates at the end: 8610 / 8625
Agreement with previous coordinates (7937 centres): median shift 2.4 km; within 1 km 36.3%; within 5 km 64.2%; within 25 km 81.7%; more than 50 km 1162 centres.

Method: Nominatim (OpenStreetMap) public API, 1 request/s, jsonv2, limit 1, countrycodes filter;
every result validated as inside the Natural Earth 1:50m polygon of the stated country with a
0.15° buffer. Audit log: geocode_log_2026.csv (one row per centre).

## Disambiguation pass (2026-10-22)

Rows re-examined: 2101 (city-level hits > 25 km from the previous coordinate, with no previous coordinate, or with a non-settlement result label).
Outcomes: {'city+country (settlement filter)': 525, 'city+country [unverified]': 19, 'previous coordinate (new geocode ambiguous)': 886, 'city+country (nearest settlement to previous coordinate)': 668, 'city+country (no code) [unverified]': 3}

Final method counts:
method
city+country                                                4615
operator+city+country                                       1758
previous coordinate (new geocode ambiguous)                  886
city+country (nearest settlement to previous coordinate)     668
city+country (settlement filter)                             525
previous coordinate (new geocode failed)                     107
city+country (no code)                                        29
city+country [unverified]                                     19
manual placement required                                     15
city+country (no code) [unverified]                            3

Agreement with previous coordinates (7937 centres): median shift 1.0 km; within 1 km 49.4%; within 5 km 79.9%; within 25 km 98.0%; more than 50 km 42 centres.
Rows marked for review: 908 (geocode_review_2026.csv).

Manual resolution (scripts/geocode_manual_fixes.py): 15 rows — corrected spellings re-queried, Puerto Rico and Kosovo coordinates reused from the 2025 file. Final: 8625/8625 rows with coordinates.

## Final coordinate choice (2026-10-22)

previous coordinate, confirmed by town-level geocode (≤25 km)         5212
new geocode: facility found by name                                   1758
previous coordinate kept, new geocode failed or ambiguous (review)     993
new geocode: town centroid (no previous coordinate)                    547
previous coordinate kept, town-level geocode disagrees (review)        100
manual fix                                                              15

Centres with a previous coordinate: 7937; final coordinate differs from it by > 1 km for 282, > 25 km for 55.
Rows for review: 1093 (geocode_review_2026.csv). Unplaced: 0.

## Post-fixes (24 Oct 2026)

Three centres could not start a TravelTime search (HTTP 422, "not near a road") and were mis-geocoded;
coordinates corrected in `Database_DIRAC_2026_fixed.csv` and `final_*` columns of `geocode_log_2026.csv`:

| Country | Centre | Was | Now |
|---|---|---|---|
| Tanzania | Kilimanjaro Christian Medical Center (KCMC) | on Mt Kilimanjaro (-3.0787, 37.4199) | OSM hospital, Moshi (-3.3192, 37.3275) |
| Uzbekistan | Navoiy Regional Oncology Center | region centroid, desert (42.6989, 64.6338) | Navoiy city centroid (40.1035, 65.3734) |
| USA | Ascension Columbia Saint Mary's - Ozaukee | Lake Michigan (43.3820, -87.6186) | OSM hospital, Mequon WI (43.2602, -87.9292) |
| USA | Lake Okeechobee Regional Cancer Center | Lake Okeechobee (26.9690, -80.7976) | Okeechobee city centroid (27.2437, -80.8299) |
