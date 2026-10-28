"""
Clean, geocode and audit a DIRAC facility export.

Pipeline (every step is logged so the methodology can be reported exactly):

 1. Load the tidy export (default: inputs/dirac/Database_DIRAC_2026.csv,
    the October 2026 DIRAC export in the 2025 column layout, in which centres
    matched to the previous export already carry its coordinates).
 2. Clean: strip whitespace; drop rows DIRAC itself labels "duplicated / do not
    use"; drop exact duplicate rows (all fields identical), keeping the first.
 3. Validate every existing coordinate against the country's polygon
    (assets/country_borders_50m.geojson, Natural Earth 1:50m, with a tolerance
    buffer for coastal sites). Coordinates outside their country are discarded
    and the row is queued for geocoding.
 4. Geocode every row without a coordinate with OpenStreetMap Nominatim
    (public endpoint, 1 request/s, identified User-Agent), in up to three
    passes: operator + city + country; city + country (city-centroid
    placement); city + country without the country-code filter. Each result is
    validated against the country polygon; failures fall through to the next
    pass. Rows that fail every pass are written out for manual placement.
 5. Write the final facility file and an audit log with one row per centre
    (method, pass, result label, coordinate, polygon check) plus a summary.

Outputs (default folder inputs/dirac/):
    Database_DIRAC_2026_fixed.csv   final facility file (same columns as the input)
    geocode_log_2026.csv            per-row audit log
    geocode_manual_2026.csv         rows needing manual placement (empty if none)
    GEOCODING_SUMMARY.md            counts for the methods section

Run on the server (network + ~1 s per request; 700 rows ≈ 15–30 min):
    scripts/remote_run.sh python scripts/geocode_dirac.py
Options: --input, --outdir, --limit N (dry run on the first N rows needing geocoding),
         --email you@example.org (included in the User-Agent, as Nominatim's policy asks).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
import requests
from shapely.geometry import Point, shape
from shapely.prepared import prep

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent

NOMINATIM = "https://nominatim.openstreetmap.org/search"
RATE_SECONDS = 1.05
POLY_TOLERANCE_DEG = 0.15        # ~15 km buffer for coastal / border sites
MACHINE_COL = "He Photon And Electron Beam Rt"

# DIRAC country labels that pycountry cannot resolve directly.
_ALIASES = {
    "USA": "US", "Russia": "RU", "South Korea": "KR", "North Korea": "KP", "Iran": "IR",
    "Syria": "SY", "Vietnam": "VN", "Bolivia": "BO", "Venezuela": "VE", "Moldova": "MD",
    "Tanzania": "TZ", "Taiwan": "TW", "Macedonia": "MK", "Republic of Ireland": "IE",
    "Reunion": "RE", "Turkey": "TR", "Cote D’Ivoire": "CI", "Cote D'Ivoire": "CI",
    "Democratic Republic of Congo": "CD", "Laos": "LA", "Czech Republic": "CZ",
    "Cape Verde": "CV", "East Timor": "TL", "The Gambia": "GM", "Bosnia - Herzegovina": "BA",
    "Kingdom of Bahrain": "BH", "Brunei": "BN", "Curacao": "CW", "State of Palestine": "PS",
    "Palestine": "PS", "Kosovo": "XK", "United Kingdom": "GB", "Hong Kong": "HK",
    "Macao": "MO", "Micronesia": "FM", "Libya": "LY", "Congo": "CG", "Brunei Darussalam": "BN",
}


def iso2(country: str) -> str | None:
    c = str(country).strip()
    if c in _ALIASES:
        return _ALIASES[c]
    try:
        return pycountry.countries.lookup(c).alpha_2
    except LookupError:
        return None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.split())


class CountryPolygons:
    def __init__(self, path: Path):
        g = json.loads(path.read_text())
        self.polys = {}
        for f in g["features"]:
            code = (f["properties"].get("iso_a2") or "").upper()
            if code and code != "-99":
                geom = shape(f["geometry"])
                self.polys.setdefault(code, []).append((geom, prep(geom.buffer(POLY_TOLERANCE_DEG))))

    def contains(self, code: str | None, lat: float, lon: float) -> bool | None:
        """True/False if the country polygon is known, None if it is not."""
        if not code or code not in self.polys:
            return None
        p = Point(lon, lat)
        return any(buf.contains(p) for _g, buf in self.polys[code])


class Nominatim:
    def __init__(self, email: str):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = f"RadMaps-geocoder/1.0 ({email})"
        self._last = 0.0

    def query(self, q: str, countrycode: str | None):
        wait = RATE_SECONDS - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {"q": q, "format": "jsonv2", "limit": 1, "addressdetails": 1}
        if countrycode:
            params["countrycodes"] = countrycode.lower()
        for attempt in range(3):
            try:
                r = self.s.get(NOMINATIM, params=params, timeout=30)
                self._last = time.time()
                if r.status_code == 200:
                    j = r.json()
                    return j[0] if j else None
                time.sleep(5 * (attempt + 1))
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(_ROOT / "inputs" / "dirac" / "Database_DIRAC_2026.csv"))
    ap.add_argument("--outdir", default=str(_ROOT / "inputs" / "dirac"))
    ap.add_argument("--polygons", default=str(_ROOT / "assets" / "country_borders_50m.geojson"))
    ap.add_argument("--limit", type=int, default=None, help="dry run: geocode only the first N rows needing it")
    ap.add_argument("--email", default="radmaps@generality.org")
    ap.add_argument("--regeocode-all", action="store_true",
                    help="geocode every record afresh; previous coordinates are kept only for comparison "
                         "and as a fallback where the new geocode fails")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str)
    n_in = len(df)
    for c in ("Country", "City", "Operator Name"):
        df[c] = df[c].fillna("").map(_norm)

    # ---- 2. cleaning -------------------------------------------------------
    dup_flag = df["Operator Name"].str.lower().str.contains(r"^duplicated|duplicated[.:]", regex=True)
    n_flagged = int(dup_flag.sum())
    df = df[~dup_flag]
    exact_dup = df.duplicated(keep="first")
    n_exact_dup = int(exact_dup.sum())
    df = df[~exact_dup].reset_index(drop=True)

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["iso2"] = df["Country"].map(iso2)
    n_unresolved_country = int(df["iso2"].isna().sum())

    # ---- 3. validate existing coordinates ----------------------------------
    polys = CountryPolygons(Path(args.polygons))
    has = df["Latitude"].notna() & df["Longitude"].notna()
    inside = []
    for i, r in df.iterrows():
        if not has[i]:
            inside.append(None); continue
        inside.append(polys.contains(r["iso2"], r["Latitude"], r["Longitude"]))
    df["coord_in_country"] = inside
    n_carried = int(has.sum())
    n_outside = int(sum(1 for v in inside if v is False))
    n_unchecked = int(sum(1 for v, h in zip(inside, has) if h and v is None))
    log = pd.DataFrame({
        "row": df.index, "Country": df["Country"], "City": df["City"], "Operator Name": df["Operator Name"],
        "status": ["carried" if h else "missing" for h in has],
        "method": ["previous export" if h else "" for h in has],
        "pass": 0, "query": "", "result_label": "", "lat": df["Latitude"], "lon": df["Longitude"],
        "in_country": df["coord_in_country"],
    })
    bad = has & (df["coord_in_country"] == False)  # noqa: E712
    df.loc[bad, ["Latitude", "Longitude"]] = float("nan")
    log.loc[bad, "status"] = "outside country - discarded"
    log["prev_lat"] = df["Latitude"]; log["prev_lon"] = df["Longitude"]
    if args.regeocode_all:
        df.loc[:, ["Latitude", "Longitude"]] = float("nan")
        log.loc[has, "status"] = "re-geocoded (previous coordinate kept for comparison)"

    # ---- 4. geocode --------------------------------------------------------
    todo = df.index[df["Latitude"].isna() | df["Longitude"].isna()].tolist()
    if args.limit:
        todo = todo[: args.limit]
    geo = Nominatim(args.email)
    counts = {"operator+city+country": 0, "city+country": 0, "city+country (no code)": 0, "failed": 0}
    print(f"geocoding {len(todo)} rows …", flush=True)
    for k, i in enumerate(todo, 1):
        r = df.loc[i]
        code = r["iso2"]
        passes = [
            ("operator+city+country", f"{r['Operator Name']}, {r['City']}, {r['Country']}", code),
            ("city+country", f"{r['City']}, {r['Country']}", code),
            ("city+country (no code)", f"{r['City']}, {r['Country']}", None),
        ]
        placed = False
        for p_no, (method, q, cc) in enumerate(passes, 1):
            if not q.strip(", "):
                continue
            res = geo.query(q, cc)
            if not res:
                continue
            lat, lon = float(res["lat"]), float(res["lon"])
            ok = polys.contains(code, lat, lon)
            res_cc = (res.get("address", {}) or {}).get("country_code", "").upper()
            if ok is False or (ok is None and code and res_cc and res_cc != code):
                continue
            df.loc[i, ["Latitude", "Longitude"]] = (lat, lon)
            log.loc[i, ["status", "method", "pass", "query", "result_label", "lat", "lon", "in_country"]] = (
                "geocoded", method, p_no, q, res.get("display_name", "")[:160], lat, lon, ok)
            counts[method] += 1
            placed = True
            break
        if not placed and args.regeocode_all and pd.notna(log.loc[i, "prev_lat"]) and log.loc[i, "in_country"] is not False:
            df.loc[i, ["Latitude", "Longitude"]] = (log.loc[i, "prev_lat"], log.loc[i, "prev_lon"])
            log.loc[i, ["status", "method", "pass", "lat", "lon"]] = (
                "geocoded", "previous coordinate (new geocode failed)", 9, log.loc[i, "prev_lat"], log.loc[i, "prev_lon"])
            counts["previous coordinate"] = counts.get("previous coordinate", 0) + 1
            placed = True
        if not placed:
            log.loc[i, ["status", "method"]] = ("failed", "manual placement required")
            counts["failed"] += 1
        if k % 25 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}  {counts}", flush=True)

    # ---- 5. outputs --------------------------------------------------------
    cols = ["Country", "City", "Operator Name", MACHINE_COL, "Proton Ion Therapy", "XRay Generator",
            "Brachy Therapy Inc El", "Last Update", "Latitude", "Longitude"]
    out = df[cols].copy()
    out.to_csv(outdir / "Database_DIRAC_2026_fixed.csv", index=False)
    log.to_csv(outdir / "geocode_log_2026.csv", index=False)
    manual = log[log["status"] == "failed"][["row", "Country", "City", "Operator Name"]]
    manual.to_csv(outdir / "geocode_manual_2026.csv", index=False)

    # agreement between the new geocode and the previous coordinate, where both exist
    both = log["prev_lat"].notna() & log["lat"].notna() & (log["status"] == "geocoded")
    if both.any():
        la1, lo1 = np.radians(log.loc[both, "prev_lat"].astype(float)), np.radians(log.loc[both, "prev_lon"].astype(float))
        la2, lo2 = np.radians(log.loc[both, "lat"].astype(float)), np.radians(log.loc[both, "lon"].astype(float))
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        dist = 2 * 6371.0088 * np.arcsin(np.sqrt(a))
        log.loc[both, "shift_km_vs_previous"] = dist.values
        agree = (f"Agreement with previous coordinates ({int(both.sum())} centres): median shift "
                 f"{np.median(dist):.1f} km; within 1 km {100 * (dist <= 1).mean():.1f}%; within 5 km "
                 f"{100 * (dist <= 5).mean():.1f}%; within 25 km {100 * (dist <= 25).mean():.1f}%; "
                 f"more than 50 km {int((dist > 50).sum())} centres.")
    else:
        agree = ""
    log.to_csv(outdir / "geocode_log_2026.csv", index=False)  # re-save with the shift column
    m = pd.to_numeric(df[MACHINE_COL], errors="coerce")
    n_final = len(df)
    n_with_coord = int(df["Latitude"].notna().sum())
    summary = f"""# DIRAC 2026 geocoding summary ({_dt.date.today().isoformat()})

Input rows: {n_in}
Dropped — DIRAC-flagged duplicates ("duplicated / do not use"): {n_flagged}
Dropped — exact duplicate rows (all fields identical): {n_exact_dup}
Rows after cleaning: {n_final}  (MVMs: {int(m.fillna(0).sum())}; blank MVM count: {int(m.isna().sum())}; zero MVM: {int((m == 0).sum())})
Countries not resolvable to an ISO code (geocoded without country filter): {n_unresolved_country}

Coordinates carried over from the 2025 file: {n_carried}
  of which outside their country polygon (discarded, re-geocoded): {n_outside}
  of which not checkable (no polygon for the country code): {n_unchecked}
Rows geocoded in this run: {len(todo)}{' (dry run, limited)' if args.limit else ''}
  operator + city + country: {counts['operator+city+country']}
  city + country (city-centroid placement): {counts['city+country']}
  city + country without country filter: {counts['city+country (no code)']}
  previous coordinate kept because the new geocode failed: {counts.get('previous coordinate', 0)}
  failed — manual placement required: {counts['failed']}
Rows with coordinates at the end: {n_with_coord} / {n_final}
{agree}

Method: Nominatim (OpenStreetMap) public API, 1 request/s, jsonv2, limit 1, countrycodes filter;
every result validated as inside the Natural Earth 1:50m polygon of the stated country with a
{POLY_TOLERANCE_DEG}° buffer. Audit log: geocode_log_2026.csv (one row per centre).
"""
    (outdir / "GEOCODING_SUMMARY.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
