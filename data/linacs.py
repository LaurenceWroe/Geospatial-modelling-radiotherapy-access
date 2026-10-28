"""
LINAC facility loader — parses DIRAC-format Excel files.

Supported coordinate formats
-----------------------------
- A ``Coordinates`` column containing ``"lat, lon"`` strings.
- Separate ``Latitude`` / ``Longitude`` columns (case-insensitive).

The machine-count column is detected by fuzzy name matching against
``He Photon And Electron Beam Rt``, ``linacs``, ``count``, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pycountry

# RADMAPS_DIRAC_CSV overrides the file (e.g. a newer DIRAC export) for scripts
# and comparisons without changing the default the app and the tests use.
import os as _os
_ROOT = Path(__file__).resolve().parent.parent
_DIRAC_2026  = _ROOT / "inputs" / "dirac" / "Database_DIRAC_2026_fixed.csv"   # October 2026 export, geocoded (default)
DIRAC_CSV_2025 = _ROOT / "inputs" / "dirac" / "Database_DIRAC_2025_fixed.csv"  # July 2025 export as used in the preprint
DIRAC_CSV = (Path(_os.environ["RADMAPS_DIRAC_CSV"]) if _os.environ.get("RADMAPS_DIRAC_CSV")
             else _DIRAC_2026)

# pycountry name → DIRAC CSV country name for known mismatches
_COUNTRY_ALIASES: dict[str, str] = {
    "United States": "USA",
    "United States of America": "USA",
    "Russian Federation": "Russia",
    "Korea, Republic of": "South Korea",
    "Korea, Democratic People's Republic of": "North Korea",
    "Iran, Islamic Republic of": "Iran",
    "Syrian Arab Republic": "Syria",
    "Viet Nam": "Vietnam",
    "Bolivia, Plurinational State of": "Bolivia",
    "Venezuela, Bolivarian Republic of": "Venezuela",
    "Moldova, Republic of": "Moldova",
    "Tanzania, United Republic of": "Tanzania",
    "Taiwan, Province of China": "Taiwan",
    "North Macedonia": "Macedonia",
    "Ireland": "Republic of Ireland",
    "Réunion": "Reunion",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Cote D\u2019Ivoire",
    "Congo, The Democratic Republic of the": "Democratic Republic of Congo",
    "Lao People's Democratic Republic": "Laos",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Timor-Leste": "East Timor",
    "Gambia": "The Gambia",
    "Bosnia and Herzegovina": "Bosnia - Herzegovina",
    "Bahrain": "Kingdom of Bahrain",
    "Brunei Darussalam": "Brunei",
    "Curaçao": "Curacao",
}


def _resolve_dirac_country(country_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return rows matching *country_name*, trying aliases then case-insensitive fallback."""
    # 1. Direct match
    result = df[df["Country"] == country_name]
    if not result.empty:
        return result
    # 2. Alias lookup
    alias = _COUNTRY_ALIASES.get(country_name)
    if alias:
        result = df[df["Country"] == alias]
        if not result.empty:
            return result
    # 3. Case-insensitive fallback
    result = df[df["Country"].str.strip().str.lower() == country_name.strip().lower()]
    if not result.empty:
        return result
    # 4. Punctuation/accent-insensitive fallback ("Côte d'Ivoire" vs "Cote DIvoire",
    #    "Palestine, State of" vs "State of Palestine")
    key = _country_key(country_name)
    return df[df["Country"].map(_country_key) == key]


def _country_key(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().strip()
    if "," in s:                      # "Palestine, State of" -> "State of Palestine"
        head, tail = s.split(",", 1)
        s = f"{tail.strip()} {head.strip()}"
    return "".join(ch for ch in s.lower() if ch.isalnum())


def load_linacs(excel_path: str | Path) -> List[Tuple[float, float, float]]:
    """Parse a DIRAC Excel file and return ``[(lat, lon, n_linacs), ...]``.

    Rows with zero machines are included with weight 1 (present but unknown
    capacity), unless the weight column contains an explicit 0, in which case
    the row is skipped.

    Parameters
    ----------
    excel_path : str or Path
        Path to a DIRAC .xlsx file.

    Returns
    -------
    List of (lat, lon, weight) tuples with at least one entry.

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    ValueError         if no valid locations can be parsed.
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"LINAC file not found: {path}")

    df = pd.read_excel(path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    coord_col = cols.get("coordinates")
    lat_col = cols.get("latitude")
    lon_col = cols.get("longitude")

    # Detect machine-count column
    weight_col = None
    for candidate in (
        "he photon and electron beam rt",
        "linacs",
        "count",
        "machines",
    ):
        if candidate in cols:
            weight_col = cols[candidate]
            break

    pts: List[Tuple[float, float, float]] = []
    for _, row in df.iterrows():
        try:
            if coord_col is not None:
                val = row[coord_col]
                if not isinstance(val, str):
                    continue
                parts = val.split(",")
                lat, lon = float(parts[0].strip()), float(parts[1].strip())
            elif lat_col is not None and lon_col is not None:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            else:
                continue

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            if weight_col is not None and pd.notna(row[weight_col]):
                w = float(row[weight_col])
                if w == 0:
                    continue  # explicitly zero machines — skip
            else:
                w = 1.0

            pts.append((lat, lon, w))
        except Exception:
            continue

    if not pts:
        raise ValueError(f"No valid LINAC locations found in {path}")

    return pts


def load_linacs_df(excel_path: str | Path) -> pd.DataFrame:
    """Load linac facilities into a DataFrame with columns: name, lat, lon, n_linacs.

    Suitable for display and editing in a UI table.
    """
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"LINAC file not found: {path}")

    df = pd.read_excel(path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    coord_col = cols.get("coordinates")
    lat_col = cols.get("latitude")
    lon_col = cols.get("longitude")

    weight_col = None
    for candidate in ("he photon and electron beam rt", "linacs", "count", "machines"):
        if candidate in cols:
            weight_col = cols[candidate]
            break

    name_col = None
    for candidate in ("name", "institution", "institution name", "facility", "centre", "center"):
        if candidate in cols:
            name_col = cols[candidate]
            break

    rows = []
    for _, row in df.iterrows():
        try:
            if coord_col is not None:
                val = row[coord_col]
                if not isinstance(val, str):
                    continue
                parts = val.split(",")
                lat, lon = float(parts[0].strip()), float(parts[1].strip())
            elif lat_col is not None and lon_col is not None:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            else:
                continue

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue

            if weight_col is not None and pd.notna(row[weight_col]):
                w = float(row[weight_col])
                if w == 0:
                    continue
            else:
                w = 1.0

            name = (
                str(row[name_col]).strip()
                if name_col is not None and pd.notna(row[name_col])
                else ""
            )
            rows.append({"name": name, "lat": round(lat, 5), "lon": round(lon, 5), "n_linacs": w})
        except Exception:
            continue

    if not rows:
        raise ValueError(f"No valid LINAC locations found in {path}")

    return pd.DataFrame(rows)


def load_linacs_from_dirac_db(
    country_name: str,
    merge_colocated: bool = False,
    csv_path: "str | Path | None" = None,
) -> Tuple[List[Tuple[float, float, float]], pd.DataFrame]:
    """Load LINAC photon/electron RT facilities for *country_name* from the DIRAC export.

    Parameters
    ----------
    country_name : str
        Country name as stored in the DIRAC CSV (e.g. "United Kingdom").
    csv_path : path, optional
        Facility file to read; defaults to the October 2026 export
        (``DIRAC_CSV``). Pass ``DIRAC_CSV_2025`` to reproduce the preprint.
    merge_colocated : bool
        Merge rows that share identical (5-decimal) coordinates into one
        facility with the summed machine count. DIRAC lists some operators
        more than once at the same address, and geocoding to a city centroid
        can also collapse distinct centres onto one point. Off by default so
        cached travel-time matrices (keyed on the coordinate list) still match.

    Returns
    -------
    locs : list of (lat, lon, n_linacs)
    facilities_df : DataFrame with columns name, city, lat, lon, n_linacs
    """
    path = Path(csv_path) if csv_path else DIRAC_CSV
    if not path.exists():
        raise FileNotFoundError(f"DIRAC database not found: {path}")

    df = pd.read_csv(path)
    country_df = _resolve_dirac_country(country_name, df).copy()

    if country_df.empty:
        raise ValueError(f"No LINAC data found for country: {country_name!r}")

    country_df["Latitude"] = pd.to_numeric(country_df["Latitude"], errors="coerce")
    country_df["Longitude"] = pd.to_numeric(country_df["Longitude"], errors="coerce")
    country_df = country_df[
        country_df["Latitude"].notna()
        & country_df["Longitude"].notna()
        & country_df["Latitude"].between(-90, 90)
        & country_df["Longitude"].between(-180, 180)
    ]

    # Blank machine count = facility exists but count unrecorded → assume 1.
    # An explicit 0 means no photon/electron machines → excluded below.
    country_df["n_linacs"] = (
        pd.to_numeric(country_df["He Photon And Electron Beam Rt"], errors="coerce")
        .fillna(1)
    )
    country_df = country_df[country_df["n_linacs"] > 0]

    if country_df.empty:
        raise ValueError(
            f"No photon/electron RT facilities found for {country_name!r}"
        )

    facilities_df = pd.DataFrame({
        "name": country_df["Operator Name"].fillna("").astype(str).str.strip(),
        "city": country_df["City"].fillna("").astype(str).str.strip(),
        "lat": country_df["Latitude"].round(5).values,
        "lon": country_df["Longitude"].round(5).values,
        "n_linacs": country_df["n_linacs"].values,
    }).reset_index(drop=True)

    if merge_colocated:
        facilities_df = merge_colocated_facilities(facilities_df)

    locs = [
        (row["lat"], row["lon"], row["n_linacs"])
        for _, row in facilities_df.iterrows()
    ]
    return locs, facilities_df


def merge_colocated_facilities(facilities_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows with identical (lat, lon) into one row with summed machines."""
    if facilities_df.empty:
        return facilities_df
    grp = facilities_df.groupby(["lat", "lon"], sort=False, as_index=False)
    merged = grp.agg(
        name=("name", lambda s: " / ".join(sorted({x for x in s if x}))),
        city=("city", "first"),
        n_linacs=("n_linacs", "sum"),
        n_merged=("name", "size"),
    )
    cols = ["name", "city", "lat", "lon", "n_linacs", "n_merged"]
    return merged[cols].reset_index(drop=True)


def load_linacs_for_region(
    region_name: str,
) -> Tuple[List[Tuple[float, float, float]], pd.DataFrame]:
    """Load all LINAC facilities for every country in a named region.

    Parameters
    ----------
    region_name : str
        Must be a key in ``data.regions.REGION_DISPLAY_NAMES``.

    Returns
    -------
    locs : list of (lat, lon, n_linacs)
    facilities_df : DataFrame with columns name, city, lat, lon, n_linacs
    """
    from data.regions import get_region  # avoid circular import at module level

    reg = get_region(region_name)
    all_dfs: List[pd.DataFrame] = []

    for alpha2 in reg.member_alpha2:
        country_obj = pycountry.countries.get(alpha_2=alpha2)
        if country_obj is None:
            continue
        try:
            _, df = load_linacs_from_dirac_db(country_obj.name)
            all_dfs.append(df)
        except (ValueError, FileNotFoundError):
            continue  # no DIRAC data for this country — skip

    if not all_dfs:
        raise ValueError(f"No LINAC data found for region {region_name!r}")

    combined = pd.concat(all_dfs, ignore_index=True)
    locs = [
        (row["lat"], row["lon"], row["n_linacs"])
        for _, row in combined.iterrows()
    ]
    return locs, combined
