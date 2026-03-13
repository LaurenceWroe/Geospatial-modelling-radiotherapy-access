"""
Convert b_cancer_incidence/all_countries_cancer_statistics.csv
→ b_cancer_incidence/globocan_xarray.nc

The output xarray has dims (Cancer, Metric, ISO3) matching the format
already consumed by data/cancer.py.  Run once to regenerate the .nc file
whenever the CSV is updated.

Usage:
    python b_cancer_incidence/build_globocan_nc.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
CSV_PATH  = HERE / "all_countries_cancer_statistics.csv"
OUT_PATH  = HERE / "globocan_xarray.nc"

# ---------------------------------------------------------------------------
# Regional / aggregate rows to skip entirely (not real countries)
# ---------------------------------------------------------------------------
SKIP_NAMES = {
    "africa", "asia", "caribbean", "caribbean hub", "central america",
    "eastern africa", "eastern asia", "eastern europe", "europe",
    "european union 27", "high hdi", "high income",
    "latin america and the caribbean", "latin america hub",
    "low hdi", "low income", "lower middle income",
    "medium hdi", "medium hdi but india", "melanesia",
    "micronesia", "micronesiapolynesia", "middle africa",
    "northern africa", "northern africa central and western asia hub",
    "northern america", "northern europe", "oceania",
    "pacific islands hub", "polynesia",
    "south america", "south central asia",
    "south east and south eastern asia hub", "south eastern asia",
    "southern africa", "southern europe",
    "sub saharan africa", "sub saharan africa hub",
    "upper middle income", "very high hdi",
    "western africa", "western asia", "western europe",
    "who africa afro", "who americas paho", "who east mediterranean emro",
    "who europe euro", "who south east asia searo", "who western pacific wpro",
    "world",
    # Artefact / overseas territories (no standalone H3 population file)
    "australianew zealand",
    "france guadeloupe", "france la reunion", "france martinique",
    "french guyana",   # French Guiana — tiny, no separate DIRAC data
    "new caledonia",   # French territory
    "guam",            # US territory
    "puerto rico",     # US territory
    "french polynesia",
}

# ---------------------------------------------------------------------------
# Manual name → ISO3 overrides (names that pycountry cannot resolve)
# ---------------------------------------------------------------------------
MANUAL: dict[str, str] = {
    "france metropolitan":              "FRA",
    "bolivia plurinational state of":   "BOL",
    "bosnia herzegovina":               "BIH",
    "brunei darussalam":                "BRN",
    "cape verde":                       "CPV",
    "congo democratic republic of":     "COD",
    "congo republic of":                "COG",
    "cote divoire":                     "CIV",
    "czechia":                          "CZE",
    "eswatini":                         "SWZ",
    "gaza strip and west bank":         "PSE",
    "iran islamic republic of":         "IRN",
    "korea republic of":                "KOR",
    "lao peoples democratic republic":  "LAO",
    "republic of moldova":              "MDA",
    "russian federation":               "RUS",
    "sao tome and principe":            "STP",
    "syrian arab republic":             "SYR",
    "tanzania united republic of":      "TZA",
    "the netherlands":                  "NLD",
    "the republic of the gambia":       "GMB",
    "timor leste":                      "TLS",
    "trinidad and tobago":              "TTO",
    "turkiye":                          "TUR",
    "viet nam":                         "VNM",
    "united states of america":         "USA",
    "united kingdom":                   "GBR",
    "guinea bissau":                    "GNB",
    "north macedonia":                  "MKD",
    "venezuela":                        "VEN",
    # Fuzzy search returns Nigeria first for "Niger"
    "niger":                            "NER",
}


def _parse_number(val) -> float:
    """Parse GLOBOCAN number strings like '24 275' or '1,234' → float."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace(" ", "").replace(",", "").strip()
    if not s or s == "-":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _country_to_iso3(name: str) -> str | None:
    """Return ISO3 for a country name, or None if unresolvable."""
    key = name.strip().lower()
    if key in SKIP_NAMES:
        return None
    if key in MANUAL:
        return MANUAL[key]
    # Try pycountry fuzzy search
    try:
        import pycountry
        results = pycountry.countries.search_fuzzy(name)
        if results:
            return results[0].alpha_3
    except (LookupError, AttributeError):
        pass
    return None


def build_nc() -> None:
    print(f"Reading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH)
    df["New_Cases_Number"] = df["New_Cases_Number"].apply(_parse_number)

    # Only keep New_Cases_Number for now (matches existing .nc Metric dim)
    metric = "New_Cases_Number"

    # Resolve ISO3 for every unique country name
    country_names = df["Country_Name"].unique()
    name_to_iso3: dict[str, str] = {}
    failed: list[str] = []

    for name in sorted(country_names):
        iso3 = _country_to_iso3(name)
        if iso3 is None:
            key = name.strip().lower()
            if key not in SKIP_NAMES:
                failed.append(name)
        else:
            name_to_iso3[name] = iso3

    if failed:
        print("\nCould not resolve ISO3 for (will be excluded):")
        for f in failed:
            print(f"  {f!r}")
    else:
        print("All country names resolved successfully.")

    # Filter to resolved countries only
    df = df[df["Country_Name"].isin(name_to_iso3)].copy()
    df["ISO3"] = df["Country_Name"].map(name_to_iso3)

    # Handle duplicate ISO3 (e.g. France Metropolitan + France if both present)
    # Keep first occurrence per (ISO3, Cancer)
    df = df.drop_duplicates(subset=["ISO3", "Cancer"], keep="first")

    all_cancers = sorted(df["Cancer"].unique())
    all_iso3    = sorted(df["ISO3"].unique())

    print(f"\nBuilding array: {len(all_cancers)} cancer types × 1 metric × {len(all_iso3)} countries")

    # Build dense array (Cancer × Metric × ISO3)
    data = np.full((len(all_cancers), 1, len(all_iso3)), np.nan, dtype=np.float32)

    cancer_idx = {c: i for i, c in enumerate(all_cancers)}
    iso3_idx   = {c: i for i, c in enumerate(all_iso3)}

    for _, row in df.iterrows():
        ci = cancer_idx[row["Cancer"]]
        ii = iso3_idx[row["ISO3"]]
        data[ci, 0, ii] = row["New_Cases_Number"]

    da = xr.DataArray(
        data,
        dims=["Cancer", "Metric", "ISO3"],
        coords={
            "Cancer": all_cancers,
            "Metric": [metric],
            "ISO3":   all_iso3,
        },
        name="cancer_incidence",
    )

    print(f"Saving → {OUT_PATH} …")
    da.to_netcdf(OUT_PATH)
    print("Done.")

    # Report coverage vs old file if it exists
    try:
        old = xr.open_dataarray(OUT_PATH)
        print(f"New .nc: {len(all_iso3)} countries, {len(all_cancers)} cancer types")
    except Exception:
        pass


if __name__ == "__main__":
    build_nc()
