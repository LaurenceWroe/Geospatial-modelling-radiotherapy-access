"""
Region definitions for multi-country population and GLOBOCAN aggregation.

A "selection" in the UI is either:
  - a plain country name (str, pycountry-resolvable), or
  - a region display name (a key in REGION_DISPLAY_NAMES).

ISO3 codes here are synthetic regional codes stored in globocan_xarray.nc.
They must not clash with real ISO-3166-1 alpha-3 codes.
"""

from __future__ import annotations
from typing import Dict, List, NamedTuple


class RegionDef(NamedTuple):
    display_name: str        # shown in UI, e.g. "Africa"
    globocan_code: str       # synthetic ISO3 stored in .nc, e.g. "AFR"
    csv_name: str            # Country_Name value in the GLOBOCAN CSV
    member_alpha2: List[str] # ISO-2 codes of Kontur files to merge
    max_resolution: int      # cap for H3 resolution slider


# ---------------------------------------------------------------------------
# ISO-2 member lists — aligned with GLOBOCAN regional definitions
# ---------------------------------------------------------------------------

_AFRICA = [
    "DZ", "AO", "BJ", "BW", "BF", "BI", "CM", "CV", "CF", "TD",
    "KM", "CG", "CD", "CI", "DJ", "EG", "ER", "SZ", "ET", "GA",
    "GM", "GH", "GN", "GW", "KE", "LS", "LR", "LY", "MG", "MW",
    "ML", "MR", "MU", "MA", "MZ", "NA", "NE", "NG", "RW", "RE",
    "ST", "SN", "SL", "SO", "ZA", "SS", "SD", "TZ", "TG", "TN",
    "UG", "ZM", "ZW",
]

# Russia included here — GLOBOCAN classifies RUS in Europe
_EUROPE = [
    "AL", "AD", "AT", "BY", "BE", "BA", "BG", "HR", "CY", "CZ",
    "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT",
    "LV", "LI", "LT", "LU", "MT", "MD", "MC", "ME", "NL", "MK",
    "NO", "PL", "PT", "RO", "RU", "SM", "RS", "SK", "SI", "ES",
    "SE", "CH", "UA", "GB", "VA",
]

_NORTHERN_AMERICA = ["CA", "PR", "US"]

_SOUTH_AMERICA = [
    "AR", "BO", "BR", "CL", "CO", "EC", "GY", "PY", "PE", "SR",
    "UY", "VE",
]

_CENTRAL_AMERICA = ["BZ", "CR", "SV", "GT", "HN", "MX", "NI", "PA"]

_CARIBBEAN = [
    "AG", "AW", "BM", "BS", "BB", "CU", "CW", "DM", "DO", "GD",
    "GP", "HT", "JM", "KN", "LC", "MQ", "VC", "TT", "TC", "VG",
]

# GLOBOCAN "Eastern Asia"
_EAST_ASIA = ["CN", "HK", "JP", "KP", "KR", "MN", "MO", "TW"]

# GLOBOCAN "South Central Asia" — Southern Asia + Central Asia
# Iran (IR) is in this grouping per GLOBOCAN (not Western Asia)
_SOUTH_CENTRAL_ASIA = [
    "AF", "BD", "BT", "IN", "IR", "KZ", "KG", "MV", "NP", "PK",
    "LK", "TJ", "TM", "UZ",
]

# GLOBOCAN "South Eastern Asia"
_SOUTHEAST_ASIA = ["BN", "KH", "TL", "ID", "LA", "MY", "MM", "PH", "SG", "TH", "VN"]

# GLOBOCAN "Western Asia" — Turkey (TR) and Cyprus (CY) included per GLOBOCAN
_WEST_ASIA = [
    "AM", "AZ", "BH", "GE", "IQ", "IL", "JO", "KW", "LB",
    "OM", "PS", "QA", "SA", "SY", "TR", "AE", "YE",
]

_OCEANIA = ["AU", "FJ", "KI", "MH", "FM", "NR", "NZ", "PW", "PG", "WS", "SB", "TO", "TV", "VU"]

_WORLD = sorted(set(
    _AFRICA + _EUROPE + _NORTHERN_AMERICA + _SOUTH_AMERICA +
    _CENTRAL_AMERICA + _CARIBBEAN + _EAST_ASIA + _SOUTH_CENTRAL_ASIA +
    _SOUTHEAST_ASIA + _WEST_ASIA + _OCEANIA
))


# ---------------------------------------------------------------------------
# Ordered region list
# ---------------------------------------------------------------------------

REGIONS: List[RegionDef] = [
    RegionDef("World",               "WLD", "World",              _WORLD,               3),
    RegionDef("Africa",              "AFR", "Africa",             _AFRICA,              3),
    RegionDef("Europe",              "EUR", "Europe",             _EUROPE,              3),
    RegionDef("Northern America",    "NAM", "Northern America",   _NORTHERN_AMERICA,    3),
    RegionDef("South America",       "SAM", "South America",      _SOUTH_AMERICA,       3),
    RegionDef("Central America",     "CAM", "Central America",    _CENTRAL_AMERICA,     3),
    RegionDef("Caribbean",           "CAR", "Caribbean",          _CARIBBEAN,           3),
    RegionDef("East Asia",           "EAS", "Eastern Asia",       _EAST_ASIA,           3),
    RegionDef("South & Central Asia","SCA", "South Central Asia", _SOUTH_CENTRAL_ASIA,  3),
    RegionDef("Southeast Asia",      "SEA", "South Eastern Asia", _SOUTHEAST_ASIA,      3),
    RegionDef("West Asia",           "WAS", "Western Asia",       _WEST_ASIA,           3),
    RegionDef("Oceania",             "OCE", "Oceania",            _OCEANIA,             3),
]

REGION_CODES: Dict[str, RegionDef] = {r.globocan_code: r for r in REGIONS}
REGION_DISPLAY_NAMES: Dict[str, RegionDef] = {r.display_name: r for r in REGIONS}
REGION_GLOBOCAN_CODES: set = {r.globocan_code for r in REGIONS}


def is_region(name: str) -> bool:
    return name in REGION_DISPLAY_NAMES


def get_region(name: str) -> RegionDef:
    return REGION_DISPLAY_NAMES[name]
