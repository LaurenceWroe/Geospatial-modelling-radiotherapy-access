"""
Travel-time matrix loader for the global access table (standalone).

For each country it fetches (or loads from disk cache) a driving-time matrix at
the given H3 resolution using the TravelTime H3 API, aligned to the country's
hex order. One isochrone search per facility; results cache to travel_time_cache/
so the run is resumable.

Credentials are read from .streamlit/secrets.toml ([traveltime] app_id/api_key).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "travel_time_cache"


def _creds():
    import tomllib
    with open(_ROOT / ".streamlit" / "secrets.toml", "rb") as fh:
        s = tomllib.load(fh)
    tt = s.get("traveltime", {})
    return tt.get("app_id", ""), tt.get("api_key", "")


def make_tt_loader(res: int, mode: str = "driving", max_minutes: int = 120):
    from data.travel_time import compute_travel_time_matrix
    app_id, api_key = _creds()
    max_sec = max_minutes * 60

    def _loader(name: str, iso3: str, gdf):
        hex_ids = list(gdf["h3"])
        try:
            locs, _ = None, None
            from data.linacs import load_linacs_from_dirac_db
            locs, _ = load_linacs_from_dirac_db(name)
        except Exception:
            return None
        if not locs:
            return None
        linac_ll = [(lat, lon) for lat, lon, _ in locs]
        lh = hashlib.md5(json.dumps(sorted(linac_ll)).encode()).hexdigest()[:8]
        cache_key = f"{iso3}_res{res}_{max_minutes//60}h_{lh}"
        matrix, errors = compute_travel_time_matrix(
            hex_ids, linac_ll, res, mode, app_id, api_key,
            cache_key=cache_key, max_travel_time_sec=max_sec,
        )
        if errors:
            print(f"    {iso3}: {len(errors)} TT batch error(s)", flush=True)
        return matrix

    return _loader
