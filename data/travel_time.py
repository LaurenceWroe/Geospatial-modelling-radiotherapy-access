"""
TravelTime API integration for computing travel times to LINAC facilities.

Supports driving and public_transport modes via the TravelTime REST API.
Results are cached to disk as .npz files keyed by location hash + mode.

API docs: https://docs.traveltime.com/api/reference/time-filter
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / "travel_time_cache"
TT_BASE_URL = "https://api.traveltimeapp.com/v4"

# TravelTime API limits
_MAX_SEARCHES_PER_REQUEST = 10   # arrival searches per request
_MAX_LOCATIONS_PER_REQUEST = 2000  # total locations (hexes + linacs) per request
_MAX_TRAVEL_TIME_SEC = 14400     # 4 hours — TravelTime upper limit


def _cache_path(cache_key: str, mode: str) -> Path:
    return CACHE_DIR / f"{cache_key}_{mode}.npz"


def _make_cache_key(
    hex_latlons: List[Tuple[float, float]],
    linac_latlons: List[Tuple[float, float]],
) -> str:
    """Short cache key from hex + linac locations."""
    payload = json.dumps(
        {"h": hex_latlons[:10], "l": linac_latlons, "n": len(hex_latlons)},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _next_wednesday_8am_utc() -> str:
    """ISO8601 string for the next Wednesday at 08:00 UTC (typical weekday morning)."""
    now = datetime.now(timezone.utc)
    days_ahead = (2 - now.weekday()) % 7 or 7  # 2 = Wednesday
    target = (now + timedelta(days=days_ahead)).replace(
        hour=8, minute=0, second=0, microsecond=0
    )
    return target.strftime("%Y-%m-%dT%H:%M:%SZ")


def _transportation(mode: str) -> dict:
    if mode == "driving":
        return {"type": "driving"}
    elif mode == "public_transport":
        return {"type": "public_transport"}
    raise ValueError(f"Unsupported mode: {mode!r}")


def _call_time_filter(
    hex_latlons: List[Tuple[float, float]],
    hex_ids: List[str],
    linac_latlons: List[Tuple[float, float]],
    linac_ids: List[str],
    mode: str,
    app_id: str,
    api_key: str,
    arrival_time: str,
) -> dict[str, dict[str, float]]:
    """
    Single TravelTime API request.

    Returns
    -------
    {linac_id: {hex_id: travel_time_minutes}} for reachable pairs only.
    """
    locations = (
        [{"id": h_id, "coords": {"lat": lat, "lng": lon}}
         for h_id, (lat, lon) in zip(hex_ids, hex_latlons)]
        + [{"id": l_id, "coords": {"lat": lat, "lng": lon}}
           for l_id, (lat, lon) in zip(linac_ids, linac_latlons)]
    )

    arrival_searches = [
        {
            "id": l_id,
            "departure_location_ids": hex_ids,
            "arrival_location_id": l_id,
            "arrival_time": arrival_time,
            "travel_time": _MAX_TRAVEL_TIME_SEC,
            "transportation": _transportation(mode),
            "properties": ["travel_time"],
        }
        for l_id in linac_ids
    ]

    resp = requests.post(
        f"{TT_BASE_URL}/time-filter",
        headers={
            "X-Application-Id": app_id,
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"locations": locations, "arrival_searches": arrival_searches},
        timeout=60,
    )
    resp.raise_for_status()

    out: dict[str, dict[str, float]] = {}
    for r in resp.json().get("results", []):
        sid = r["search_id"]
        out[sid] = {}
        for loc in r.get("locations", []):
            props = loc.get("properties", [{}])
            tt = props[0].get("travel_time") if props else None
            if tt is not None:
                out[sid][loc["id"]] = tt / 60.0  # seconds → minutes
    return out


def compute_travel_time_matrix(
    hex_latlons: List[Tuple[float, float]],
    linac_latlons: List[Tuple[float, float]],
    mode: str,
    app_id: str,
    api_key: str,
    cache_key: str = "",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute travel time (minutes) from every hex centroid to every LINAC.

    Parameters
    ----------
    hex_latlons : list of (lat, lon) for each H3 hex centroid
    linac_latlons : list of (lat, lon) for each LINAC facility
    mode : "driving" or "public_transport"
    app_id, api_key : TravelTime credentials
    cache_key : optional override for the cache file name
    progress_callback : called as (requests_done, total_requests)

    Returns
    -------
    matrix : np.ndarray, shape (n_hexes, n_linacs), float32, minutes.
        np.inf where unreachable within the 4-hour limit.
    errors : list of str
        One entry per failed API batch, with hex/linac range and error message.
        Empty if all batches succeeded.
    """
    n_hexes = len(hex_latlons)
    n_linacs = len(linac_latlons)

    if not cache_key:
        cache_key = _make_cache_key(hex_latlons, linac_latlons)

    cache_file = _cache_path(cache_key, mode)
    if cache_file.exists():
        return np.load(cache_file)["matrix"], []

    arrival_time = _next_wednesday_8am_utc()
    matrix = np.full((n_hexes, n_linacs), np.inf, dtype=np.float32)
    failed_batches: List[tuple] = []

    hex_ids = [f"h{i}" for i in range(n_hexes)]
    linac_ids = [f"l{j}" for j in range(n_linacs)]

    # Batch linacs (max 10 per request) and hexes (fill remaining location slots)
    total_requests = sum(
        len(range(0, n_hexes, _MAX_LOCATIONS_PER_REQUEST - min(linac_batch, n_linacs - l_start)))
        for l_start in range(0, n_linacs, _MAX_SEARCHES_PER_REQUEST)
        for linac_batch in [min(_MAX_SEARCHES_PER_REQUEST, n_linacs - l_start)]
    )
    done = 0

    for l_start in range(0, n_linacs, _MAX_SEARCHES_PER_REQUEST):
        l_end = min(l_start + _MAX_SEARCHES_PER_REQUEST, n_linacs)
        l_batch_ids = linac_ids[l_start:l_end]
        l_batch_latlons = linac_latlons[l_start:l_end]
        hex_batch_size = _MAX_LOCATIONS_PER_REQUEST - len(l_batch_ids)

        for h_start in range(0, n_hexes, hex_batch_size):
            h_end = min(h_start + hex_batch_size, n_hexes)
            h_batch_ids = hex_ids[h_start:h_end]
            h_batch_latlons = hex_latlons[h_start:h_end]

            try:
                results = _call_time_filter(
                    h_batch_latlons, h_batch_ids,
                    l_batch_latlons, l_batch_ids,
                    mode, app_id, api_key, arrival_time,
                )
                for l_local_id, hex_times in results.items():
                    j = int(l_local_id[1:])
                    for h_local_id, tt_min in hex_times.items():
                        i = int(h_local_id[1:])
                        matrix[i, j] = min(matrix[i, j], tt_min)
            except Exception as e:
                failed_batches.append((h_start, h_end, l_start, l_end, str(e)))

            done += 1
            if progress_callback:
                progress_callback(done, total_requests)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, matrix=matrix)
    errors = [
        f"hexes {h0}–{h1}, linacs {l0}–{l1}: {msg}"
        for h0, h1, l0, l1, msg in failed_batches
    ]
    return matrix, errors


def clear_cache(cache_key: str, mode: str) -> None:
    """Delete a cached travel time file."""
    p = _cache_path(cache_key, mode)
    if p.exists():
        p.unlink()
