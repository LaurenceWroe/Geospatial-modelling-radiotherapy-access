"""
TravelTime H3 API integration for computing travel times to LINAC facilities.

Uses the /v4/h3 endpoint which returns H3 cell travel times natively — one
request per LINAC (up to 10 LINACs per request via arrival_searches), no
hex-centroid batching needed. `remove_water_bodies` is set to False so that
all H3 cells are returned regardless of water coverage.

Modes: "driving" and "public_transport" via the TravelTime REST API.
Results are cached to disk as .npz files keyed by location hash + mode.

API docs: https://docs.traveltime.com/api/reference/h3
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import h3
import numpy as np
import requests

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / "travel_time_cache"
TT_BASE_URL = "https://api.traveltimeapp.com/v4"

# TravelTime H3 API limits
_MAX_SEARCHES_PER_REQUEST = 10   # arrival_searches per request

# Maximum travel time (seconds) by H3 resolution, per TravelTime docs
# https://docs.traveltime.com/api/reference/h3#limits-of-resolution-and-traveltime
# Documented range is 6–12. Empirically, resolution 5 also works (returns results).
# Resolution 4 returns HTTP 500. Resolutions 1–3 are rejected by the API.
MAX_TRAVEL_TIME_BY_RES: dict[int, int] = {
    5: 36000,   # 10 hours (undocumented but works empirically)
    6: 36000,   # 10 hours
    7: 36000,   # 10 hours
    8: 28800,   # 8 hours
    9: 14400,   # 4 hours
    10: 5400,   # 90 minutes
    11: 2700,   # 45 minutes
    12: 1800,   # 30 minutes
}
_DEFAULT_MAX_TRAVEL_TIME_SEC = 36000  # 10 hours fallback (res 5–7)

# Resolutions supported by the TravelTime H3 endpoint (5 empirical, 6–12 documented)
TT_SUPPORTED_RESOLUTIONS = frozenset(MAX_TRAVEL_TIME_BY_RES.keys())


def _cache_path(cache_key: str, mode: str) -> Path:
    return CACHE_DIR / f"{cache_key}_{mode}.npz"


def _make_cache_key(
    hex_ids: List[str],
    linac_latlons: List[Tuple[float, float]],
) -> str:
    """Short cache key from hex IDs + linac locations."""
    payload = json.dumps(
        {"h": hex_ids[:10], "l": linac_latlons, "n": len(hex_ids)},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _next_wednesday_8am_utc() -> str:
    """ISO8601 string for the next Wednesday at 08:00 UTC.

    Wednesday morning UTC is used as a representative weekday departure.
    Note: this corresponds to different local times across the globe (e.g.
    08:00 in UK/West Africa, 09:00–10:00 in Europe, 16:00 in East Asia,
    03:00 in the Americas). Travel times in regions far from UTC±0 will
    reflect off-peak or night-time conditions rather than a typical morning
    commute. For a global planning tool this is a known limitation.
    """
    now = datetime.now(timezone.utc)
    days_ahead = (2 - now.weekday()) % 7 or 7  # 2 = Wednesday
    target = (now + timedelta(days=days_ahead)).replace(
        hour=8, minute=0, second=0, microsecond=0
    )
    return target.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _transportation(mode: str, max_travel_time_sec: int) -> dict:
    if mode == "driving":
        # "driving" uses TravelTime's typical road-speed model (no real-time traffic).
        return {"type": "driving"}
    elif mode == "public_transport":
        return {
            "type": "public_transport",
            "walking_time": max_travel_time_sec,
        }
    raise ValueError(f"Unsupported mode: {mode!r}")


def _call_h3(
    linac_latlons: List[Tuple[float, float]],
    linac_ids: List[str],
    h3_resolution: int,
    mode: str,
    app_id: str,
    api_key: str,
    departure_time: str,
    max_travel_time_sec: int,
    snap_threshold_m: int,
) -> dict[str, dict[str, float]]:
    """
    Single POST to /v4/h3.

    Returns
    -------
    {linac_id: {h3_cell_id: travel_time_minutes}} for reachable cells only.
    """
    departure_searches = [
        {
            "id": l_id,
            "coords": {"lat": lat, "lng": lon},
            "departure_time": departure_time,
            "travel_time": max_travel_time_sec,
            "transportation": _transportation(mode, max_travel_time_sec),
            "remove_water_bodies": False,
            "snapping": {
                "penalty": "enabled",
                "accept_roads": "any_drivable",
                "threshold": snap_threshold_m,
            },
        }
        for l_id, (lat, lon) in zip(linac_ids, linac_latlons)
    ]

    resp = requests.post(
        f"{TT_BASE_URL}/h3",
        headers={
            "X-Application-Id": app_id,
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "resolution": h3_resolution,
            "properties": ["min", "mean"],
            "departure_searches": departure_searches,
        },
        timeout=120,
    )
    resp.raise_for_status()

    out: dict[str, dict[str, float]] = {}
    for r in resp.json().get("results", []):
        sid = r["search_id"]
        out[sid] = {}
        for cell in r.get("cells", []):
            # SDK confirmed: field is "id", travel time nested under "properties"
            cell_id = cell.get("id")
            props = cell.get("properties") or {}
            tt_sec = props.get("mean") if props.get("mean") is not None else props.get("min")
            if cell_id is not None and tt_sec is not None:
                out[sid][cell_id] = tt_sec / 60.0  # seconds → minutes
    return out


def compute_travel_time_matrix(
    hex_ids: List[str],
    linac_latlons: List[Tuple[float, float]],
    h3_resolution: int,
    mode: str,
    app_id: str,
    api_key: str,
    cache_key: str = "",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_travel_time_sec: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute travel time (minutes) from every H3 cell to every LINAC.

    Uses the TravelTime /v4/h3 endpoint: one request per batch of up to 10
    LINACs. The API returns all reachable H3 cells natively, with water body
    filtering enabled to avoid routing failures near coastlines.

    Parameters
    ----------
    hex_ids : list of H3 cell ID strings (the ``h3`` column of your GeoDataFrame)
    linac_latlons : list of (lat, lon) for each LINAC facility
    h3_resolution : H3 resolution of the hex grid (determines travel time cap)
    mode : "driving" or "public_transport"
    app_id, api_key : TravelTime credentials
    cache_key : optional override for the cache file name
    progress_callback : called as (requests_done, total_requests)
    max_travel_time_sec : optional upper bound on travel time (seconds).
        If None, defaults to the resolution-based cap in MAX_TRAVEL_TIME_BY_RES.
        Cannot exceed that cap.

    Returns
    -------
    matrix : np.ndarray, shape (n_hexes, n_linacs), float32, minutes.
        np.inf where unreachable within the travel time cap.
    errors : list of str
        One entry per failed API batch. Empty if all batches succeeded.
    """
    n_hexes = len(hex_ids)
    n_linacs = len(linac_latlons)

    if not cache_key:
        cache_key = _make_cache_key(hex_ids, linac_latlons)

    cache_file = _cache_path(cache_key, mode)
    if cache_file.exists():
        return np.load(cache_file)["matrix"], []

    _res_cap = MAX_TRAVEL_TIME_BY_RES.get(h3_resolution, _DEFAULT_MAX_TRAVEL_TIME_SEC)
    if max_travel_time_sec is None:
        max_travel_time_sec = _res_cap
    else:
        max_travel_time_sec = min(max_travel_time_sec, _res_cap)
    # Snap threshold = 3 × H3 edge length in metres, so coastal/rural hex centroids
    # that sit away from the road network can still be snapped to a road.
    snap_threshold_m = int(h3.average_hexagon_edge_length(h3_resolution, unit="m") * 3)
    departure_time = _next_wednesday_8am_utc()
    matrix = np.full((n_hexes, n_linacs), np.inf, dtype=np.float32)
    failed_batches: List[str] = []

    # Build lookup: H3 cell ID → row index in matrix
    cell_to_idx: dict[str, int] = {cell_id: i for i, cell_id in enumerate(hex_ids)}

    linac_ids = [f"l{j}" for j in range(n_linacs)]
    total_requests = (n_linacs + _MAX_SEARCHES_PER_REQUEST - 1) // _MAX_SEARCHES_PER_REQUEST
    done = 0

    _MAX_RETRIES = 10
    _RETRY_BACKOFF = [2, 5, 10]  # seconds between attempts

    for l_start in range(0, n_linacs, _MAX_SEARCHES_PER_REQUEST):
        l_end = min(l_start + _MAX_SEARCHES_PER_REQUEST, n_linacs)
        l_batch_ids = linac_ids[l_start:l_end]
        l_batch_latlons = linac_latlons[l_start:l_end]

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                results = _call_h3(
                    l_batch_latlons, l_batch_ids,
                    h3_resolution, mode,
                    app_id, api_key,
                    departure_time, max_travel_time_sec,
                    snap_threshold_m,
                )
                for l_local_id, cell_times in results.items():
                    j = int(l_local_id[1:])
                    for cell_id, tt_min in cell_times.items():
                        i = cell_to_idx.get(cell_id)
                        if i is not None:
                            matrix[i, j] = min(matrix[i, j], tt_min)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)])

        if last_exc is not None:
            failed_batches.append(f"linacs {l_start}–{l_end - 1}: {last_exc}")

        done += 1
        if progress_callback:
            progress_callback(done, total_requests)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not failed_batches:
        np.savez_compressed(cache_file, matrix=matrix)
    return matrix, failed_batches


def aggregate_tt_matrix(
    matrix_res5: np.ndarray,
    hex_ids_res5: List[str],
    pops_res5: np.ndarray,
    hex_ids_target: List[str],
    target_resolution: int,
) -> np.ndarray:
    """Aggregate a res-5 TT matrix to a coarser resolution using population-weighted mean.

    For each target hex, finds its res-5 children present in the matrix and
    computes the population-weighted average travel time to each LINAC.
    Hexes with no reachable children remain np.inf.
    """
    n_target = len(hex_ids_target)
    n_linacs = matrix_res5.shape[1]
    target_idx = {hid: i for i, hid in enumerate(hex_ids_target)}

    weighted_sum = np.zeros((n_target, n_linacs), dtype=np.float64)
    weight_sum   = np.zeros((n_target, n_linacs), dtype=np.float64)

    for row_i, (hid, pop) in enumerate(zip(hex_ids_res5, pops_res5)):
        parent = h3.cell_to_parent(hid, target_resolution)
        t_idx = target_idx.get(parent)
        if t_idx is None:
            continue
        tts = matrix_res5[row_i].astype(np.float64)
        valid = np.isfinite(tts)
        if not valid.any():
            continue
        w = max(float(pop), 1.0)
        weighted_sum[t_idx, valid] += w * tts[valid]
        weight_sum[t_idx, valid]   += w

    result = np.full((n_target, n_linacs), np.inf, dtype=np.float32)
    has_data = weight_sum > 0
    result[has_data] = (weighted_sum[has_data] / weight_sum[has_data]).astype(np.float32)
    return result


def clear_cache(cache_key: str, mode: str) -> None:
    """Delete a cached travel time file."""
    p = _cache_path(cache_key, mode)
    if p.exists():
        p.unlink()
