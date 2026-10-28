"""
Self-describing sparse store of travel times at H3 resolution 6.

One file per country and mode, ``travel_time_store/{ISO3}_{mode}.npz``, holding
only the (hexagon, facility) pairs reachable within the fetch cut-off:

    h3          uint64  [n_hex]   every populated res-6 cell of the country
    population  float64 [n_hex]   Kontur population of each cell (aggregation weights)
    fac_lat     float64 [n_fac]   facility coordinates, in fetch order
    fac_lon     float64 [n_fac]
    unroutable  bool    [n_fac]   facility TravelTime could not snap to a road
    rows        int32   [nnz]     index into h3
    cols        int32   [nnz]     index into fac_*
    minutes     float32 [nnz]     travel time
    meta        str               JSON: mode, resolution, cutoff_min, departure,
                                  fetched, facilities source

Every pair not listed is unreachable within ``cutoff_min``. Because the cell
ids and facility coordinates travel with the data, a matrix can be rebuilt for
any hexagon list at resolutions 3-6 and any facility list without relying on
the row or column order used at fetch time:

    store = load_store("PRT")
    M = store.matrix(hex_ids, locs)          # res taken from hex_ids
    M, found = store.matrix(hex_ids, locs, return_found=True)

Coarser resolutions use the population-weighted mean of the res-6 children
(weight max(population, 1)); a child that cannot reach a facility counts at the
cut-off, and the parent stays unreachable only if no child reaches it. This is
the rule of ``data.travel_time.aggregate_tt_matrix``.

Facilities are matched on coordinates rounded to 1e-6 degrees. A requested
facility absent from the store gets an all-unreachable column and
``found[j] = False``, so callers can refetch or fall back.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import h3
import numpy as np

STORE_DIR = Path(__file__).resolve().parent.parent / "travel_time_store"
STORE_RES = 6
# Published copy of the store, fetched on first use when a country's file is not on disk
RELEASE_URL = ("https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access/"
               "releases/download/dashboard-store-v1/{name}")
_UNAVAILABLE: set = set()  # files the release does not have (e.g. countries TravelTime does not serve)


def _download(p: Path) -> bool:
    """Fetch *p* from the release; False when unavailable (offline, or no such file)."""
    import shutil
    import tempfile
    import urllib.request
    if p.name in _UNAVAILABLE:
        return False
    try:
        with urllib.request.urlopen(RELEASE_URL.format(name=p.name), timeout=60) as r, \
                tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
            shutil.copyfileobj(r, tmp)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp.name, p)
        return True
    except Exception:  # noqa: BLE001
        _UNAVAILABLE.add(p.name)
        return False


def store_path(iso3: str, mode: str = "driving") -> Path:
    return STORE_DIR / f"{iso3}_{mode}.npz"


def _key(lat: float, lon: float) -> tuple:
    return (round(float(lat), 6), round(float(lon), 6))


@dataclass(frozen=True)
class TTStore:
    h3: np.ndarray
    population: np.ndarray
    fac_lat: np.ndarray
    fac_lon: np.ndarray
    unroutable: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    minutes: np.ndarray
    meta: dict

    @property
    def cutoff_min(self) -> float:
        return float(self.meta["cutoff_min"])

    def _columns(self, locs) -> tuple[np.ndarray, np.ndarray]:
        """Store column for each requested facility (-1 if absent).

        Facilities sharing coordinates were fetched as separate columns and can
        differ slightly (e.g. one unroutable), so the k-th requested copy of a
        coordinate takes the k-th stored copy (the last one if there are fewer).
        """
        idx: dict = {}
        for j, (la, lo) in enumerate(zip(self.fac_lat, self.fac_lon)):
            idx.setdefault(_key(la, lo), []).append(j)
        seen: dict = {}
        col = np.full(len(locs), -1, dtype=np.int64)
        for j, l in enumerate(locs):
            k = _key(l[0], l[1])
            if k in idx:
                n = seen.get(k, 0)
                col[j] = idx[k][min(n, len(idx[k]) - 1)]
                seen[k] = n + 1
        return col, col >= 0

    def matrix(self, hex_ids: Sequence[str], locs, return_found: bool = False):
        """Dense (len(hex_ids), len(locs)) float32 matrix in minutes, inf = unreachable.

        *hex_ids* are H3 strings, all at one resolution between 3 and 6.
        *locs* are (lat, lon[, ...]) tuples in the caller's order.
        """
        hex_ids = list(hex_ids)
        n_t = len(hex_ids)
        out = np.full((n_t, len(locs)), np.inf, dtype=np.float32)
        col, found = self._columns(locs)
        if n_t == 0 or not found.any():
            return (out, found) if return_found else out
        res = h3.get_resolution(hex_ids[0])
        if res > STORE_RES:
            raise ValueError(f"store is at resolution {STORE_RES}; cannot refine to {res}")

        # build the matrix over the store columns in use, then lay it out in caller order
        used = np.unique(col[found])
        pos = np.full(len(self.fac_lat), -1, dtype=np.int64)
        pos[used] = np.arange(len(used))
        n_f = len(used)
        sub = np.full((n_t, n_f), np.inf, dtype=np.float32)
        keep = pos[self.cols] >= 0
        rows, c, mins = self.rows[keep], pos[self.cols[keep]], self.minutes[keep]

        t_idx = {h: i for i, h in enumerate(hex_ids)}
        cells = [h3.int_to_str(int(x)) for x in self.h3]
        if res == STORE_RES:
            parent = np.array([t_idx.get(x, -1) for x in cells], dtype=np.int64)
            p = parent[rows]
            ok = p >= 0
            sub[p[ok], c[ok]] = mins[ok]
        else:
            parent = np.array([t_idx.get(h3.cell_to_parent(x, res), -1) for x in cells], dtype=np.int64)
            w = np.maximum(self.population.astype(np.float64), 1.0)
            w_all = np.bincount(parent[parent >= 0], weights=w[parent >= 0], minlength=n_t)
            p = parent[rows]
            ok = p >= 0
            p, c, t, wr = p[ok], c[ok], mins[ok].astype(np.float64), w[rows[ok]]
            key = p * n_f + c
            uk, inv = np.unique(key, return_inverse=True)
            s_wt = np.bincount(inv, weights=wr * t)
            s_w = np.bincount(inv, weights=wr)
            pi, ci = uk // n_f, uk % n_f
            sub[pi, ci] = ((s_wt + (w_all[pi] - s_w) * self.cutoff_min) / w_all[pi]).astype(np.float32)

        out[:, found] = sub[:, pos[col[found]]]
        return (out, found) if return_found else out


@lru_cache(maxsize=8)
def load_store(iso3: str, mode: str = "driving") -> Optional[TTStore]:
    """Load a country's store, or None if it has none (e.g. not served by TravelTime)."""
    p = store_path(iso3, mode)
    if not p.exists() and not _download(p):
        return None
    z = np.load(p, allow_pickle=False)
    return TTStore(
        h3=z["h3"], population=z["population"], fac_lat=z["fac_lat"], fac_lon=z["fac_lon"],
        unroutable=z["unroutable"], rows=z["rows"], cols=z["cols"], minutes=z["minutes"],
        meta=json.loads(str(z["meta"])),
    )


def save_store(iso3: str, mode: str, hex_ids: Sequence[str], population, locs,
               rows, cols, minutes, unroutable=(), **meta) -> Path:
    """Write a store file. *hex_ids*/*population* are every populated res-6 cell;
    *locs* the facilities in the column order of *cols*; *meta* must include
    ``cutoff_min``."""
    if "cutoff_min" not in meta:
        raise ValueError("meta must include cutoff_min")
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    bad = np.zeros(len(locs), dtype=bool)
    bad[list(unroutable)] = True
    order = np.lexsort((cols, rows))
    p = store_path(iso3, mode)
    np.savez_compressed(
        p,
        h3=np.array([h3.str_to_int(h) for h in hex_ids], dtype=np.uint64),
        population=np.asarray(population, dtype=np.float64),
        fac_lat=np.array([l[0] for l in locs], dtype=np.float64),
        fac_lon=np.array([l[1] for l in locs], dtype=np.float64),
        unroutable=bad,
        rows=np.asarray(rows, dtype=np.int32)[order],
        cols=np.asarray(cols, dtype=np.int32)[order],
        minutes=np.asarray(minutes, dtype=np.float32)[order],
        meta=np.array(json.dumps({"mode": mode, "resolution": STORE_RES, **meta})),
    )
    load_store.cache_clear()
    return p


def store_matrix(iso3: str, hex_ids: Sequence[str], locs, mode: str = "driving") -> Optional[np.ndarray]:
    """Matrix for *hex_ids* × *locs* from the country's store, or None when the
    country has no store, any requested facility is missing from it (the
    facility list has changed since the fetch), or nothing is reachable."""
    store = load_store(iso3, mode)
    if store is None:
        return None
    m, found = store.matrix(hex_ids, locs, return_found=True)
    if not found.all() or not np.isfinite(m).any():
        return None
    return m
