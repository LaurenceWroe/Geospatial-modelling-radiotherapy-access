from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple, List

import pandas as pd
import re


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    return df


def _canonicalize(col: str) -> str:
    """Lowercase, remove non-alphanumeric (to spaces), collapse spaces."""
    col = col.lower()
    col = re.sub(r"[^a-z0-9]+", " ", col)
    col = re.sub(r"\s+", " ", col).strip()
    return col


def _find_best_columns(cols: List[str]) -> Dict[str, Optional[str]]:
    """Pick best matching columns for coordinates, count, and name.

    Returns mapping of canonical keys -> source column name or None.
    Canonical keys: 'coordinates', 'count', 'name'
    """
    normalized_to_original: Dict[str, str] = {}
    for c in cols:
        normalized_to_original[_canonicalize(c)] = c

    def has(*candidates: str) -> Optional[str]:
        for cand in candidates:
            if cand in normalized_to_original:
                return normalized_to_original[cand]
        return None

    # Candidates for name
    name_col = has(
        "operator name",
        "operator",
        "facility name",
        "centre name",
        "center name",
        "institution",
        "hospital name",
        "site name",
        "name",
    )

    # Candidates for a single coordinates column
    coord_single = has(
        "coordinates",
        "coord",
        "gps coordinates",
        "geo coordinates",
        "lat lon",
        "latitude longitude",
    )

    # Candidates for separate latitude/longitude columns
    lat_col = has("latitude", "lat")
    # Accept multiple spellings of longitude
    lon_col = has("longitude", "long", "lon")

    # Candidates for LINAC count
    count_col = has(
        "he photon and electron beam rt",
        "he photon electron beam rt",
        "he photon and electron beam",
        "linear accelerators",
        "number of linear accelerators",
        "no of linear accelerators",
        "linacs",
        "linac count",
        "number of linacs",
        "no linacs",
    )

    return {
        "name": name_col,
        "coordinates": coord_single if coord_single else None,
        "lat": lat_col,
        "lon": lon_col,
        "count": count_col,
    }


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns: 'Operator Name', 'Coordinates', 'He Photon And Electron Beam Rt'.

    - Accepts alternate column headers for name and count.
    - Builds 'Coordinates' from either a single column or separate lat/lon columns.
    """
    df = _normalize_columns(df)
    selected = _find_best_columns(list(df.columns))

    # Prepare output columns
    out_name_col = "Operator Name"
    out_coords_col = "Coordinates"
    out_count_col = "He Photon And Electron Beam Rt"

    # Start with a copy to avoid SettingWithCopy warnings
    working = df.copy()

    # Build Coordinates
    if selected["coordinates"]:
        coords_series = working[selected["coordinates"]]
        # Normalize separators to comma
        coords_series = coords_series.astype(str).str.replace(";", ",", regex=False)
        # Ensure there's a comma separator between two numbers if space-separated
        coords_series = coords_series.str.replace(r"\s+", ", ", regex=True)
        working[out_coords_col] = coords_series
    elif selected["lat"] and selected["lon"] and selected["lat"] in working.columns and selected["lon"] in working.columns:
        def _fmt(lat, lon):
            try:
                lat_f = float(lat)
                lon_f = float(lon)
                return f"{lat_f}, {lon_f}"
            except Exception:
                return None
        working[out_coords_col] = [
            _fmt(la, lo) for la, lo in zip(working[selected["lat"]], working[selected["lon"]])
        ]
    else:
        # If we cannot build coordinates, leave as-is if an exact 'Coordinates' exists
        if out_coords_col not in working.columns:
            working[out_coords_col] = None

    # Pick name
    if selected["name"] and selected["name"] in working.columns:
        working[out_name_col] = working[selected["name"]]
    else:
        # Fallback to any column that looks like a name
        if out_name_col not in working.columns:
            name_like = None
            for c in working.columns:
                cn = _canonicalize(c)
                if cn in {"operator name", "operator", "facility name", "name"}:
                    name_like = c
                    break
            working[out_name_col] = working[name_like] if name_like else "Unknown"

    # Pick count column
    if selected["count"] and selected["count"] in working.columns:
        working[out_count_col] = working[selected["count"]]
    else:
        # Try to infer from any column mentioning linear accelerators or linacs
        found_count = None
        for c in working.columns:
            cn = _canonicalize(c)
            if ("linear" in cn and "accelerator" in cn) or ("linac" in cn):
                found_count = c
                break
        if found_count is not None:
            working[out_count_col] = working[found_count]
        else:
            working[out_count_col] = 0

    # Ensure types are correct
    # Count should be numeric
    working[out_count_col] = pd.to_numeric(working[out_count_col], errors="coerce").fillna(0).astype(int)

    # Drop rows missing coordinates entirely
    working = working[working[out_coords_col].notna() & (working[out_coords_col] != "None")]

    # Return only required columns + keep original for downstream if needed
    return working[[out_name_col, out_coords_col, out_count_col]]


def read_linac_excel(filepath: str | Path, sheet: Optional[str | int] = None) -> pd.DataFrame:
    """Read LINAC Excel robustly, picking a sheet with expected columns.

    Tries in order:
    - Explicit sheet if provided
    - First sheet (0)
    - Sheet named 'Sheet'
    - Any sheet containing required columns
    """
    path = Path(filepath)
    required_cols = {"Coordinates", "He Photon And Electron Beam Rt"}

    # Attempt explicit or default sheet reads with openpyxl
    try_targets: list[str | int] = []
    if sheet is not None:
        try_targets.append(sheet)
    # Common defaults
    try_targets.extend(["Sheet", 0])

    for target in try_targets:
        try:
            df_raw = pd.read_excel(path, engine="openpyxl", sheet_name=target)
            df_std = _standardize_dataframe(df_raw)
            if required_cols.issubset(set(df_std.columns)) and not df_std.empty:
                return df_std
        except Exception:
            continue

    # Fallback: iterate sheet names via openpyxl to avoid pandas sheet_name=None bug
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet_names = [ws.title for ws in wb.worksheets]
    except Exception as e:
        raise RuntimeError(f"Failed to inspect Excel sheets at {path}: {e}")

    for name in sheet_names:
        try:
            df_raw = pd.read_excel(path, engine="openpyxl", sheet_name=name)
            df_std = _standardize_dataframe(df_raw)
            if required_cols.issubset(set(df_std.columns)) and not df_std.empty:
                return df_std
        except Exception:
            continue

    raise ValueError(
        f"No sheet containing required LINAC data found in {path}. Checked sheets: {try_targets + sheet_names}"
    )


