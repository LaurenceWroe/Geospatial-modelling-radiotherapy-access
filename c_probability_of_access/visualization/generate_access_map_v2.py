from __future__ import annotations

import os
import math
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import re

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio import warp
from rasterio.transform import Affine
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, Normalize

from pyproj import CRS, Transformer, Geod

# Optional: use your existing helper if present; otherwise fall back to pandas parser
"""try:
    from c_probability_of_access.analysis.excel_utils import read_linac_excel  # type: ignore
    _HAS_EXCEL_HELPER = True
except Exception:
    _HAS_EXCEL_HELPER = False
    import pandas as pd"""

_HAS_EXCEL_HELPER = False
import pandas as pd


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------- small utilities ----------

def _canonical_basename(country: str, lambda_km: float, output_name: Optional[str]) -> str:
    iso3 = (country or "country").lower()
    if output_name:
        return output_name[:-4] if output_name.lower().endswith(".png") else output_name
    return f"{iso3}_{int(round(lambda_km))}km_access_probability"


def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """Approximate metres per degree (lon, lat) at a given latitude (good enough for pixel size reporting)."""
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat) - 0.0023 * math.cos(6 * lat)
    m_per_deg_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat) + 0.118 * math.cos(5 * lat)
    return m_per_deg_lon, m_per_deg_lat


def _grid_coords(transform: Affine, width: int, r0: int, r1: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return world X,Y coordinate grids (no CRS change) for rows [r0:r1], all columns.
    Works with rotated affine transforms too; uses pixel centers (+0.5).
    """
    cols = (np.arange(width, dtype=np.float64) + 0.5)[None, :]           # shape (1, W)
    rows = (np.arange(r0, r1, dtype=np.float64) + 0.5)[:, None]          # shape (Htile, 1)
    # Affine: x = a*col + b*row + c; y = d*col + e*row + f
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    X = c + a * cols + b * rows
    Y = f + d * cols + e * rows
    return X, Y


def _guess_local_projected_crs(src: rasterio.DatasetReader) -> CRS:
    """Pick a suitable local projected CRS (UTM) from raster centroid."""
    # get centroid in lon/lat
    center_x = (src.bounds.left + src.bounds.right) / 2
    center_y = (src.bounds.top + src.bounds.bottom) / 2

    if not src.crs or CRS.from_user_input(src.crs).is_geographic:
        lon, lat = center_x, center_y
    else:
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        lon, lat = to_wgs84.transform(center_x, center_y)

    # compute UTM zone
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": south, "ellps": "WGS84", "datum": "WGS84", "units": "m"})


def _load_linac_points(linac_path: str, weight_field: Optional[str] = None) -> List[Tuple[float, float, float]]:
    """
    Returns [(lat, lon, weight>=0)], weight defaults to 1.0 if missing.
    Supports either:
      - 'Coordinates' column as "lat, lon"
      - OR 'Latitude' & 'Longitude' columns (case-insensitive).
    If your excel_utils is available, it will be used first.
    """
    pts: List[Tuple[float, float, float]] = []

    """if _HAS_EXCEL_HELPER:
        df = read_linac_excel(linac_path)
    else:
        if not os.path.exists(linac_path):
            raise FileNotFoundError(f"LINAC Excel not found: {linac_path}")
        df = pd.read_excel(linac_path)"""

    if not os.path.exists(linac_path):
        raise FileNotFoundError(f"LINAC Excel not found: {linac_path}")
    df = pd.read_excel(linac_path)

    cols = {str(c).strip().lower(): c for c in df.columns}

    # Detect coordinate style
    coord_col = cols.get("coordinates")
    lat_col = cols.get("latitude")
    lon_col = cols.get("longitude")

    # Choose weight field
    weight_col = None
    if weight_field:
        weight_col = cols.get(str(weight_field).strip().lower())
    # sensible default seen in your codebase:
    if weight_col is None:
        weight_col = cols.get("he photon and electron beam rt") or cols.get("linacs") or cols.get("count")

    for _, row in df.iterrows():
        try:
            if coord_col:
                val = row[coord_col]
                if isinstance(val, str):
                    lat_s, lon_s = [x.strip() for x in val.split(",")]
                    lat, lon = float(lat_s), float(lon_s)
                else:
                    continue
            elif lat_col and lon_col:
                lat, lon = float(row[lat_col]), float(row[lon_col])
            else:
                continue

            w = float(row[weight_col]) if weight_col in row and pd.notna(row[weight_col]) else 1.0
            if w < 0:
                w = 0.0
            pts.append((lat, lon, w))
        except Exception:
            # ignore bad row
            continue

    # Drop zero-weight points
    pts = [(la, lo, w) for (la, lo, w) in pts if w > 0]

    if not pts:
        raise ValueError(f"No valid LINAC points parsed from {linac_path}")

    return pts


# ---------- core computation ----------

def compute_access(
    pop_path: str,
    linac_path: str,
    lambda_km: float,
    cutoff_km: Optional[float],
    mode: str = "nearest",           # "nearest" (fast) or "multi"
    weight_field: Optional[str] = None,
    use_weights: bool = True,        
) -> Dict:
    """
    Returns dict with:
      - prob: 2D float32 array in [0,1] (nan on nodata)
      - pop:  2D float32 array (nan on nodata)
      - pop_weighted: prob*pop (float32)
      - transform, crs, nodata_mask (bool array), pixel_size_m (float)
      - stats: {"p_mean": float, "pop_total": float, "pop_with_access": float, "n_facilities": int}
    """
    lambda_m = float(lambda_km) * 1000.0
    cutoff_km = float(cutoff_km) if cutoff_km is not None else 5.0 * float(lambda_km)

    with rasterio.open(pop_path) as src:
        pop_raw = src.read(1).astype(np.float32)
        ds_mask = (src.read_masks(1) > 0)

        # Land mask: pixels that are valid AND have population > 0
        # (This reliably strips oceans for WorldPop/GPW-style rasters.)
        land_mask = ds_mask & (pop_raw > 0)

        # Make seas NaN right away so they stay transparent everywhere downstream
        pop = np.where(land_mask, pop_raw, np.nan).astype(np.float32)

        H, W = pop.shape
        dist_km = np.full((H, W), np.nan, dtype=np.float32)  # holds distance to nearest LINAC (km)

        crs_in = src.crs
        transform_in = src.transform

        # approximate pixel size in metres for reporting
        if crs_in and CRS.from_user_input(crs_in).is_geographic:
            # estimate at raster centroid
            cx = (src.bounds.left + src.bounds.right) / 2
            cy = (src.bounds.top + src.bounds.bottom) / 2
            m_lon, m_lat = _meters_per_degree(cy)
            px = abs(transform_in.a) * m_lon
            py = abs(transform_in.e) * m_lat
            pixel_size_m = float(math.hypot(px, py))
        else:
            px = abs(transform_in.a)
            py = abs(transform_in.e)
            pixel_size_m = float(math.hypot(px, py))

        # facilities
        pts = _load_linac_points(linac_path, weight_field=weight_field)
        if not use_weights:
            pts = [(lat, lon, 1.0) for (lat, lon, _w) in pts]
        n_fac = len(pts)
        if n_fac == 0:
            prob = np.zeros_like(pop, dtype=np.float32)
            dist_km = np.full_like(pop, np.nan, dtype=np.float32)
        else:
            if mode == "nearest":
                # project to local metres grid, compute EDT distance to nearest facility
                crs_proj = _guess_local_projected_crs(src)
                # choose approx target resolution in metres close to native
                res_m = max(px, py) if (px > 0 and py > 0) else None

                # reproject population to projected grid (for shape & transform)
                dst_transform, dst_width, dst_height = warp.calculate_default_transform(
                    crs_in, crs_proj, W, H, *src.bounds, resolution=res_m
                )
                pop_proj = np.empty((dst_height, dst_width), dtype=np.float32)
                mask_proj = np.empty((dst_height, dst_width), dtype=np.uint8)

                warp.reproject(
                    source=pop, destination=pop_proj,
                    src_transform=transform_in, src_crs=crs_in,
                    dst_transform=dst_transform, dst_crs=crs_proj,
                    resampling=Resampling.average
                )
                warp.reproject(
                    source=land_mask.astype(np.uint8), destination=mask_proj,
                    src_transform=transform_in, src_crs=crs_in,
                    dst_transform=dst_transform, dst_crs=crs_proj,
                    resampling=Resampling.nearest
                )
                mask_proj = mask_proj > 0

                # facility raster (mark nearest pixel to each point)
                fac = np.zeros_like(pop_proj, dtype=np.uint8)
                to_proj = Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)
                xs, ys = to_proj.transform([p[1] for p in pts], [p[0] for p in pts])
                # map to rows/cols
                a, b, c, d, e, f = dst_transform.a, dst_transform.b, dst_transform.c, dst_transform.d, dst_transform.e, dst_transform.f
                # invert affine
                inv = ~dst_transform
                for x, y, w in zip(xs, ys, [p[2] for p in pts]):
                    col, row = inv * (x, y)
                    r = int(round(row))
                    c0 = int(round(col))
                    if 0 <= r < fac.shape[0] and 0 <= c0 < fac.shape[1]:
                        fac[r, c0] = 1  # ignore weight in nearest-mode

                # EDT distance in metres (pixel units * pixel diag)
                pxx = abs(dst_transform.a)
                pyy = abs(dst_transform.e)
                pix_m = float(math.hypot(pxx, pyy))
                edt = distance_transform_edt(fac == 0).astype(np.float32) * pix_m

                # probability on projected grid
                prob_proj = np.exp(-edt / lambda_m).astype(np.float32)

                # apply cutoff (zero beyond cutoff_km)
                if cutoff_km is not None:
                    prob_proj = np.where(edt <= (cutoff_km * 1000.0), prob_proj, 0.0).astype(np.float32)

                # mask sea/nodata
                prob_proj = np.where(mask_proj, prob_proj, np.nan).astype(np.float32)

                # NEW: distance (km) on projected grid, masked like prob
                dist_proj_km = (edt * 1e-3).astype(np.float32)
                dist_proj_km = np.where(mask_proj, dist_proj_km, np.nan).astype(np.float32)
                
                # reproject prob back to original grid to align with input raster
                prob = np.empty((H, W), dtype=np.float32)
                warp.reproject(
                    source=prob_proj, destination=prob,
                    src_transform=dst_transform, src_crs=crs_proj,
                    dst_transform=transform_in, dst_crs=crs_in,
                    resampling=Resampling.bilinear
                )
                prob[~land_mask] = np.nan
                warp.reproject(
                    source=dist_proj_km, destination=dist_km,
                    src_transform=dst_transform, src_crs=crs_proj,
                    dst_transform=transform_in, dst_crs=crs_in,
                    resampling=Resampling.bilinear
                )
                dist_km[~land_mask] = np.nan

            elif mode == "multi":
                # compute geodesic distances on original grid (tile-wise)
                geod = Geod(ellps="WGS84")
                to_wgs84 = None
                if crs_in and CRS.from_user_input(crs_in).is_geographic:
                    # identity
                    to_wgs84 = None
                else:
                    to_wgs84 = Transformer.from_crs(crs_in, "EPSG:4326", always_xy=True)

                prob = np.zeros((H, W), dtype=np.float32)
                tile = 512
                # product accumulator for (1 - p_j)
                for r0 in range(0, H, tile):
                    r1 = min(H, r0 + tile)
                    # grid in source CRS
                    XX, YY = _grid_coords(transform_in, W, r0, r1)
                    if to_wgs84 is not None:
                        lons, lats = to_wgs84.transform(XX, YY)
                    else:
                        lons, lats = XX, YY

                    prod = np.ones((r1 - r0, W), dtype=np.float64)
                    lons_f = lons.reshape(-1)
                    lats_f = lats.reshape(-1)

                    tile_dist_min = None  # NEW


                    for (lat_fac, lon_fac, w_fac) in pts:
                        # vectorised geodesic distance (m)
                        _, _, dists_m = geod.inv(
                            np.full_like(lons_f, lon_fac, dtype=np.float64),
                            np.full_like(lats_f, lat_fac, dtype=np.float64),
                            lons_f, lats_f
                        )
                        dists_km = (dists_m * 1e-3).reshape((r1 - r0, W))

                        # track min distance across facilities
                        if tile_dist_min is None:
                            tile_dist_min = dists_km.astype(np.float32)
                        else:
                            np.minimum(tile_dist_min, dists_km, out=tile_dist_min)

                        p = np.exp(-dists_km / lambda_km)  # float64 for stability
                        if cutoff_km is not None:
                            p = np.where(dists_km <= cutoff_km, p, 0.0)

                        # account for facility "weight" by repeating independent effect w times:
                        # (1 - p)^w
                        prod *= np.power(1.0 - p, w_fac, dtype=np.float64)

                    prob[r0:r1, :] = (1.0 - prod).astype(np.float32)

                    # write tile min-distance back to full grid
                    if tile_dist_min is not None:
                        dist_km[r0:r1, :] = tile_dist_min.astype(np.float32)

                prob[~land_mask] = np.nan
                dist_km[~land_mask] = np.nan

                    

            else:
                raise ValueError(f"Unknown mode '{mode}'; expected 'nearest' or 'multi'.")

        # stats
        pop_total = float(np.nansum(pop))
        pop_with_access = float(np.nansum(prob * pop)) if pop_total > 0 else 0.0
        p_mean = (pop_with_access / pop_total) if pop_total > 0 else 0.0

        out = {
            "prob": prob.astype(np.float32),
            "pop": pop.astype(np.float32),
            "pop_weighted": (prob * pop).astype(np.float32),
            "transform": transform_in,
            "crs": crs_in,
            "nodata_mask": land_mask,  # True where valid
            "pixel_size_m": pixel_size_m,
            "stats": {
                "p_mean": float(p_mean),
                "pop_total": float(pop_total),
                "pop_with_access": float(pop_with_access),
                "n_facilities": int(n_fac),
                "weights_used": bool(use_weights),

                
            },
            "distance_km": dist_km.astype(np.float32), 

        }
        return out


# ---------- writers / renderers ----------

def write_geotiff(path: str, array: np.ndarray, transform, crs, nodata: float = np.nan, compress: str = "LZW") -> None:
    """
    Write a single-band float32 GeoTIFF. NaNs are preserved.
    """
    path = str(path)
    profile = {
        "driver": "GTiff",
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "compress": compress,
        "BIGTIFF": "IF_NEEDED",
        "nodata": nodata if np.isnan(nodata) else float(nodata),
    }
    # replace inf with nan to be safe
    arr = array.astype(np.float32)
    arr[~np.isfinite(arr)] = np.nan
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def render_png(
    array: np.ndarray,
    kind: str,                 # "prob" or "pop_weighted"
    out_png: Optional[str],
    title: str,
    dpi: int = 300,
    show: bool = False
) -> Optional[bytes]:
    """
    Render array to a PNG.
      - "prob": linear 0..1 with % ticks
      - "pop_weighted": LogNorm auto (2–98th pct), label "People with access (per pixel)"
    Returns PNG bytes if out_png is None; else saves to out_png and returns None.
    """
    # prepare figure
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    arr = np.array(array, dtype=float)

    # flip for visual "north-up" if desired; we avoid flipping here to keep it literal
    # Use no extent here (quick-look). GUI will display GeoTIFF for georeferenced/interactive view.

    if kind == "prob":
        vmin, vmax = 0.0, 1.0
        im = ax.imshow(arr, origin="upper", vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
        cbar.set_label("Probability of Access")
        ax.set_xlabel("Pixel column")
        ax.set_ylabel("Pixel row")

    elif kind == "distance_km":
        data = np.array(arr, copy=True)
        data[~np.isfinite(data)] = np.nan
        vals = data[np.isfinite(data)]
        if vals.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = 0.0
            vmax = float(np.nanpercentile(vals, 98))
            if vmax <= vmin:
                vmax = vmin + 1.0
        im = ax.imshow(data, origin="upper", norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.4)
        cbar.set_label("Distance to nearest LINAC (km)")
        ax.set_xlabel("Pixel column")
        ax.set_ylabel("Pixel row")

    elif kind == "pop_weighted":
        # mask non-positive for LogNorm safety; zeros will be invisible in this quick-look
        data = np.array(arr, copy=True)
        data[~np.isfinite(data)] = np.nan
        vals = data[np.isfinite(data) & (data > 0)]
        if vals.size == 0:
            vmin = 1e-6
            vmax = 1.0
        else:
            vmin = float(np.percentile(vals, 2))
            vmax = float(np.percentile(vals, 98))
            if vmax <= vmin:
                vmax = vmin * 10.0
        im = ax.imshow(data, origin="upper", norm=LogNorm(vmin=max(vmin, 1e-12), vmax=vmax), cmap="viridis", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("People with access (per pixel)")
        ax.set_xlabel("Pixel column")
        ax.set_ylabel("Pixel row")
    else:
        plt.close(fig)
        raise ValueError("kind must be 'prob' or 'pop_weighted'")

    ax.set_title(title, fontsize=12, pad=6)

    if show:
        plt.show()

    png_bytes = None
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    else:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        png_bytes = buf.getvalue()
        buf.close()

    plt.close(fig)
    return png_bytes


# ---------- GUI-friendly wrapper ----------

def generate_accessibility_plot(
    population_raster_path: str,
    linac_excel_path: str,
    country: str = "",
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,      # basename (no extension) or ".png" name; we'll normalise
    lambda_km: float = 30.0,
    max_distance_km: Optional[float] = None,
    dpi: int = 300,
    show_plot: bool = False,
    *,
    value_to_plot: str = "pop_weighted",    # "prob" or "pop_weighted"
    mode: str = "nearest",                  # "nearest" (fast) or "multi"
    write_tif: bool = True,
    overwrite: bool = False,
    use_weights: bool = False,               #

) -> Tuple[np.ndarray, Optional[str], Optional[str], Dict]:
    """
    Returns (array_plotted, tif_path, png_path, stats).
    - tif_path is the numeric layer the GUI can load for interactive scaling
    - array_plotted is the numeric array used to render the PNG (prob or pop_weighted)
    - stats["p_mean"] is the national population-weighted mean probability
    """
    # compute
    result = compute_access(
        pop_path=population_raster_path,
        linac_path=linac_excel_path,
        lambda_km=lambda_km,
        cutoff_km=max_distance_km if max_distance_km is not None else 5.0 * float(lambda_km),
        mode=mode,
        weight_field=None,  # change if you want capacity weighting
        use_weights=use_weights
    )

    # choose array to plot
    if value_to_plot not in ("prob", "pop_weighted", "distance_km"):
        raise ValueError("value_to_plot must be 'prob', 'pop_weighted', or 'distance_km'")
    arr_plot = result[value_to_plot]

    # outputs
    out_dir = Path(output_dir) if output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _canonical_basename(country, lambda_km, output_name)
    tif_path = str(out_dir / f"{base}.tif") if write_tif else None
    png_path = str(out_dir / f"{base}.png")

    # overwrite guard
    if not overwrite:
        if tif_path and os.path.exists(tif_path):
            raise FileExistsError(f"File exists: {tif_path} (set overwrite=True)")
        if os.path.exists(png_path):
            raise FileExistsError(f"File exists: {png_path} (set overwrite=True)")

    # write GeoTIFF (numeric)
    if write_tif and tif_path:
        write_geotiff(
            path=tif_path,
            array=arr_plot,
            transform=result["transform"],
            crs=result["crs"],
            nodata=np.nan,
            compress="LZW",
        )

    # render PNG
    if value_to_plot == "distance_km":
        title = (
            f"Distance to Nearest LINAC (km)\n"
            f"(N facilities={result['stats']['n_facilities']})"
        )
    else:
        title = (
            f"Probability of Access to Cancer Treatment\n"
            f"(λ={lambda_km:.0f} km; Mean={result['stats']['p_mean']:.1%}; "
            f"N facilities={result['stats']['n_facilities']})"
        )
    render_png(
        array=arr_plot,
        kind=value_to_plot,
        out_png=png_path,
        title=title,
        dpi=dpi,
        show=show_plot,
    )

    return arr_plot, tif_path, png_path, result["stats"]
