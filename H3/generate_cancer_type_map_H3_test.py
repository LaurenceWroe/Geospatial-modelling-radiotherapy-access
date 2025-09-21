"""
Generate Cancer Type Maps (H3 compatible): 

These functions adapt the previous raster-based mapping workflow to work
with Kontur H3 population geopackages (gzipped gpkg) produced by the
download/load code. 

Workflow (per map request):
1. Load H3 population GeoDataFrame (gdf) containing columns:
     - 'h3' (H3 index string)
     - 'population' (numeric)
     - geometry (hex polygon)
2. Compute national case totals (from xarray DataArray).
3. Apportion the national total over H3 cells proportional to population:
     value_per_hex = total_cases * (population_hex / population_sum)
4. Optionally apply RT utilisation fractions or capacity weighting.
5. Save outputs:
     - GeoPackage of hexes with apportioned values (always)
     - PNG heatmap (polygon plotting) for GUI preview (always)
     - Optional GeoTIFF rasterization (if rasterize=True or template is provided)
6. Return a GUI-friendly dict with image bytes and paths/metadata.

Inputs: 
- country_code : ISO3 (string)
- gdf OR path to Kontur H3 geopackage (gzipped .gpkg.gz)
- cancer totals come from `da` (xarray DataArray) as before
- options for RT utilisation, capacity weighting (linac capacity), rasterize, etc.

Returns: 
A dict with:
  - status, country, cancers, mode, image_bytes, gpkg_path, tif_path (optional),
    message, original_population, apportioned_total, etc.
"""

import os
import h3 
import io
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.cm as cm
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds


# ------------- Utilities --------------
def _norm_key(s: str) -> str:
    """Normalize cancer names for robust dict matching."""
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())

def _load_rad_utilisation_csv(csv_path: str) -> Dict[str, float]:
    """
    Load a CSV where the *last* field on each line is the utilisation value
    and everything before it (which may contain commas) is the cancer name.

    Accepts comma-, tab-, or semicolon-separated lines. Supports values given
    as fractions (0–1) or percentages (e.g., "45" or "45%").
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    mapping: Dict[str, float] = {}

    # utf-8-sig handles possible BOM on first line
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Pick a delimiter present on this line
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            elif "\t" in line:
                parts = [p.strip() for p in line.split("\t")]
            elif ";" in line:
                parts = [p.strip() for p in line.split(";")]
            else:
                # Not a data line
                continue

            if len(parts) < 2:
                # Not enough fields
                continue

            value_str = parts[-1].strip().strip('"').strip("'")
            name = ", ".join([p for p in parts[:-1] if p != ""]).strip().strip('"').strip("'")
            if not name:
                continue

            # Parse value as fraction or percent
            try:
                if value_str.endswith("%"):
                    val = float(value_str[:-1]) / 100.0
                else:
                    val = float(value_str)
                    if 1.0 < val <= 100.0:  # treat bare % numbers as percent
                        val = val / 100.0
            except ValueError:
                raise ValueError(f"Bad numeric value on line {i} in {csv_path!s}: {value_str!r}")

            if 0.0 <= val <= 1.0:
                mapping[_norm_key(name)] = float(val)
            # silently skip out-of-range rows

    if not mapping:
        raise ValueError(f"No usable rows found in {csv_path}")

    return mapping

def _get_rt_utilisation_maps(
    optimal_csv_path: str,
    actual_dir: str,
    iso3: str
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """
    Load optimal RT utilisations and (optionally) actual RT utilisations for a given country.
    actual file path: {actual_dir}/{iso3}.csv
    """
    optimal_map =   _load_rad_utilisation_csv(optimal_csv_path)

    actual_path = os.path.join(actual_dir, f"{iso3.upper()}.csv")
    if os.path.exists(actual_path):
        actual_map =   _load_rad_utilisation_csv(actual_path)
    else:
        actual_map = None

    return optimal_map, actual_map

def _get_cases_for_cancers(
    da: xr.DataArray,
    iso3: str,
    cancer_types: List[str],
    metric_name: str = "New_Cases_Number",
) -> Dict[str, float]:
    """
    Extract per-cancer national case counts from the tensor for the given ISO3.
    Returns mapping cancer (normalized original string) -> cases (float).
    """
    cases_map: Dict[str, float] = {}
    for ct in cancer_types:
        # We try exact match first; fall back to case-insensitive match
        try:
            val = float(da.sel(ISO3=iso3, Cancer=ct, Metric=metric_name).item())
            cases_map[ct] = val
            continue
        except Exception:
            pass

        # Fuzzy match by normalized key among available cancers
        cancers_available = list(da.coords["Cancer"].values)
        target_key = _norm_key(ct)
        matched = None
        for cand in cancers_available:
            if _norm_key(cand) == target_key:
                matched = cand
                break
        if matched is None:
            # try substring containment (lowercased)
            low = ct.strip().lower()
            subset = [c for c in cancers_available if low in str(c).strip().lower()]
            if len(subset) == 1:
                matched = subset[0]

        if matched is None:
            raise ValueError(f"Cancer type '{ct}' not found in tensor for ISO3={iso3}.")

        val = float(da.sel(ISO3=iso3, Cancer=matched, Metric=metric_name).item())
        cases_map[matched] = val

    return cases_map

# ------------ H3-specific helpers ------------

def _load_h3_gdf_from_path(gpkg_gz_path: str) -> gpd.GeoDataFrame:
    """
    Load a Kontur H3 geopackage (can be .gpkg.gz) into a GeoDataFrame.

    Expects columns: 'h3', 'population' and geometry.
    """
    p = Path(gpkg_gz_path)
    if not p.exists():
        raise FileNotFoundError(f"H3 geopackage not found: {p}")
    # geopandas can read gzipped gpkg via zip-like path; Kontur file is a single .gpkg.gz
    # If geopandas can't read "zip://", try gpd.read_file(p) directly.
    try:
        gdf = gpd.read_file(f"zip://{p}")
    except Exception:
        gdf = gpd.read_file(str(p))
    if "h3" not in gdf.columns or "population" not in gdf.columns:
        raise ValueError("H3 geopackage must contain 'h3' and 'population' columns.")
    return gdf


def _apportion_over_h3(gdf: gpd.GeoDataFrame, total: float, population_col: str = "population") -> gpd.GeoDataFrame:
    """
    Apportion `total` across H3 hexes in `gdf` proportional to population_col.
    Returns a new GeoDataFrame with an added column 'apportioned'.
    """
    g = gdf.copy()
    # treat non-positive population as NaN (outside / missing)
    pop = pd.to_numeric(g[population_col], errors="coerce").astype(float)
    pop = pop.where(pop > 0, np.nan)
    total_pop = float(np.nansum(pop))
    if not np.isfinite(total_pop) or total_pop <= 0:
        raise ValueError("H3 population sum is non-positive; cannot apportion.")
    g["population_clean"] = pop
    g["apportioned"] = (g["population_clean"] / total_pop) * float(total)
    # ensure apportioned NaNs remain NaN for outside-country hexes
    g.loc[g["population_clean"].isna(), "apportioned"] = np.nan
    return g


def _plot_h3_heatmap_to_png_bytes(gdf: gpd.GeoDataFrame,
                                  value_col: str = "apportioned",
                                  title: Optional[str] = None,
                                  vmin: float = 1.0,
                                  vmax: Optional[float] = None,
                                  cmap_name: str = "viridis",
                                  figsize: Tuple[int, int] = (10, 8)) -> bytes:
    """
    Render a polygon heatmap (GeoDataFrame) to PNG bytes for GUI preview.
    Uses LogNorm scaling with vmin/vmax.
    """
    import matplotlib
    matplotlib.use("Agg")
    # Prepare data
    plot_gdf = gdf.copy()
    # Mask values <=0 or NaN for plotting as empty
    plot_gdf["plot_val"] = plot_gdf[value_col].where(np.isfinite(plot_gdf[value_col]) & (plot_gdf[value_col] > 0))
    # Create colormap with dark under color
    cmap = cm.get_cmap(cmap_name, 256)
    new_colors = cmap(np.linspace(0, 1, 256))
    dark_blue = np.array([0, 0, 139 / 255, 1.0])
    new_colors[0] = dark_blue
    custom_cmap = ListedColormap(new_colors)
    custom_cmap.set_under(dark_blue)
    # Prepare vmin/vmax
    if vmax is None:
        vmax = float(np.nanmax(plot_gdf["plot_val"])) if np.isfinite(np.nanmax(plot_gdf["plot_val"])) else vmin
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    # For GeoDataFrame plotting, GeoPandas will handle polygons
    plot_gdf.plot(column="plot_val", ax=ax, cmap=custom_cmap,
                  norm=LogNorm(vmin=vmin, vmax=vmax), linewidth=0, edgecolor=None)
    cbar = plt.cm.ScalarMappable(norm=LogNorm(vmin=vmin, vmax=vmax), cmap=custom_cmap)
    fig.colorbar(cbar, ax=ax, label="Cases per hex")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_axis_on()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200)
    buf.seek(0)
    data = buf.getvalue()
    plt.close(fig)
    return data


def _save_h3_geopackage(gdf: gpd.GeoDataFrame, out_path: str, layer_name: Optional[str] = None) -> str:
    """
    Save GeoDataFrame to a GeoPackage. Returns the path string.
    """
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # geopandas will create the .gpkg; layer name optional
    gdf.to_file(outp, driver="GPKG")
    return str(outp)


def _rasterize_h3_gdf_to_geotiff(gdf: gpd.GeoDataFrame,
                                 value_col: str,
                                 out_tif: str,
                                 pixel_size_deg: Optional[float] = None,
                                 template_raster: Optional[str] = None,
                                 nodata: float = -9999.0):
    """
    Rasterize GeoDataFrame polygons to a GeoTIFF using value_col.
    Either provide template_raster (to copy transform/shape/crs) or provide pixel_size_deg
    (pixel size in degrees; we'll compute a bounds-based transform).
    """
    if template_raster:
        if not os.path.exists(template_raster):
            raise FileNotFoundError(f"Template raster not found: {template_raster}")
        with rasterio.open(template_raster) as src:
            meta = src.meta.copy()
            width = src.width
            height = src.height
            transform = src.transform
            crs = src.crs
    else:
        # compute bounds and transform in EPSG:4326
        bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = bounds
        if pixel_size_deg is None:
            raise ValueError("If template_raster not provided, pixel_size_deg must be set.")
        # compute width/height
        width = int(np.ceil((maxx - minx) / pixel_size_deg))
        height = int(np.ceil((maxy - miny) / pixel_size_deg))
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        crs = "EPSG:4326"
        meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": nodata,
            "count": 1,
            "height": height,
            "width": width,
            "transform": transform,
            "crs": crs,
            "compress": "DEFLATE",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }

    # Prepare shapes for rasterize: (geometry, value)
    shapes = ((geom, float(val)) for geom, val in zip(gdf.geometry, gdf[value_col].fillna(nodata)))

    # Rasterize (note: rasterio writes row-major arrays)
    out_arr = rasterize(shapes=shapes, out_shape=(height, width), transform=transform,
                        fill=nodata, dtype="float32")

    # Write
    outp = Path(out_tif)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(outp, "w", **meta) as dst:
        dst.write(out_arr.astype(np.float32), 1)
    return str(outp)


# --------- Main H3-aware mapping function  ----------

def generate_cancer_type_map_h3(
    country_iso3: str,
    h3_gpkg_path: Optional[str] = None,
    h3_gdf: Optional[gpd.GeoDataFrame] = None,
    da: Optional[object] = None,  # xarray.DataArray (same contract as before)
    cancer_type: Optional[str] = None,
    cancer_types: Optional[List[str]] = None,
    include_RT_utilisation: bool = False,
    include_optimal_RT_utilisation: bool = False,
    optimal_rt_csv_path: str = "b_cancer_incidence/optimal_rt_utilisations.csv",
    actual_rt_dir: str = "b_cancer_incidence/actual_data",
    include_capacity_weighted: bool = False,
    linac_capacity: Optional[float] = None,
    n_linacs: Optional[int] = None,
    rasterize_output: bool = False,
    template_raster: Optional[str] = None,
    pixel_size_deg: Optional[float] = None,
    output_dir: Optional[str] = None,
    basename: Optional[str] = None,
    return_image: bool = True,
    overwrite: bool = False,
) -> Dict:
    """
    H3-aware variant of the previous raster mapping function.

    Key differences:
    - Uses H3 hex polygons & population (gdf) instead of raster pixels.
    - Produces a GeoPackage of hexes with apportioned values and a PNG heatmap.
    - Optionally rasterizes hexes to GeoTIFF (template raster or pixel_size_deg required).
    - Returns a dict with image bytes and file paths (GUI-friendly).
    """
    # 1) Load H3 population (either gdf provided or path)
    if h3_gdf is None:
        if not h3_gpkg_path:
            # auto-locate using your download naming convention (ISO2 expected in file name)
            raise ValueError("Provide h3_gdf or h3_gpkg_path")
        h3_gdf = _load_h3_gdf_from_path(h3_gpkg_path)

    # sanitize inputs for cancers
    if cancer_types is None:
        if cancer_type is None:
            raise ValueError("Provide cancer_type (str) or cancer_types (list).")
        cancer_types = [cancer_type]

    # Auto-load DataArray if provided None — keep your _load_default_da if available
    if da is None:
        from xarray import load_dataarray
        # try default path
        da = load_dataarray("b_cancer_incidence/globocan_xarray.nc")

    # 2) Compute national totals for cancers (reuse your _get_cases_for_cancers)
    cases_map = _get_cases_for_cancers(da, country_iso3, cancer_types, metric_name="New_Cases_Number")
    incidence_total = float(sum(cases_map.values()))

    # 3) determine the `total` according to RT utilisation flags (reuse your CSV loader logic)
    optimal_map, actual_map = _get_rt_utilisation_maps(optimal_rt_csv_path, actual_rt_dir, country_iso3)
    used_optimal_for_missing_actual = False
    missing_actual_for = []

    if include_optimal_RT_utilisation:
        total = 0.0
        for ct_name, cases in cases_map.items():
            frac = optimal_map.get(_norm_key(ct_name))
            if frac is None:
                raise ValueError(f"No optimal RT utilisation for cancer: {ct_name}")
            total += cases * frac
        title_mode = "Optimal RT-treated cases"

    elif include_RT_utilisation:
        total = 0.0
        if actual_map is None:
            used_optimal_for_missing_actual = True
            missing_actual_for = list(cases_map.keys())
            for ct_name, cases in cases_map.items():
                frac = optimal_map.get(_norm_key(ct_name))
                if frac is None:
                    raise ValueError(f"No optimal RT utilisation for '{ct_name}' while falling back.")
                total += cases * frac
        else:
            for ct_name, cases in cases_map.items():
                key = _norm_key(ct_name)
                frac = actual_map.get(key)
                if frac is None:
                    used_optimal_for_missing_actual = True
                    missing_actual_for.append(ct_name)
                    frac = optimal_map.get(key)
                    if frac is None:
                        raise ValueError(f"No RT utilisation (actual/optimal) for '{ct_name}'.")
                total += cases * frac
        title_mode = "Actual RT-treated cases"
        if used_optimal_for_missing_actual:
            title_mode += " (fallback to optimal for some cancers)"
    else:
        total = incidence_total
        title_mode = "New cases"

    # 4) Apportion over H3 hexes
    h3_with_vals = _apportion_over_h3(h3_gdf, total, population_col="population")

    # 5) Capacity weighting if requested
    if include_capacity_weighted:
        if linac_capacity is None and n_linacs is None:
            raise ValueError("For capacity weighting, provide linac_capacity and/or n_linacs")
        # If n_linacs provided as None, optionally try a lookup similar to your get_n_liancs_from_excel
        if n_linacs is None:
            # keep previous behaviour: call the get_n_liancs_from_excel() if present
            from inspect import getsource
            try:
                from . import get_n_liancs_from_excel  # if same module
                n_linacs = get_n_liancs_from_excel(country_iso3[:3])  # adjust as needed
            except Exception:
                n_linacs = None
        total_capacity = linac_capacity * n_linacs if (linac_capacity is not None and n_linacs is not None) else None
        if total_capacity is not None:
            apportioned_sum = float(np.nansum(h3_with_vals["apportioned"].fillna(0.0)))
            if apportioned_sum > total_capacity:
                # scale down proportionally
                scale_factor = total_capacity / apportioned_sum
                h3_with_vals["apportioned"] = h3_with_vals["apportioned"] * scale_factor
                title_mode += " (capacity-limited)"
            else:
                title_mode += " (below-capacity)"

    # 6) Prepare outputs
    # Output paths
    if output_dir is None:
        output_dir = Path("b_cancer_incidence/cancer_type_maps_h3")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if basename is None:
        safe_label = "_".join([str(ct).strip().replace(" ", "_") for ct in cancer_types])
        base_name = f"{country_iso3.lower()}_{safe_label}"
    else:
        base_name = basename

    suffix = "incidence"
    if include_optimal_RT_utilisation:
        suffix = "optimally_treated"
    elif include_RT_utilisation:
        suffix = "treated"
    if include_capacity_weighted:
        suffix = f"{suffix}_capacity_weighted"

    gpkg_path = str(output_dir / f"{base_name}_{suffix}.gpkg")
    # Always save a vector geopackage (apportioned per hex)
    _save_h3_geopackage(h3_with_vals.drop(columns=["population_clean"], errors="ignore"), gpkg_path)

    # Produce PNG preview
    cancers_title = " + ".join([str(ct) for ct in cancer_types])
    title = f"{country_iso3} — {cancers_title} ({title_mode})"
    if include_RT_utilisation and used_optimal_for_missing_actual and missing_actual_for:
        title += f" [no actual for: {', '.join(missing_actual_for)}]"

    # Determine vmax reference: we keep baseline incidence per-hex as comparand if desired
    # compute baseline incidence per hex (incidence_total apportioned)
    baseline_gdf = _apportion_over_h3(h3_gdf, incidence_total, population_col="population")
    baseline_max = float(np.nanmax(baseline_gdf["apportioned"].fillna(0.0)))

    png_bytes = _plot_h3_heatmap_to_png_bytes(h3_with_vals, value_col="apportioned",
                                             title=title, vmin=1.0, vmax=baseline_max)

    # Optional rasterization to GeoTIFF
    tif_path = None
    if rasterize_output:
        tif_path = str(output_dir / f"{base_name}_{suffix}.tif")
        _rasterize_h3_gdf_to_geotiff(h3_with_vals, "apportioned", tif_path,
                                     template_raster=template_raster, pixel_size_deg=pixel_size_deg)

    return {
        "status": "ok",
        "country": country_iso3,
        "cancers": cancer_types,
        "mode": title_mode,
        "image_bytes": png_bytes if return_image else None,
        "gpkg_path": gpkg_path,
        "tif_path": tif_path,
        "original_population": float(np.nansum(h3_gdf["population"].where(h3_gdf["population"] > 0, np.nan))),
        "apportioned_total": float(np.nansum(h3_with_vals["apportioned"].fillna(0.0))),
        "message": f"H3 map generated for {country_iso3}, mode={title_mode}"
    }

