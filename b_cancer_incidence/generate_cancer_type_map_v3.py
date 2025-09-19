"""
Generate cancer-type maps by apportioning national case counts (from an xarray tensor)
over a population raster, optionally multiplying by RT utilisation percentage (optimal or actual).

*New Feature* 
This will now produce capacity-weighted treated/optimally-treated maps. --> GUI will have option to select this. 

Replaces the previous Excel-driven proportions with:
- an xarray DataArray `da` providing New_Cases_Number,
- a CSV mapping cancer -> optimal RT utilisation,
- an optional per-country CSV mapping cancer -> actual treated RT utilisation.

Outputs:
- GeoTIFF of apportioned counts per pixel,
- PNG heatmap with LogNorm and consistent styling,
- Always also emits a PNG for raw population density (for comparison).

original Sophia + Alika, v2 by Archie: adapted for tensor+CSV workflow, v3 by Sophia: adding capacity-weighted treated/optimally-treated maps
"""

import os
import io
from pathlib import Path
from token import OP
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import xarray as xr

# ---- Paths ----
DEFAULT_OPTIMAL_RT_CSV = "b_cancer_incidence/optimal_rt_utilisations.csv"
DEFAULT_ACTUAL_RT_DIR  = "b_cancer_incidence/actual_data"
DEFAULT_XARRAY_PATH    = "b_cancer_incidence/globocan_xarray.nc"


# ----------------- Utilities ----------------------------

def _load_default_da(xarray_path: Optional[str] = None) -> xr.DataArray:
    """
    Load the default cancer tensor DataArray from disk.
    Expects dims ['Cancer','Metric','ISO3'] (order can vary).
    """
    p = Path(xarray_path) if xarray_path else DEFAULT_XARRAY_PATH
    #if not p.exists():
    #    raise FileNotFoundError(f"Xarray DataArray file not found: {p}")
    da = xr.load_dataarray(p)

    required = {"Cancer", "Metric", "ISO3"}
    missing = required.difference(set(da.dims))
    if missing:
        raise ValueError(f"Tensor missing dims {missing}; found dims: {list(da.dims)}")
    return da

def _fmt_res(res: float) -> str:
    """Format a resolution value consistently for filenames/paths."""
    s = f"{res:.1f}"
    return s.rstrip('0').rstrip('.')

def get_n_liancs_from_excel(country_code: str) -> int: 
    """ 
    determining the number of linacs in each country by reading 
    the the number rof coulmns in the excel file
    """
    excel_path = f"c_probability_of_access/linac/{country_code}_DIRAC.xlsx"
    df = pd.read_excel(excel_path) 

    n_linacs = df.shape[1] -1 

    if n_linacs < 1: 
        raise ValueError(f"No linacs founc in {excel_path}") 
    return n_linacs 

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


def save_raster_like(
    template_raster_path: str,
    array: np.ndarray,
    output_path: str,
    nodata_value: float = -9999.0,
) -> None:
    """
    Save array as GeoTIFF using spatial metadata from template raster.
    Writes nodata for NaNs; applies compression and tiling for performance.
    """
    if not os.path.exists(template_raster_path):
        raise FileNotFoundError(f"Template raster not found: {template_raster_path}")

    with rasterio.open(template_raster_path) as src:
        meta = src.meta.copy()

    meta.update({
        "count": 1,
        "dtype": "float32",
        "nodata": nodata_value,
        "compress": "DEFLATE",
        "predictor": 2,   # good for floats
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    })

    arr = np.asarray(array, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, nodata_value).astype(np.float32)

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(arr, 1)

# ---------------- Population-only PNG -------------------

def generate_population_density_map_only(
    country_code: str,
    population_raster_path: str,
    output_dir: Path,
    resolution: float = 1.0,
    return_image: bool = True,
    overwrite_existing: bool = False,
) -> Tuple[Optional[bytes], str, str]:
    """
    Generate and save a raw population density map (GeoTIFF + PNG).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(population_raster_path) as src:
        if src.count < 1:
            raise RuntimeError(f"Raster has no bands: {population_raster_path}")
        population = src.read(1)
        bounds = src.bounds

    # Mask outside-country / non-positive population
    population = np.where(population > 0, population, np.nan)

    # Output filenames
    basename = f"{country_code.lower()}_population_density_{_fmt_res(resolution)}km"
    output_tif = os.path.join(output_dir, f"{basename}.tif")
    output_png = os.path.join(output_dir, f"{basename}.png")

    # Save raster (copy of population density)
    if not os.path.exists(output_tif) or overwrite_existing:
        if population.ndim != 2:
            raise ValueError(f"Population array is not 2D. Shape: {population.shape}")
        save_raster_like(population_raster_path, population, output_tif)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Values <1 are set to 0.5 so they fall into the lowest log bin.
    norm_data = population.copy()
    norm_data[(norm_data < 1) & (~np.isnan(norm_data))] = 0.5

    cmap = cm.get_cmap("viridis", 256)
    new_colors = cmap(np.linspace(0, 1, 256))
    dark_blue = np.array([0, 0, 139 / 255, 1.0])
    new_colors[0] = dark_blue
    custom_cmap = ListedColormap(new_colors)
    custom_cmap.set_under(dark_blue)

    im = ax.imshow(
        norm_data,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap=custom_cmap,
        norm=LogNorm(vmin=1, vmax=np.nanmax(norm_data)),
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Population density (people/km²)")
    ax.set_title(f"{country_code.upper()} — Population Density")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()

    if not os.path.exists(output_png) or overwrite_existing:
        plt.savefig(output_png, dpi=300)

    image_bytes = None
    if return_image:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        image_bytes = buf.getvalue()

    plt.close()

    return image_bytes, output_tif, output_png

# --------------- Core computation -----------------------

def _apportion_total_over_population(population_raster_path: str, total: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a population raster and a national total, apportion the total over pixels
    proportional to population (population-weighted allocation).

    Returns (population_masked, apportioned_array)
    """
    with rasterio.open(population_raster_path) as src:
        pop = src.read(1).astype(np.float64)

    pop = np.where(pop > 0, pop, np.nan)  # outside-country -> NaN
    denom = np.nansum(pop)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Population raster sum is non-positive or NaN; cannot apportion.")

    scale = float(total) / denom
    apportioned = pop * scale  # cases per pixel (can be fractional)
    return pop, apportioned


# --------------- Main mapping function ------------------

def generate_cancer_type_map(
    country_code: str,
    cancer_type: Optional[str] = None,
    cancer_types: Optional[List[str]] = None,
    resolution: float = 1.0,
    population_raster_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    basename: Optional[str] = None,
    global_vmin: float = 1.0,
    global_vmax: Optional[float] = None,
    return_image: bool = True,
    overwrite_cancer_type_map: bool = False,
    include_RT_utilisation: bool = False,           # "actual treated" if file exists, else falls back to optimal
    include_optimal_RT_utilisation: bool = False,   # "optimal treated"
    optimal_rt_csv_path: str = DEFAULT_OPTIMAL_RT_CSV,
    actual_rt_dir: str = DEFAULT_ACTUAL_RT_DIR,
    da: Optional[xr.DataArray] = None,
    xarray_path: Optional[str] = None,
    include_capacity_weighted: bool = False, 
    linac_capacity: Optional[float] = None,
    n_linacs: Optional[int] = None,

) -> Tuple[Optional[bytes], str, str]:
    """
    Generate a cancer-type map using xarray tensor totals (New_Cases_Number),
    population-weighted apportioning, and RT utilisations from CSVs.

    Args:
        country_code: ISO3 (e.g., "GBR")
        cancer_type / cancer_types: one or more cancer names present in da.Cancer
        resolution: km resolution (used for file naming/path inference)
        population_raster_path: path to population raster (.tif)
        output_dir: base output directory
        basename: custom base name for outputs
        global_vmin/global_vmax: log scale bounds
        return_image: return PNG bytes for UI
        overwrite_cancer_type_map: allow overwrite
        include_RT_utilisation: if True, try per-country "actual" treated; fallback to optimal
        include_optimal_RT_utilisation: if True, use optimal treated RT utilisation
        optimal_rt_csv_path: CSV of cancer -> optimal RT utilisation
        actual_rt_dir: directory with per-country CSVs of cancer -> actual RT utilisation
        da: xarray DataArray with dims ["Cancer", "Metric", "ISO3"]
        xarray_path: path to DataArray
        include_capacity_weighted: if True, returns capacity-weighted treated/optimally-treated maps
        linac_capacity: user input of linac capacity (patients/year)
        n_linacs: number of linacs in a country (found from excel spreadsheets) 

    Returns:
        (image_bytes | None, output_tif_path, output_png_path)
    """

    # Auto-load default DataArray if not provided
    if da is None:
        da = _load_default_da(xarray_path=xarray_path)

    # Normalize inputs
    iso3 = country_code.upper()
    if cancer_types is None:
        if cancer_type is None:
            raise ValueError("Provide cancer_type (str) or cancer_types (list).")
        cancer_types = [cancer_type]

    if include_RT_utilisation and include_optimal_RT_utilisation:
        # If both toggled, prefer optimal to avoid ambiguity.
        include_RT_utilisation = False
    
    # ----- Linac capacity handling ----- 
    total_capacity = None 
    # --- adding n_linac definition here as well as in GUI --- 
    n_linacs = get_n_liancs_from_excel(country_code) 
    if include_capacity_weighted: 
        if linac_capacity is None: 
            raise ValueError("When include_capacity_weighted is True, you must provide linac_capacity") 
        total_capacity = linac_capacity * n_linacs if n_linacs is not None else linac_capacity

    


    # Resolve default directories relative to repo root (two-levels up from this file)
    base_dir = Path(__file__).resolve().parents[1]

    # Resolve population raster path if None
    if population_raster_path is None:
        res_str = _fmt_res(resolution)
        pop_default = base_dir / "a_population_density" / "resampled" / f"{iso3.lower()}_{res_str}km.tif"
        population_raster_path = str(pop_default)

    if not os.path.exists(population_raster_path):
        raise FileNotFoundError(f"Population raster not found: {population_raster_path}")

    # Output directory based on map type --> changes to include capacity-weighted maps 
    if output_dir is None:
        output_dir_path = base_dir / "b_cancer_incidence" / "cancer_type_maps"
        
        if include_optimal_RT_utilisation:
            if include_capacity_weighted:
                output_dir_path /= "optimally_treated_capacity_weighted"
            else:
                output_dir_path /= "optimally_treated"
        
        elif include_RT_utilisation:
            if include_capacity_weighted:
                output_dir_path /= "treated_maps_capacity_weighted"
            else:
                output_dir_path /= "treated_maps"
        
        else:
            output_dir_path /= "incidence_maps"
    else:
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load RT RT_utilisation maps
    optimal_map, actual_map = _get_rt_utilisation_maps(optimal_rt_csv_path, actual_rt_dir, iso3)

    # Fetch national case counts for requested cancers
    cases_map = _get_cases_for_cancers(da, iso3, cancer_types, metric_name="New_Cases_Number")


    # --- Always compute baseline incidence for scaling ---
    incidence_total = float(sum(cases_map.values()))
    population, baseline_incidence = _apportion_total_over_population(
    population_raster_path, incidence_total
    )

    # --- Compute mode-specific totals --- 
    used_optimal_for_missing_actual = False
    missing_actual_for = []

    if include_optimal_RT_utilisation:
        # Sum cases * optimal RT utilisation per cancer
        total = 0.0
        for ct_name, cases in cases_map.items():
            key = _norm_key(ct_name)
            frac = optimal_map.get(key)
            if frac is None:
                raise ValueError(f"No optimal RT utilisation found in CSV for cancer: '{ct_name}'")
            total += cases * frac

        title_mode = "Optimal RT-treated cases"
        cbar_label = "Estimated RT-treated cases per pixel"

    elif include_RT_utilisation:
        # Try actual per-country; fall back to optimal for missing cancers or whole file
        total = 0.0
        if actual_map is None:
            # No actual file at all → full fallback
            used_optimal_for_missing_actual = True
            missing_actual_for = list(cases_map.keys())

            for ct_name, cases in cases_map.items():
                key = _norm_key(ct_name)
                frac = optimal_map.get(key)
                if frac is None:
                    raise ValueError(f"No optimal RT utilisation found for '{ct_name}' while falling back.")
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
                        raise ValueError(f"No RT utilisation (actual/optimal) found for '{ct_name}'.")
                total += cases * frac

        title_mode = "Actual RT-treated cases"
        if used_optimal_for_missing_actual:
            title_mode += " (fallback to optimal for some cancers)"
        cbar_label = "Estimated RT-treated cases per pixel"

    else:
        # Incidence: sum cases (no utilisation)
        total = incidence_total
        title_mode = "New cases"
        cbar_label = "Estimated new cases per pixel"
    
    # Apportion total over population 
    population, array = _apportion_total_over_population(population_raster_path, total)
 
    # --- Apply Linac capacity scaling --- 
    if include_capacity_weighted and total_capacity is not None: 
        if total > total_capacity:
            # scale down all pixel values proportionally  
            scale_factor = total_capacity / total 
            array = array * scale_factor 
            title_mode += " (capacity-limited)" 
        
        else: 
            scale_factor = total_capacity / total if total > 0 else 1.0 
            array = array * scale_factor
            title_mode += " (below-capacity)" 

    
    # Also generate the population density map (separate folder)
    population_output_dir = base_dir / "a_population_density" / "population_density_maps"
    population_output_dir.mkdir(parents=True, exist_ok=True)
    generate_population_density_map_only(
        country_code=iso3,
        population_raster_path=population_raster_path,
        output_dir=population_output_dir,
        resolution=resolution,
        return_image=False,
        overwrite_existing=overwrite_cancer_type_map,
    )

    # Filenames
    if basename is None:
        safe_label = "_".join([str(ct).strip().replace(" ", "_") for ct in cancer_types])
        base_name = f"{iso3.lower()}_{safe_label}_{_fmt_res(resolution)}km"
    else:
        base_name = basename

    # --- updating suffix block to include capacity-weighted maps --- 
    
    if include_optimal_RT_utilisation:
        suffix = f"optimally_treated_capacity_{linac_capacity}" if include_capacity_weighted else "optimally_treated"
    
    elif include_RT_utilisation:
        suffix = f"treated_capacity_{linac_capacity}" if include_capacity_weighted else "treated"
    else:
        suffix = "incidence"


    output_tif = os.path.join(output_dir_path, f"{base_name}_{suffix}_density.tif")
    output_png = os.path.join(output_dir_path, f"{base_name}_{suffix}_density.png")

    # Respect overwrite flag for PNG
    if not overwrite_cancer_type_map and os.path.exists(output_png):
        image_bytes = None
        if return_image:
            with open(output_png, "rb") as f:
                image_bytes = f.read()
        return image_bytes, output_tif, output_png

    # Save GeoTIFF (array already NaN outside country; save_raster_like converts to nodata)
    save_raster_like(population_raster_path, array, output_tif)

    # Plot PNG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(population_raster_path) as src:
        bounds = src.bounds

    in_country_mask = np.isfinite(population)
    plot_data = array.copy()

    # Mask to show only valid in-country positive values
    plot_data_masked = np.full_like(plot_data, np.nan, dtype=np.float32)
    pos_mask = in_country_mask & np.isfinite(plot_data) & (plot_data > 0)
    plot_data_masked[pos_mask] = plot_data[pos_mask]

    # Values <1 get a small dummy value to appear in the lowest log bin
    norm_data = plot_data_masked.copy()
    norm_data[(norm_data < 1) & (~np.isnan(norm_data))] = 0.5

    vmin = global_vmin if global_vmin is not None else 1.0
    #vmax = global_vmax if global_vmax is not None else np.nanmax(norm_data)
    # --- Fixing the colour scaling so maps can be compared --- 
    if global_vmax is not None:
        vmax = global_vmax
    else:
        vmax = np.nanmax(baseline_incidence)  # <-- fixed reference for comparability

    cmap = cm.get_cmap("viridis", 256)
    new_colors = cmap(np.linspace(0, 1, 256))
    dark_blue = np.array([0, 0, 139 / 255, 1.0])
    new_colors[0] = dark_blue
    custom_cmap = ListedColormap(new_colors)
    custom_cmap.set_under(dark_blue)

    # Title
    cancers_title = " + ".join([str(ct) for ct in cancer_types])
    title = f"{iso3} — {cancers_title} ({title_mode})"
    if include_RT_utilisation and used_optimal_for_missing_actual and missing_actual_for:
        title += f" [no actual for: {', '.join(missing_actual_for)}]"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        norm_data,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap=custom_cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)

    image_bytes = None
    if return_image:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        image_bytes = buf.getvalue()
    plt.close()

    #return image_bytes, output_tif, output_png
    # --- changing return to include capacity-weighted maps --- 
    return {
        "status": "ok",
        "country": iso3,
        "cancers": cancer_types,
        "mode": title_mode,
        "linac_capacity": linac_capacity,
        "n_linacs": n_linacs,
        "image_bytes": image_bytes if return_image else None,
        "tif_path": output_tif,
        "png_path": output_png,
        "message": f"Map generated for {iso3}, cancers={cancer_types}, mode={title_mode}"
    }



# ---------------- Example usage -------------------------
# Assuming you already have `da` loaded in memory:
#
# image_bytes, tif_path, png_path = generate_cancer_type_map(
#     da=da,
#     country_code="GBR",
#     cancer_types=["Breast", "Lung"],
#     resolution=1.0,
#     include_optimal_RT_utilisation=True,  # or include_RT_utilisation=True
#     optimal_rt_csv_path="b_cancer_incidence/optimal_rt_utilisations.csv",
#     actual_rt_dir="b_cancer_incidence/actual_data",
# )
#
# This will also emit a population density PNG next to:
#   a_population_density/population_density_maps/