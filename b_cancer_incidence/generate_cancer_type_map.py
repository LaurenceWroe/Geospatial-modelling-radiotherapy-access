#!/usr/bin/env python3
"""
Generate a country map for a selected cancer type by multiplying the
population density raster with the by the cancer-type proportion from an Excel file.
THIS IS THE NEW FUNCTION THAT WORKS WITH THE GUI BETTER AND IS NOT DEPENDENT ON COMMAND LINE INPUTS
"""

import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List 
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io


# DEFAULT_EXCEL_PATH = "/Users/sophiamartin/Desktop/src/b_cancer_incidence/cancer_type_radiotherapy.xlsx"
DEFAULT_EXCEL_PATH = "b_cancer_incidence/cancer_type_radiotherapy.xlsx"  # Relative path for the project structure

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_cancer_fractions(excel_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Load cancer type proportions from the Excel file.
    Returns a mapping: lowercased cancer type -> (proportion, fraction, optimal_fraxtion)
    """
    df = pd.read_excel(excel_path, sheet_name=0)
    df = _normalize_columns(df)

    # Find best matching columns
    col_map = {
        "type": None,
        "prop": None,
        "frac": None,
        "optimal_frac": None
    }
    for c in df.columns:
        lc = c.strip().lower()
        if col_map["type"] is None and (lc == "cancer type" or lc.startswith("cancer type")):
            col_map["type"] = c
        if col_map["prop"] is None and (lc == "proportion of cases" or "proportion" in lc):
            col_map["prop"] = c
        if col_map["frac"] is None and (lc == "fraction" or "fraction" in lc):
            col_map["frac"] = c
        if col_map["optimal_frac"] is None and (lc == "optimal fraction" or "optimal" in lc): 
            col_map["optimal_frac"] = c

    if col_map["type"] is None or col_map["prop"] is None or col_map["frac"] is None or col_map["optimal_frac"] is None:
        raise ValueError(
            f"Could not find required columns in {excel_path}. Found columns: {list(df.columns)}"
        )

    # Build mapping
    mapping = {}
    for _, row in df.iterrows():
        ct = str(row[col_map["type"]]).strip()
        if ct.lower() in ("nan", "", "none"):
            continue
        try:
            prop = float(row[col_map["prop"]])
            frac = float(row[col_map["frac"]])
            opt_frac = float(row[col_map["optimal_frac"]]) 
        except Exception:
            continue
        mapping[ct.lower()] = (prop, frac, opt_frac)

    if not mapping:
        raise ValueError(f"No valid cancer-type proportions found in {excel_path}")

    return mapping


def multiply_population_by_fraction(
    population_raster_path: str,
    proportion: float,
    fraction: float,
) -> np.ndarray:
    """Load population raster and return population * proportion * fraction array."""
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
    cancer_case_prop = 0.043 #number of people wth cancer in UK is approx 3 mil, and UK pop. approx 70 mil. 
    # Clean population values (remove nodata / negatives)
    population = np.where(population > 0, population, 0)

    # Multiply by proportion and fraction
    result = population.astype(np.float64) * cancer_case_prop * float(proportion) * float(fraction)

    return population, result


def save_raster_like(
    template_raster_path: str,
    array: np.ndarray,
    output_path: str,
    nodata_value: float = -9999.0,
) -> None:
    """Save array as GeoTIFF using spatial metadata from template raster."""
    with rasterio.open(template_raster_path) as src:
        meta = src.meta.copy()
        meta.update({
            "count": 1,
            "dtype": "float32",
            "nodata": nodata_value,
        })
        transform = src.transform
        crs = src.crs

    with rasterio.open(output_path, "w", **meta) as dst:
        # write array, converting NaNs to nodata
        array = np.array(array)  # Force conversion
        if not np.issubdtype(array.dtype, np.number):
            try:
                array = array.astype(np.float32)
            except Exception as e:
                raise TypeError(f"Array could not be converted to float32: dtype={array.dtype}, error: {e}")
        data = np.where(np.isfinite(array), array, nodata_value).astype(np.float32)
        dst.write(data, 1)

#can also produce populatiom density map: 
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
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io

    with rasterio.open(population_raster_path) as src:
        if src.count < 1:
            raise RuntimeError(f"Raster has no bands: {population_raster_path}")
        population = src.read(1)
        bounds = src.bounds

    population = np.where(population > 0, population, np.nan)

    # Output filenames
    basename = f"{country_code.lower()}_population_density_{resolution}km"
    output_tif = os.path.join(output_dir, f"{basename}.tif")
    output_png = os.path.join(output_dir, f"{basename}.png")

    # Save raster
    if not os.path.exists(output_tif) or overwrite_existing:
        if population.ndim != 2:
            raise ValueError(f"Population array is not 2D. Shape: {population.shape}")
        save_raster_like(population_raster_path, population, output_tif)

    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 8))
    norm_data = population.copy()
    norm_data[(norm_data < 1) & (~np.isnan(norm_data))] = 0.5

    cmap = cm.get_cmap('viridis', 256)
    new_colors = cmap(np.linspace(0, 1, 256))
    new_colors[0] = np.array([0, 0, 139 / 255, 1.0])
    custom_cmap = ListedColormap(new_colors)

    im = ax.imshow(
        norm_data,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap=custom_cmap,
        norm=LogNorm(vmin=1, vmax=np.nanmax(norm_data))
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
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        image_bytes = buf.getvalue()

    plt.close()

    return image_bytes, output_tif, output_png

def multiply_population_by_multiplier(
    population_raster_path: str,
    multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load population raster and return population * multiplier array."""
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
        if not np.issubdtype(population.dtype, np.number):
            population = population.astype(np.float32)
    cancer_case_prop = 0.043
    population = np.where(population > 0, population, 0)
    result = population.astype(np.float64) * float(multiplier) * cancer_case_prop
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"Resulting array contains non-numeric data. dtype={result.dtype}")
    return population, result


def generate_cancer_type_map(
    country_code: str,
    cancer_type: Optional[str] = None,
    cancer_types: Optional[List[str]]= None,
    resolution: float = 1.0,
    excel_path: str = DEFAULT_EXCEL_PATH,
    population_raster_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    basename: Optional[str] = None,
    global_vmin: float = 1e-6,
    global_vmax: Optional[float] = None,
    return_image: bool = True,
    overwrite_cancer_type_map: bool = False,
    include_fraction: bool = False,
    include_optimal_fraction: bool = False,
) -> Tuple[Optional[bytes], str, str, str]:
    """
    Main function to generate cancer type map.
    
    Args:
        country_code: ISO country code (e.g., NGA, GBR, USA)
        cancer_type: Cancer type name as listed in Excel
        resolution: Resolution in km
        excel_path: Path to Excel with cancer-type proportions
        population_raster_path: Path to population raster (optional)
        output_dir: Directory to save outputs (optional)
        basename: Custom base name for outputs (optional)
        global_vmin: Minimum value for log color scale
        global_vmax: Maximum value for color scale
        return_image: Whether to return image bytes for display
    
    Returns:
        Tuple of (image_bytes, output_tif_path, output_png_path)
    """


    # Load cancer fractions
    fractions = load_cancer_fractions(excel_path)
    
    array = None
    population = None
    combined_label = []

    for idx, ct in enumerate(cancer_types):
        ct_key = ct.strip().lower()
        if ct_key not in fractions:
            # Try substring match
            matches = [k for k in fractions.keys() if ct_key in k]
            if len(matches) == 1:
                ct_key = matches[0]
            else:
                raise ValueError(f"Cancer type '{ct}' not found.")

        prop, frac, opt_frac = fractions[ct_key]
        combined_label.append(ct_key.replace(" ", "_"))

        if include_optimal_fraction: 
            multiplier = prop * opt_frac 
        elif include_fraction: 
            multiplier = prop * frac 
        else: 
            multiplier = prop 

        pop, temp_array = multiply_population_by_multiplier(population_raster_path, multiplier)

        if array is None:
            array = temp_array
            population = pop
        else:
            array += temp_array
        
        if array is None or array.ndim != 2:
            raise ValueError(f"Invalid array generated: shape {getattr(array, 'shape', 'None')}")
    # Resolve default paths
    base_dir = Path(__file__).resolve().parents[1]
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = base_dir / "b_cancer_incidence" / "cancer_type_maps" 
        if include_optimal_fraction:
            output_dir /= "optimally_treated"
        elif include_fraction:
            output_dir /= "treated_maps"
        else:
            output_dir /= "incidence_maps"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default population raster path if not provided
    if population_raster_path is None:
        resolution_str = f"{resolution:.1f}"
        pop_default = base_dir / "a_population_density" / "resampled" / f"{country_code.lower()}_{resolution_str}km.tif"
        population_raster_path = str(pop_default)

    # Construct the base name for output files
    if basename is None:
        safe_label = "_".join(combined_label)
        base_name = f"{country_code.lower()}_{safe_label}_{resolution}km"
    else:
        base_name = basename

    # Choose suffix based on selected option
    if include_optimal_fraction:
        suffix = "optimally_treated"
    elif include_fraction:
        suffix = "treated"
    else:
        suffix = "incidence"
    
    # Generate population density map (4th map)
    population_output_dir = base_dir / "a_population_density" / "population_density_maps"
    population_output_dir.mkdir(parents=True, exist_ok=True)

    generate_population_density_map_only(
        country_code=country_code,
        population_raster_path=population_raster_path,
        output_dir=population_output_dir,
        resolution=resolution,
        return_image=True,
        overwrite_existing=overwrite_cancer_type_map
    )

    # Full output paths
    output_tif = os.path.join(output_dir, f"{base_name}_{suffix}_density.tif")
    output_png = os.path.join(output_dir, f"{base_name}_{suffix}_density.png")

    print(f"Saving to: {output_png}")
    print(f"Overwrite allowed? {'Yes' if overwrite_cancer_type_map else 'No'}")

    # Check if file exists and we shouldn't overwrite
    if not overwrite_cancer_type_map and os.path.exists(output_png):
        image_bytes = None
        print("WE ARE IN THE DON'T OVERWRITE SECTION")
        if return_image and os.path.exists(output_png):
            with open(output_png, 'rb') as f:
                image_bytes = f.read()
        return image_bytes, output_tif, output_png 
    

    # Set matplotlib backend to Agg to avoid GUI conflicts (otherwise GUI will crash)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    

    
    # Save GeoTIFF
    save_raster_like(population_raster_path, array, output_tif)
    
    # Generate and save PNG

    if include_optimal_fraction:
        title = f"{country_code.upper()} — {' + '.join(cancer_types)} (Optimal treated: pop × prop × optimal fraction)"
    elif include_fraction:
        title = f"{country_code.upper()} — {' + '.join(cancer_types)} (Treated: pop × prop × fraction)"
    else:
        title = f"{country_code.upper()} — {' + '.join(cancer_types)} (Incidence: pop × prop)"

    # Plot the map
    with rasterio.open(population_raster_path) as src:
        bounds = src.bounds
    
    plot_data = array.copy()
    in_country_mask = population > 0
    valid_mask = (plot_data > 0) & in_country_mask

    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 8))
    # Prepare masked array for plotting
    plot_data_masked = np.full_like(plot_data, np.nan, dtype=np.float32)
    plot_data_masked[valid_mask] = plot_data[valid_mask]

    # Replace values <1 with a small dummy (e.g. 0.5), so they fall into lowest bin
    norm_data = plot_data_masked.copy()
    norm_data[(norm_data < 1) & (~np.isnan(norm_data))] = 0.5  # <-- this is key

    # Set log color scale
    vmin = 1
    vmax = global_vmax if global_vmax is not None else np.nanmax(plot_data_masked)

    # Define custom colormap
    cmap = cm.get_cmap('viridis', 256)
    new_colors = cmap(np.linspace(0, 1, 256))
    dark_blue = np.array([0, 0, 139 / 255, 1.0])  # RGBA for dark blue
    new_colors[0] = dark_blue  # Replace the lowest color
    custom_cmap = ListedColormap(new_colors)

    # Plot
    im = ax.imshow(
        norm_data,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap=custom_cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cancer-type population density proxy (people/km² × proportion)")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()

    # Save to file
    plt.savefig(output_png, dpi=300)

    # Return image bytes if requested
    image_bytes = None
    if return_image:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        image_bytes = buf.getvalue()

    plt.close()
    
    return image_bytes, output_tif, output_png



# Example usage of function:
# generate_cancer_type_map(
#     'GBR',
#     'Bladder')

