#!/usr/bin/env python3
"""
Generate a country map for a selected cancer type by multiplying the
population density raster with the by the cancer-type proportion from an Excel file.
THIS IS THE NEW FUNCTION THAT WORKS WITH THE GUI BETTER AND IS NOT DEPENDENT ON COMMAND LINE INPUTS
"""

import os
from pathlib import Path
from typing import Optional, Dict, Tuple
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


def load_cancer_fractions(excel_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Load cancer type proportions from the Excel file.
    Returns a mapping: lowercased cancer type -> (proportion, fraction)
    """
    df = pd.read_excel(excel_path, sheet_name=0)
    df = _normalize_columns(df)

    # Find best matching columns
    col_map = {
        "type": None,
        "prop": None,
        "frac": None,
    }
    for c in df.columns:
        lc = c.strip().lower()
        if col_map["type"] is None and (lc == "cancer type" or lc.startswith("cancer type")):
            col_map["type"] = c
        if col_map["prop"] is None and (lc == "proportion of cases" or "proportion" in lc):
            col_map["prop"] = c
        if col_map["frac"] is None and (lc == "fraction" or "fraction" in lc):
            col_map["frac"] = c

    if col_map["type"] is None or col_map["prop"] is None or col_map["frac"] is None:
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
        except Exception:
            continue
        mapping[ct.lower()] = (prop, frac)

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

    # Clean population values (remove nodata / negatives)
    population = np.where(population > 0, population, 0)

    # Multiply by proportion and fraction
    result = population.astype(np.float64) * float(proportion) * float(fraction)

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
        data = np.where(np.isfinite(array), array, nodata_value).astype(np.float32)
        dst.write(data, 1)

def multiply_population_by_multiplier(
    population_raster_path: str,
    multiplier: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load population raster and return population * multiplier array."""
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)

    population = np.where(population > 0, population, 0)
    result = population.astype(np.float64) * float(multiplier)

    return population, result


def generate_cancer_type_map(
    country_code: str,
    cancer_type: str,
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
) -> Tuple[Optional[bytes], str, str]:
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
    
    # Validate cancer type
    cancer_key = cancer_type.strip().lower()
    if cancer_key not in fractions:
        # Try substring match
        matches = [k for k in fractions.keys() if cancer_key in k]
        if len(matches) == 1:
            cancer_key = matches[0]
        else:
            raise ValueError(f"Cancer type '{cancer_type}' not found. Available types: {sorted(fractions.keys())}")
    
    proportion, fraction_val = fractions[cancer_key]

    if include_fraction:
        population, array = multiply_population_by_multiplier(population_raster_path, proportion * fraction_val)
    else:
        population, array = multiply_population_by_multiplier(population_raster_path, proportion)

    # Resolve default paths
    base_dir = Path(__file__).resolve().parents[1]
    
    if output_dir is None:
        output_dir = base_dir / "b_cancer_incidence" / "cancer_type_maps"
        output_dir /= "treated_maps" if include_fraction else "incidence_maps"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    if population_raster_path is None:
        resolution_str = f"{resolution:.1f}"
        pop_default = base_dir / "a_population_density" / "resampled" / f"{country_code.lower()}_{resolution_str}km.tif"
        population_raster_path = str(pop_default)
    
    if basename is None:
        safe_cancer = cancer_key.replace(" ", "_")
        base_name = f"{country_code.lower()}_{safe_cancer.lower()}_{resolution}km"
    else:
        base_name = basename
    
    suffix = "treated" if include_fraction else "incidence"
    output_tif = str(output_dir / f"{base_name}_{suffix}_density.tif")
    output_png = str(output_dir / f"{base_name}_{suffix}_density.png")

    print(f"Saving to: {output_png}")
    print(f"Overwrite allowed? {'Yes' if overwrite_cancer_type_map else 'No'}")

    # Check if file exists and we shouldn't overwrite
    if not overwrite_cancer_type_map and os.path.exists(output_png): # if overwrite is False and file exists, we don't to overwrite simply to plot
        # Load the existing image and return it
        image_bytes = None
        print('WE ARE IN THE DON"T OVERWRITE SECTION')
        if return_image and os.path.exists(output_png):
            with open(output_png, 'rb') as f:
                image_bytes = f.read()
        return image_bytes, output_tif, output_png  # Return 3 values
    

    # Set matplotlib backend to Agg to avoid GUI conflicts (otherwise GUI will crash)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    

    
    # Save GeoTIFF
    save_raster_like(population_raster_path, array, output_tif)
    
    # Generate and save PNG
    title = f"{country_code.upper()} — {cancer_type} (population × proportion × fraction of cases treated by radiotherapy)"
    
    # Plot the map
    with rasterio.open(population_raster_path) as src:
        bounds = src.bounds
    
    #plot_data = array.copy()
    #positive_mask = plot_data > 0

    #if np.any(positive_mask):
        #local_vmin = np.min(plot_data[positive_mask])
        #local_vmax = np.max(plot_data[positive_mask])
        #vmin = global_vmin if global_vmin is not None else max(local_vmin, 1e-6)
        #vmax = global_vmax if global_vmax is not None else max(local_vmax, vmin * 10)
    #else:
        #vmin = global_vmin if global_vmin is not None else 1e-6
        #vmax = global_vmax if global_vmax is not None else 1
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

