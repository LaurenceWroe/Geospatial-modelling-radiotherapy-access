#!/usr/bin/env python3
"""
Generate a country map for a selected cancer type by multiplying the
population density raster with the by the cancer-type proportion from an Excel file.

Default Excel path (can be overridden):
  /Users/sophiamartin/Desktop/src/b_cancer_incidence/cancer_type_radiotherapy.xlsx

Typical GUI flow:
  1) Run resample_population.py to create the country population raster at a target resolution
  2) Run this script to produce a cancer-type-specific density map (and GeoTIFF)
"""

import argparse
from pathlib import Path
from typing import Optional, Dict
from unittest import result

import numpy as np
import pandas as pd
import rasterio
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


DEFAULT_EXCEL_PATH = "/Users/sophiamartin/Desktop/src/b_cancer_incidence/cancer_type_radiotherapy.xlsx"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_cancer_fractions(excel_path: str) -> Dict[str, float]:
    """
    Load cancer type proportions from the Excel file.

    Returns a mapping: lowercased cancer type -> proportion (float)

    Expected columns (case/space tolerant):
      - "Cancer Type" (or with trailing spaces)
      - "proportion of cases"
    """
    df = pd.read_excel(excel_path, sheet_name=0)
    df = _normalize_columns(df)

    # Find best matching columns
    col_map: Dict[str, Optional[str]] = {
        "type": None,
        "prop": None,
    }
    for c in df.columns:
        lc = c.strip().lower()
        if col_map["type"] is None and (lc == "cancer type" or lc.startswith("cancer type")):
            col_map["type"] = c
        if col_map["prop"] is None and (lc == "proportion of cases" or "proportion" in lc):
            col_map["prop"] = c

    if col_map["type"] is None or col_map["prop"] is None:
        raise ValueError(
            f"Could not find required columns in {excel_path}. Found columns: {list(df.columns)}"
        )

    # Build mapping
    mapping: Dict[str, float] = {}
    for _, row in df.iterrows():
        ct = str(row[col_map["type"]]).strip()
        if ct.lower() in ("nan", "", "none"):
            continue
        try:
            prop = float(row[col_map["prop"]])
        except Exception:
            continue
        mapping[ct.lower()] = prop
    if not mapping:
        raise ValueError(f"No valid cancer-type proportions found in {excel_path}")
    return mapping


def prompt_for_cancer_type(fractions: Dict[str, float]) -> str:
    """
    Interactively prompt the user to choose a cancer type.

    Returns the selected cancer type key (lowercased) present in fractions.
    """
    options = sorted(fractions.keys())
    print("\nAvailable cancer types:")
    for idx, key in enumerate(options, start=1):
        print(f"  {idx}. {key} ")

    while True:
        choice = input("\nSelect by number or type a name: ").strip()
        # Numeric selection
        if choice.isdigit():
            i = int(choice)
            if 1 <= i <= len(options):
                return options[i - 1]
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
                continue
        # Text selection: direct or substring match
        key = choice.lower()
        if key in fractions:
            return key
        matches = [k for k in options if key in k]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print("Ambiguous match. Did you mean one of:")
            for m in matches:
                print(f"  - {m}")
        else:
            print("No match found. Please try again.")


def multiply_population_by_fraction(
    population_raster_path: str,
    fraction: float,
) -> np.ndarray:
    """Load population raster and return population * fraction array.

    Notes:
    - Any nodata or non-positive population values are treated as 0 for plotting
    - Returns the calculated array; raster metadata should be reused by caller
    """
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
    # Clean population values
    population = np.where(population > 0, population, 0)
    result = population.astype(np.float64) * float(fraction)
    return population,result


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


def plot_cancer_type_map(
    population: np.ndarray,
    array: np.ndarray,
    template_raster_path: str,
    output_png_path: str,
    title: str,
    dpi: int = 300,
) -> None:
    """
    Plot the cancer-type density array using the template raster's bounds.
    Uses log color scaling with a small floor to improve visibility.
    """
    with rasterio.open(template_raster_path) as src:
        bounds = src.bounds

    # Prepare data for plotting
    plot_data = array.copy()
    # Set a minimum floor for log-scale visualization (ignore zeros)
    #positive_mask = plot_data > 0
    #min_positive = plot_data[positive_mask].min() if np.any(positive_mask) else 1e-6
    #floor_value = max(min_positive, 1e-6)
    #plot_data = np.where(plot_data > 0, plot_data, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))
    #im = ax.imshow(
        #plot_data,
        #extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        #origin="upper",
        #cmap="viridis",
        #norm=LogNorm(vmin=floor_value, vmax=np.nanmax(plot_data) if np.isfinite(np.nanmax(plot_data)) else 1)
    #)

# Mask to find positive values (required for LogNorm)
    positive_mask = plot_data > 0

    if np.any(positive_mask):
        vmin = np.min(plot_data[positive_mask])
        vmax = np.max(plot_data[positive_mask])
    # Make sure vmin and vmax are valid for LogNorm
        if vmin <= 0 or not np.isfinite(vmin):
            vmin = 1e-6
        if vmax <= vmin or not np.isfinite(vmax):
            vmax = vmin * 10
    else:
    # No positive values in plot_data: fallback safe defaults
        vmin = 1e-6
        vmax = 1

# Use np.where to mask zero or negative values (to avoid log(0))
    plot_data_masked = np.where(plot_data > 0, plot_data, np.nan)

    im = ax.imshow(
        plot_data_masked,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cancer-type population density proxy (people/km² × proportion)")

    cbar.set_label("Cancer-type population density proxy (people/km² × proportion)")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=dpi)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create a cancer-type map by multiplying population density by a cancer-type proportion."
    )
    parser.add_argument(
        "country_code",
        help="ISO country code (e.g., NGA, GBR, USA) used to infer default paths"
    )
    parser.add_argument(
        "cancer_type",
        nargs='?',
        help="Cancer type name as listed in the Excel (case-insensitive). If omitted, you will be prompted."
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default=DEFAULT_EXCEL_PATH,
        help=f"Path to Excel with cancer-type proportions (default: {DEFAULT_EXCEL_PATH})"
    )
    parser.add_argument(
        "--population-raster",
        type=str,
        default=None,
        help=(
            "Path to population raster (GeoTIFF). If omitted, uses actual_data/resampled/"
            "{code}_{resolution}km.tif based on --resolution."
        )
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1,
        help="Resolution in km used to infer default population raster and output file names (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: b_cancer_incidence/cancer_type_maps)"
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="Optional custom base name for outputs (without extension)."
    )
    parser.add_argument(
        "--list-cancers",
        action="store_true",
        help="List available cancer types from the Excel and exit"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive prompt for cancer type selection"
    )

    args = parser.parse_args()

    # Load proportions
    fractions = load_cancer_fractions(args.excel_path)

    if args.list_cancers:
        print("Available cancer types:")
        for name in sorted(fractions.keys()):
            print(f"  - {name}")
        return

    # Determine cancer type (CLI or interactive)
    if args.cancer_type and not args.interactive:
        cancer_key = args.cancer_type.strip().lower()
        if cancer_key not in fractions:
            # Try a more forgiving match
            matches = [k for k in fractions.keys() if cancer_key in k]
            if len(matches) == 1:
                cancer_key = matches[0]
            else:
                raise ValueError(
                    f"Cancer type '{args.cancer_type}' not found. Use --list-cancers to see available types or use --interactive."
                )
    else:
        cancer_key = prompt_for_cancer_type(fractions)
    fraction = fractions[cancer_key]

    # Resolve default paths
    base_dir = Path(__file__).resolve().parents[1]
    #data_dir = base_dir 
    if args.output_dir is None:
        output_dir = base_dir / "b_cancer_incidence" / "cancer_type_maps"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.population_raster is None:
        pop_default = base_dir /"actual_data"/ "resampled" / f"{args.country_code.lower()}_{args.resolution}.0km.tif"
        population_raster_path = str(pop_default)
    else:
        population_raster_path = args.population_raster

    if args.basename is None:
        safe_cancer = cancer_key.replace(" ", "_")
        base_name = f"{args.country_code.lower()}_{safe_cancer}_{args.resolution}km"
    else:
        base_name = args.basename

    output_tif = str(output_dir / f"{base_name}_cancer_type_density.tif")
    output_png = str(output_dir / f"{base_name}_cancer_type_density.png")

    # Calculation
    print(f"Loading population raster: {population_raster_path}")
    print(f"Using cancer-type proportion: {fraction} ({cancer_key})")
    population, array = multiply_population_by_fraction(population_raster_path, fraction)

    # Save GeoTIFF
    print(f"Saving GeoTIFF: {output_tif}")
    save_raster_like(population_raster_path, array, output_tif)

    # Save PNG map
    title = f"{args.country_code.upper()} — {args.cancer_type} (population × proportion)"
    print(f"Saving PNG map: {output_png}")
    plot_cancer_type_map(population,array, population_raster_path, output_png, title)

    print("Done.")


if __name__ == "__main__":
    main()


