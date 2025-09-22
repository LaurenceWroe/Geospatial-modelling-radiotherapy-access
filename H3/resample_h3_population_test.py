"""
Resample Kontur H3 Population Data: 

This script provides a utility to resample population data from the
Kontur Population dataset across different H3 resolutions, supporting
both aggregation to coarser hexagons and subdivision to finer ones.

Workflow:
1. Load the country-specific H3 population dataset (GeoPackage .gpkg.gz).
2. Detect the original H3 resolution of the data.
3. Compare target vs. original resolution:
   - If target < original: aggregate population into coarser parent hexagons.
   - If target > original: subdivide each hexagon into child hexagons and
     distribute its population evenly across them.
4. Reconstruct geometries for the resulting hexagons.
5. Save the resampled data as a new GeoPackage file.

Inputs:
- country_name : str
    Human-readable country name (e.g., "United Kingdom").
- input_dir : str | Path
    Directory containing the downloaded `.gpkg.gz` Kontur population file.
- output_dir : str | Path
    Directory where the resampled GeoPackage will be saved.
- target_resolution : int
    Desired H3 resolution (0 = very coarse, 15 = very fine).
- overwrite : bool
    If False and the output file already exists, skip processing.

Returns:
dict
    A dictionary with keys:
      - success (bool): whether resampling succeeded
      - message (str): status or error message
      - output_file (str | None): path to the saved GeoPackage, if created
      - original_resolution (int): detected H3 resolution of input
      - target_resolution (int): requested H3 resolution
      - original_population (float): sum of populations in input
      - resampled_population (float): sum of populations in output
"""


import h3
from pycountry import countries
import h3.api.basic_int as h3  # h3-py library
from H3.download_h3_test import load_h3_population
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional
from matplotlib import pyplot as plt, cm
from matplotlib.colors import LogNorm, ListedColormap
from io import BytesIO

from H3.download_h3_test import load_h3_population


def resample_h3_population(country_name,
                           input_dir="H3_zipped_pop_density_maps",
                           output_dir="H3_resampled",
                           target_resolution=6,
                           overwrite=False):
    """
    Resample Kontur H3 population data to a different H3 resolution
    (coarser or finer).

    If target resolution < original: aggregate to coarser hexes.
    If target resolution > original: subdivide into finer hexes,
    distributing population evenly among children.

    Returns a dict with metadata and result status.
    """
    try:
        # Resolve input file
        country = countries.lookup(country_name)
        alpha_2 = country.alpha_2
        input_file = Path(input_dir) / f"{alpha_2}_H3_population_density_map.gpkg.gz"

        if not input_file.exists():
            return {"success": False, "message": f"Input file not found: {input_file}"}

        # Load GeoDataFrame
        gdf = gpd.read_file(f"zip://{input_file}")
        if "h3" not in gdf.columns or "population" not in gdf.columns:
            return {"success": False, "message": "Missing required columns (h3, population)"}

        # Infer original resolution
        h3_index_sample = gdf["h3"].iloc[0]
        original_resolution = h3.h3_get_resolution(h3_index_sample)

        if target_resolution == original_resolution:
            return {"success": True,
                    "message": f"Target resolution is the same ({target_resolution}). No resampling needed.",
                    "output_file": None,
                    "original_resolution": original_resolution,
                    "target_resolution": target_resolution,
                    "original_population": float(gdf["population"].sum()),
                    "resampled_population": float(gdf["population"].sum())}

        # Prepare output path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir) / f"{alpha_2}_H3_pop_res{target_resolution}.gpkg"

        if output_file.exists() and not overwrite:
            return {"success": True, "message": f"File already exists: {output_file}", "output_file": str(output_file)}

        # --- Case 1: Aggregate to coarser resolution ---
        if target_resolution < original_resolution:
            gdf["parent_h3"] = gdf["h3"].apply(lambda h: h3.h3_to_parent(h, target_resolution))
            df_agg = gdf.groupby("parent_h3")["population"].sum().reset_index()
            df_agg["geometry"] = df_agg["parent_h3"].apply(h3.h3_to_geo_boundary, geo_json=True)
            gdf_out = gpd.GeoDataFrame(df_agg,
                                       geometry=gpd.GeoSeries.from_polygons(df_agg["geometry"]),
                                       crs="EPSG:4326")

        # --- Case 2: Subdivide to finer resolution ---
        else:
            records = []
            for _, row in gdf.iterrows():
                children = h3.h3_to_children(row["h3"], target_resolution)
                if not children:
                    continue
                # Distribute population evenly among children
                pop_per_child = row["population"] / len(children)
                for child in children:
                    records.append({"h3": child, "population": pop_per_child})

            df_sub = pd.DataFrame(records)
            df_agg = df_sub.groupby("h3")["population"].sum().reset_index()
            df_agg["geometry"] = df_agg["h3"].apply(h3.h3_to_geo_boundary, geo_json=True)
            gdf_out = gpd.GeoDataFrame(df_agg,
                                       geometry=gpd.GeoSeries.from_polygons(df_agg["geometry"]),
                                       crs="EPSG:4326")

        # Save to GeoPackage
        gdf_out.to_file(output_file, driver="GPKG")

        return {
            "success": True,
            "message": f"Resampled from res {original_resolution} to {target_resolution}",
            "output_file": str(output_file),
            "original_resolution": original_resolution,
            "target_resolution": target_resolution,
            "original_population": float(gdf["population"].sum()),
            "resampled_population": float(gdf_out["population"].sum()),
        }

    except Exception as e:
        return {"success": False, "message": f"Resampling failed: {str(e)}"}


def generate_population_density_map_only_h3(
    country_name: str,
    input_dir="H3_zipped_pop_density_maps",
    output_dir="h3_population_maps",
    resolution: int = None,  # optional placeholder
    overwrite_existing: bool = False,
    return_image: bool = True,
    h3_gpkg_path: Path = None
):
    """
    Generate a polygon-only population density map (GeoPackage + PNG preview bytes).
    
    Returns:
        dict: {"gpkg_path": Path to saved GeoPackage, "image_bytes": PNG bytes}
    """
    # Determine gpkg path
    if h3_gpkg_path is None:
        gdf, df = load_h3_population(country_name, input_dir=input_dir)
    else:
        gdf, df = load_h3_population(country_name, input_dir=str(h3_gpkg_path.parent))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_gpkg = Path(output_dir) / f"{country_name}_population_density.gpkg"
    output_png = Path(output_dir) / f"{country_name}_population_density.png"

    # Save GeoPackage
    if not output_gpkg.exists() or overwrite_existing:
        gdf.to_file(output_gpkg, driver="GPKG")

    image_bytes = None
    if return_image:
        # Prepare population values
        pop = gdf["population"].to_numpy()
        pop = np.where(pop > 0, pop, np.nan)

        # Custom colormap (viridis with dark blue for zero)
        cmap = cm.get_cmap("viridis", 256)
        new_colors = cmap(np.linspace(0, 1, 256))
        dark_blue = np.array([0, 0, 139 / 255, 1.0])
        new_colors[0] = dark_blue
        custom_cmap = ListedColormap(new_colors)
        custom_cmap.set_under(dark_blue)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(column="population", cmap=custom_cmap,
                 norm=LogNorm(vmin=1, vmax=np.nanmax(pop)),
                 linewidth=0, ax=ax)

        cbar = plt.colorbar(cm.ScalarMappable(norm=LogNorm(vmin=1, vmax=np.nanmax(pop)),
                                              cmap=custom_cmap), ax=ax)
        cbar.set_label("Population density (people per hex)")
        ax.set_title(f"{country_name} — Population Density (H3)")
        ax.axis("off")
        plt.tight_layout()

        # Save PNG bytes to memory
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        image_bytes = buf.getvalue()

    return {"gpkg_path": output_gpkg, "image_bytes": image_bytes}



