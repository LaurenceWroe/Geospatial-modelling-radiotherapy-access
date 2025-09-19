from pathlib import Path
from pycountry import countries
import geopandas as gpd
import h3.api.basic_int as h3  # h3-py library


def resample_h3_population(country_name,
                           input_dir="H3_zipped_pop_density_maps",
                           output_dir="H3_resampled",
                           target_resolution=6,
                           overwrite=False):
    """
    Resample Kontur H3 population data to a coarser H3 resolution.

    Parameters
    ----------
    country_name : str
        Human-readable country name (e.g., "United Kingdom").
    input_dir : str | Path
        Directory where the downloaded `.gpkg.gz` file is stored.
    output_dir : str | Path
        Directory to save resampled results.
    target_resolution : int
        Desired H3 resolution (0=coarse, 15=fine). Must be lower (coarser) than input.
    overwrite : bool
        If False and file already exists, skip processing.

    Returns
    -------
    dict
        {
          "success": bool,
          "message": str,
          "output_file": str | None,
          "original_resolution": int,
          "target_resolution": int,
          "original_population": float,
          "resampled_population": float
        }
    """
    try:
        # Resolve input file
        country = countries.lookup(country_name)
        alpha_2 = country.alpha_2
        input_file = Path(input_dir) / f"{alpha_2}_H3_population_density_map.gpkg.gz"

        if not input_file.exists():
            return {
                "success": False,
                "message": f"Input file not found: {input_file}",
                "output_file": None,
                "original_resolution": None,
                "target_resolution": target_resolution,
                "original_population": None,
                "resampled_population": None,
            }

        # Load GeoDataFrame
        gdf = gpd.read_file(f"zip://{input_file}")
        if "h3" not in gdf.columns or "population" not in gdf.columns:
            return {
                "success": False,
                "message": f"Missing required columns (h3, population) in {input_file}",
                "output_file": None,
                "original_resolution": None,
                "target_resolution": target_resolution,
                "original_population": None,
                "resampled_population": None,
            }

        # Infer resolution of input
        h3_index_sample = gdf["h3"].iloc[0]
        original_resolution = h3.h3_get_resolution(h3_index_sample)

        if target_resolution >= original_resolution:
            return {
                "success": False,
                "message": f"Target resolution {target_resolution} must be coarser "
                           f"(lower) than input resolution {original_resolution}",
                "output_file": None,
                "original_resolution": original_resolution,
                "target_resolution": target_resolution,
                "original_population": float(gdf["population"].sum()),
                "resampled_population": None,
            }

        # Prepare output path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir) / f"{alpha_2}_H3_pop_res{target_resolution}.gpkg"

        if output_file.exists() and not overwrite:
            return {
                "success": True,
                "message": f"File already exists: {output_file}",
                "output_file": str(output_file),
                "original_resolution": original_resolution,
                "target_resolution": target_resolution,
                "original_population": float(gdf["population"].sum()),
                "resampled_population": None,
            }

        # Aggregate to coarser resolution
        gdf["parent_h3"] = gdf["h3"].apply(
            lambda h: h3.h3_to_parent(h, target_resolution)
        )
        df_agg = gdf.groupby("parent_h3")["population"].sum().reset_index()

        # Convert back to GeoDataFrame with polygons
        df_agg["geometry"] = df_agg["parent_h3"].apply(h3.h3_to_geo_boundary, geo_json=True)
        gdf_out = gpd.GeoDataFrame(
            df_agg, geometry=gpd.GeoSeries.from_polygons(df_agg["geometry"]), crs="EPSG:4326"
        )

        # Save
        gdf_out.to_file(output_file, driver="GPKG")

        return {
            "success": True,
            "message": f"Resampled to resolution {target_resolution} and saved to {output_file}",
            "output_file": str(output_file),
            "original_resolution": original_resolution,
            "target_resolution": target_resolution,
            "original_population": float(gdf["population"].sum()),
            "resampled_population": float(df_agg["population"].sum()),
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Resampling failed: {str(e)}",
            "output_file": None,
            "original_resolution": None,
            "target_resolution": target_resolution,
            "original_population": None,
            "resampled_population": None,
        }

