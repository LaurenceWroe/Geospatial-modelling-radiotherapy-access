import os
import rioxarray
import shutil
import numpy as np
import xarray as xr
from pycountry import countries
from rasterio.enums import Resampling
from pathlib import Path

def resample_population(country_name, resolution_km, input_dir="a_population_density/raw_from_worldpop", output_dir="a_population_density/resampled", overwrite_resample=False):
    """
    Resample a country's WorldPop raster to a target resolution (km) using rioxarray,
    preserving population counts via SUM aggregation, and write the result to disk.

    Parameters:
        country_name (str): Human-readable country (e.g., "United Kingdom").
        resolution_km (float | int): Target pixel size in kilometers (> 0).
        input_dir (str | Path): Directory containing the raw input TIFF named
            `{iso3_lower}_raw.tif` (e.g., "gbr_raw.tif").
        output_dir (str | Path): Directory where the resampled TIFF will be written,
            named `{iso3_lower}_{resolution_km}km.tif`.
        overwrite_resample (bool): If False and the output already exists, the function
            returns early with `success=True` and a message, without recomputing.

    Behavior & data handling:
        - Resolves `country_name` to ISO-3 code via `pycountry.countries.lookup`.
        - Converts kilometers to degrees with an approximate factor: 1° ≈ 111 km
          (no latitude-dependent correction).
        - INPUT expectations:
              Input raster path: `{input_dir}/{iso3_lower}_raw.tif`.
              WorldPop nodata is set to -9999 (explicitly enforced).
        - Cleans values before and after resampling:
              Replaces nodata with 0 and clips negatives to 0 to avoid count leakage.
        - Resampling:
              Reprojects to the same CRS with `resolution=<km/111>` degrees.
              Uses `Resampling.sum` to conserve counts across coarser pixels.
        - Special case: `resolution_km == 1.0`, simply copies the input file (no resample).
        - Outputs:
              Writes the resampled GeoTIFF to `{output_dir}/{iso3_lower}_{resolution_km}km.tif`.
              Computes and returns total population before and after resampling.
        - Creates `output_dir` if it does not exist.
        - On non-1 km runs, prints basic stats (min/max/NaN/negative counts) to stdout.

    Returns:
        dict: {
            'success': bool,                 # True on success (or if output already existed)
            'message': str,                  # Human-readable status/explanation
            'output_path': str | None,       # Path to written/existing output TIFF
            'original_population': float | None,   # Sum of cleaned input raster
            'resampled_population': float | None   # Sum of cleaned resampled raster
        }
        Notes:
            - If the output exists and `overwrite_resample` is False, returns
              success=True with `original_population` and `resampled_population` as None.
            - If the input file is missing or an exception occurs, returns success=False
              with `output_path=None` and population fields as None.

    Side effects:
        - Reads/writes TIFF files on disk; may create directories.
        - Prints simple diagnostics to stdout for non-1 km runs.

    Example:
        # Resample UK to 2 km and write to a_population_density/resampled/gbr_2km.tif
        result = resample_population("United Kingdom", 2.0)
        if result['success']:
            print("Output:", result['output_path'])
    """
    try:
        # Validate inputs
        if not isinstance(resolution_km, (int, float)) or resolution_km <= 0:
            raise ValueError("Resolution must be a positive number")

        # Get country code and bounds
        country = countries.lookup(country_name)
        country_code = country.alpha_3.lower()

        # Convert resolution from km to degrees (approximate)
        resolution_deg = resolution_km / 111
        
        # Prepare file paths
        input_file = os.path.join(input_dir, f"{country_code}_raw.tif")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{country_code}_{resolution_km}km.tif")
        
        # Check if file exists and we shouldn't overwrite
        if not overwrite_resample and os.path.exists(output_file):
            return {
                'success': True,
                'message': f"File already exists at:\n{output_file}",
                'output_path': output_file,
                'original_population': None,
                'resampled_population': None
            }


        # Check if input file exists
        if not os.path.exists(input_file):
            return {
                'success': False,
                'message': f"Input file not found: {input_file}",
                'output_path': None,
                'original_population': None,
                'resampled_population': None
            }
        
        # If resolution is 1km, just copy the original file
        if resolution_km == 1.0:
            
            output_file = os.path.join(output_dir, f"{country_code}_{resolution_km}km.tif")
            
            # Simply copy the file instead of processing
            shutil.copy2(input_file, output_file)
            
            # Calculate population from the original file
            with rioxarray.open_rasterio(input_file, masked=True) as src:
                src = src.rio.write_nodata(-9999)
                src = src.where(src >= 0, 0)
                original_pop = float(src.sum(skipna=True).values)
            
            return {
                'success': True,
                'message': f"Copied original 1km data to:\n{output_file}",
                'output_path': output_file,
                'original_population': original_pop,
                'resampled_population': original_pop
            }

        
        # For resolution not 1km, load the data and resample:
        with rioxarray.open_rasterio(input_file) as src:
            # Explicitly handle values with no data 
            src = src.rio.write_nodata(-9999)  # WorldPop standard
            src = src.where(src != src.rio.nodata, 0) # Replaces nodata with 0

            # Clip any negative values to zero
            src = src.where(src >= 0, 0)
            
            # Calculate original population
            original_pop = float(src.sum(skipna=True).values)
            
            print("Data stats before processing:")
            print(f"Min: {src.min().values}, Max: {src.max().values}")
            print(f"NaN count: {src.isnull().sum().values}")
            print(f"Negative count: {(src < 0).sum().values}")


            # Reproject to target resolution
            resampled = src.rio.reproject(
                src.rio.crs,
                resolution=resolution_deg,
                resampling=Resampling.sum, # Use sum to aggregate population counts
                nodata=np.nan
            )
            
             # Explicitly handle values with no data for the resampled data 
            resampled = resampled.rio.write_nodata(-9999)  # WorldPop standard
            resampled = resampled.where(resampled != resampled.rio.nodata, 0) # Replaces nodata with 0

            # Clip any negative values to zero for the resampled data
            resampled = resampled.where(resampled >= 0, 0)
            
            # Calculate resampled population
            resampled_pop = float(resampled.sum(skipna=True).values)
            
            # Save output
            resampled.rio.to_raster(output_file)

            delta = (resampled_pop - original_pop) / (original_pop + 1e-12) * 100
            print(f"Total pop before: {original_pop:.2f}, after: {resampled_pop:.2f} (Δ {delta:.5f}%)")
            
            return {
                'success': True,
                'message': f"Successfully resampled to {resolution_km}km",
                'output_path': output_file,
                'original_population': original_pop,
                'resampled_population': resampled_pop
            }
            
    except Exception as e:
        return {
            'success': False,
            'message': f"Resampling failed: {str(e)}",
            'output_path': None,
            'original_population': None,
            'resampled_population': None
        }