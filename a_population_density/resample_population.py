# This function inputs a country name and resolution and resamples the WorldPop population TIF file for that country.
# It saves the resampled data into actual_data/resampled
# It is called by the GUI_show_population_v2.py file

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
    Resamples population data to specified resolution using rioxarray
    
    Args:
        country_name: Name of the country (e.g., "United Kingdom")
        resolution_km: Target resolution in kilometers
        input_dir: Directory containing raw WorldPop files
        output_dir: Directory to save resampled files
        
    Returns:
        dict: {
            'success': bool,
            'message': str,
            'output_path': str,
            'original_population': float,
            'resampled_population': float
        }
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
            return True, f"File already exists at:\n{output_file}"


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
    
# example usage
# result = resample_population("United Kingdom", 0.5)
# print(result)