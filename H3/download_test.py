#imports 
import pydeck as pdk
import pandas as pd
import numpy as np
import h3
import io
import requests
import geodatasets as gds
import geopandas as gpd
import numpy as np
import pycountry
from geopy.geocoders import Nominatim
from pathlib import Path
import gzip, shutil

"""
Downloads and extracts a country-specific H3 population density geopackage (GPKG) from the Kontur Population dataset.

Inputs:
country_name --> Full name of the country (must be recognized by pycountry).
output_dir --> Directory to save the downloaded files (default "H3_pop_density_maps").
progress_callback --> Optional function to report download progress (e.g., update a progress bar in a GUI).
overwrite_download --> If True, redownload even if the file already exists.

Process:
Uses pycountry to look up the country’s ISO Alpha-2 code.
Builds the download URL for the country’s compressed geopackage (.gpkg.gz).
Streams the file from the Kontur S3 bucket, writing it to disk in chunks.
If a progress_callback is provided, it reports progress as a percentage.
Decompresses the .gz file into a .gpkg using unzip_gpkg.
Deletes the temporary compressed file after extraction.

Outputs (tuple):
Success flag (bool) --> Whether the operation succeeded.
File path (Path) --> Path to the extracted .gpkg file (if successful).
Message (str) --> Explanation of the result (success message, error details, etc.).

"""


def unzip_gpkg(gz_path):
    gpkg_path = gz_path.with_suffix("")

    # Unzip
    with gzip.open(gz_path, 'rb') as f_in, open(gpkg_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return gpkg_path

def download_H3_population_density_gpkg(
        country_name, 
        output_dir="H3_pop_density_maps", 
        progress_callback= None, 
        overwrite_download = False):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        selected_country = pycountry.countries.get(name = country_name)
        selected_country_alpha_2 = selected_country.alpha_2
    except LookupError:
        return False, f"Country, {country_name}, not found"
    
    base_url = (
    "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
    "kontur_population_{country_alpha_2}_20231101.gpkg.gz"
    )

    target_url = base_url.format(country_alpha_2=selected_country_alpha_2)
    # Path of downloaded zipped file
    gz_path = Path(output_dir)/ f"{selected_country_alpha_2}_H3_population_density_map.gpkg.gz"
    # Path of final unzipped file
    output_file = gz_path.with_suffix("")


    # Check if file exists and we shouldn't overwrite
    if not overwrite_download and Path(output_file).exists():
        return True, output_file, f"File already exists at:\n{output_file}"

    try:
        # Use session to enable connection pooling and compression
        with requests.session() as session:

            with session.get(target_url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("Content-Length",0))

                with open(gz_path, mode="wb") as file:
                    
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size= 10 * 1024):
                        file.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size>0:
                            progress = int(100* downloaded/total_size)
                            progress_callback(progress)
    
    except requests.exceptions.RequestException as e:
        # Handle incomplete shit
        if Path(gz_path).exists() :
            Path(gz_path).unlink(missing_ok=True)
        return False, f"Download failed, {str(e)}"
    except Exception as e:
        return False, f"Error occured: {str(e)}"
    
    gpkg_path = unzip_gpkg(gz_path)
        
    # Delete .gz file
    Path(gz_path).unlink(missing_ok=True)

    return False, gpkg_path, f"File saved to {gpkg_path}"