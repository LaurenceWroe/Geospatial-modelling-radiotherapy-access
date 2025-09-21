"""
(21/09/25)  
Downloads a compressed geopackage (.gpkg.gz) containing population counts in hexagonal cells
for a given country. 

donwload_H3_population_density_zipped 
Inputs: 
country_name: the country name 
output_dir: where to save the file 
progress_callback: optional function that reports download progress 
overwrite_download: whether to redownload the file if it alreadye exists. 

Process: 
1. Gets the ISO alpha-2 country code (e.g. IT for Italy) 
2. Builds the download url from Kontur's S3 bucket 
3. Defines the output file name

Returns: 
(True/False, message) depending on success of download 

load_h3_population 
Reads the compressed geopackage file directly 

Returns: 
gdf: the full GeoDataFrame, including hexagon polygons 
df: just a Pandas DataFrame with a "h3" index and "population" (useful if
geometry isnt needed)


Once downloaded, the hex population can be loaded into python. 

"""


from pycountry import countries
import os
import h3
from pathlib import Path
import geopandas as gpd
import pandas as pd
import requests

def download_H3_population_density_zipped(country_name, output_dir="H3_zipped_pop_density_maps", progress_callback= None, overwrite_download = False):
    try:
        selected_country = countries.get(name = country_name)
        selected_country_alpha_2 = selected_country.alpha_2
    except LookupError:
        return False, f"Country, {country_name}, not found"
    
    base_url = (
    "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
    "kontur_population_{country_alpha_2}_20231101.gpkg.gz"
    )

    target_url = base_url.format(country_alpha_2=selected_country_alpha_2)

    output_file = Path(output_dir)/ f"{selected_country_alpha_2}_H3_population_density_map.gpkg.gz"


    # Check if file exists and we shouldn't overwrite
    if not overwrite_download and os.path.exists(output_file):
        return True, f"File already exists at:\n{output_file}"

    try:
        # Use session to enable connection pooling and compression
        with requests.session() as session:

            with session.get(target_url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("Content-Length",0))

                with open(output_file, mode="wb") as file:
                    
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size= 10 * 1024):
                        file.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size>0:
                            progress = int(100* downloaded/total_size)
                            progress_callback(progress)
    
    except requests.exceptions.RequestException as e:
        # Handle incomplete shit
        if Path.exists(output_file):
            Path.unlink(output_file)
        return False, f"Download failed, {str(e)}"
    except Exception as e:
        return False, f"Error occured: {str(e)}"


def load_h3_population(country_name, input_dir="H3_zipped_pop_density_maps"):
    from pycountry import countries
    selected_country = countries.get(name=country_name)
    alpha_2 = selected_country.alpha_2

    input_file = Path(input_dir) / f"{alpha_2}_H3_population_density_map.gpkg.gz"

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # GeoPandas can read gzipped gpkg directly
    gdf = gpd.read_file(f"zip://{input_file}")

    # Keep just h3 index + population if you don’t need geometry
    df = gdf[["h3", "population"]].copy()

    return gdf, df

# Example
#gdf, df = load_h3_population("Italy")
#print(df.head())