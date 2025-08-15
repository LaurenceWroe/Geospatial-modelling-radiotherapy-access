# Function inputs a country name and downloads the WorldPop population TIF file for that country.

import requests # Makes HTTP requests to download files for big files more efficiently
import os  # Provides a way to interact with the operating system, such as creating directories
from pycountry import countries # Provides access to ISO country data, allowing us to look up country codes by name, e.g. 'United Kingdom' to 'GBR'

def download_worldpop_tif(country_name, output_dir="actual_data/raw_from_worldpop"):
    """
    Downloads a WorldPop population TIF file for the specified country.

    Args:
        country_name (str): Name of the country (e.g., 'United Kingdom').
        output_dir (str): Directory to save the downloaded TIF file. Default is 'actual_data/raw_from_worldpop'.
    """
    # Look up the ISO Alpha-3 country code
    try:
        country = countries.lookup(country_name)
        country_code = country.alpha_3.lower()
    except LookupError:
        raise ValueError(f"Country '{country_name}' not found in the country database.")

    # Construct the download URL
    url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/{country_code.upper()}/{country_code}_ppp_2020_UNadj.tif"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
    output_file = os.path.join(output_dir, f"{country_code}_ppp_2020_UNadj.tif")

    # Download the file
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded '{output_file}'")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download the file: {e}")

# Example usage:
# download_worldpop_tif("United Kingdom")