# Function inputs a country name and downloads the WorldPop population TIF file for that country.

import os
import requests
from pycountry import countries

def download_worldpop_tif(country_name, output_dir, progress_callback=None, overwrite=False):
    """
    Downloads WorldPop population TIF file with file existence check.
    
    Args:
        country_name: Name of the country
        output_dir: Target directory
        progress_callback: Progress reporting function
        overwrite: Whether to overwrite existing files
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        country = countries.lookup(country_name)
        country_code = country.alpha_3.lower()
    except LookupError:
        return (False, f"Country '{country_name}' not found.")

    url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/{country_code.upper()}/{country_code}_ppp_2020_UNadj.tif"
    output_file = os.path.join(output_dir, f"{country_code}_ppp_2020_UNadj.tif")
    
    # Check file existence if not forcing overwrite
    if not overwrite and os.path.exists(output_file):
        return (True, f"File already exists at:\n{output_file}")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192*8):  # 64KB chunks
                    if chunk:
                        f.write(chunk)
                        if progress_callback and total_size > 0:
                            downloaded = f.tell()
                            progress = int(100 * downloaded / total_size)
                            progress_callback(progress)
            
            return (True, f"Successfully downloaded to:\n{output_file}")
    
    except requests.exceptions.RequestException as e:
        # Clean up partially downloaded file if error occurs
        if os.path.exists(output_file):
            os.remove(output_file)
        return (False, f"Download failed: {str(e)}")
    except Exception as e:
        return (False, f"An error occurred: {str(e)}")