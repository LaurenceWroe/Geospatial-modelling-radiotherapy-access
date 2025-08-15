# Function inputs a country name and downloads the WorldPop population TIF file for that country.

import os
import requests
from pycountry import countries

def download_worldpop_tif(country_name, output_dir, progress_callback=None):
    """
    Downloads WorldPop population TIF file with progress reporting.
    
    Args:
        country_name: Name of the country to download
        output_dir: Directory to save the file
        progress_callback: Function to report download progress
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        country = countries.lookup(country_name)
        country_code = country.alpha_3.lower()
    except LookupError:
        return (False, f"Country '{country_name}' not found.")

    url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/{country_code.upper()}/{country_code}_ppp_2020_UNadj.tif"
    
    try:
        # Get file size first
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{country_code}_ppp_2020_UNadj.tif")
            
            downloaded = 0
            with open(output_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192*8):  # Larger chunk size
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            progress = int(100 * downloaded / total_size)
                            progress_callback.emit(progress)
            
            return (True, f"Successfully downloaded to:\n{output_file}")
    
    except requests.exceptions.RequestException as e:
        return (False, f"Download failed: {str(e)}")
    except Exception as e:
        return (False, f"An error occurred: {str(e)}")
    


    