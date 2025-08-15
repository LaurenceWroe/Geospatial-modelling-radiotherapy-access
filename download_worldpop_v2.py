import os
import requests
from pycountry import countries

def download_worldpop_v2(country_name, output_dir="actual_data/raw_from_worldpop", progress_callback=None, overwrite=False):
    """
    Downloads WorldPop population TIF file with proper compression handling.
    
    Args:
        country_name: Name of the country (e.g., "United Kingdom")
        output_dir: Directory to save the downloaded file
        progress_callback: Function to report download progress
        overwrite: Whether to overwrite existing files
        
    Returns:
        tuple: (success: bool, message: str)
    """

    try:
        country = countries.lookup(country_name)
        country_code = country.alpha_3.lower()
    except LookupError:
        return False, f"Country '{country_name}' not found."

    url = f"https://data.worldpop.org/GIS/Population_Density/Global_2000_2020_1km_UNadj/2020/{country_code.upper()}/{country_code}_pd_2020_1km_UNadj.tif"
    #url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/{country_code.upper()}/{country_code}_ppp_2020_1km_Aggregated.tif"
    # this old url was for the 2020 population data, but the new one is for population density

    output_file = os.path.join(output_dir, f"{country_code}_raw.tif")
    
    # Check if file exists and we shouldn't overwrite
    if not overwrite and os.path.exists(output_file):
        return True, f"File already exists at:\n{output_file}"

    try:
        # Use session to enable connection pooling and compression
        with requests.Session() as session:
            session.headers.update({
                'Accept-Encoding': 'gzip, deflate',
                'User-Agent': 'Mozilla/5.0'
            })
            
            # First get the headers to check file size
            with session.head(url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                # Verify expected file size (3-10MB typical)
                if total_size > 50 * 1024 * 1024:  # 50MB threshold
                    return False, f"Unexpected large file size ({total_size/1024/1024:.1f}MB). Aborting download."

            # Stream the download with compression
            with session.get(url, stream=True) as response:
                response.raise_for_status()
                os.makedirs(output_dir, exist_ok=True)
                
                # Use chunked download with progress
                downloaded = 0
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback and total_size > 0:
                                progress = int(100 * downloaded / total_size)
                                progress_callback(progress)
            
        # Verify downloaded file size
        actual_size = os.path.getsize(output_file) / (1024 * 1024)
        if actual_size > 50:  # MB
            os.remove(output_file)
            return False, f"Downloaded file too large ({actual_size:.1f}MB). Removed suspicious file."
        
        return True, f"Successfully downloaded to:\n{output_file} ({actual_size:.1f}MB)"
    
    except requests.exceptions.RequestException as e:
        # Clean up partial download on failure
        if os.path.exists(output_file):
            os.remove(output_file)
        return False, f"Download failed: {str(e)}"
    except Exception as e:
        return False, f"An error occurred: {str(e)}"
    

# Example usage:
# print("Download WorldPop TIF file for a specific country")
# result = download_worldpop_v2("United States of America")
# print(result)