#!/usr/bin/env python3
"""
Downloads and saves WorldPop data for supported countries.
"""
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# WorldPop data configuration - Updated with actual URLs
COUNTRY_DATA = {
    'GBR': {
        'name': 'United Kingdom',
        'url': 'https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/GBR/gbr_ppp_2020_UNadj.tif'
    },
    'NGA': {
        'name': 'Nigeria',
        'url': 'https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/NGA/nga_ppp_2020_UNadj.tif'
    },
    'USA': {
        'name': 'United States',
        'url': 'https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/USA/usa_ppp_2020_UNadj.tif'
    }
}

def download_file(url: str, save_path: Path) -> None:
    """Downloads a file from URL and saves it locally"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Downloaded: {save_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise

def main():
    """Main download function"""
    raw_data_dir = Path("actual_data/raw_from_worldpop") 
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting WorldPop data download...")
    
    for country_code, data in COUNTRY_DATA.items():
        output_file = raw_data_dir / f"{country_code.lower()}_population.tif"
        
        if not output_file.exists():
            logger.info(f"Downloading {data['name']} data...")
            download_file(data['url'], output_file)
        else:
            logger.info(f"{data['name']} data already exists at {output_file}")

if __name__ == "__main__":
    main()