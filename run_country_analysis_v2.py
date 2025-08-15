
import requests # Makes HTTP requests to download files for big files more efficiently
import os  # Provides a way to interact with the operating system, such as creating directories
from pycountry import countries # Provides access to ISO country data, allowing us to look up country codes by name, e.g. 'United Kingdom' to 'GBR'


download_worldpop_tif(country_name, output_dir="actual_data/raw_from_worldpop")