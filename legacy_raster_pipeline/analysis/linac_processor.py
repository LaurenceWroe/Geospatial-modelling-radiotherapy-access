import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from typing import Union, List, Dict, Tuple
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinacProcessor:
    """Class for processing LINAC center data and calculating accessibility."""
    
    def __init__(self):
        """Initialize the LINAC processor."""
        self.centers = None
        self.geometry = None
        
    def load_centers(self, 
                    data_path: Union[str, Path], 
                    lat_col: str = 'latitude',
                    lon_col: str = 'longitude',
                    name_col: str = 'name') -> None:
        """
        Load LINAC center data from a CSV file.
        
        Args:
            data_path: Path to the CSV file containing center data
            lat_col: Name of the latitude column
            lon_col: Name of the longitude column
            name_col: Name of the center name column
        """
        try:
            # Read the CSV file
            df = pd.read_csv(data_path)
            
            # Create GeoDataFrame
            geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
            self.centers = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            
            logger.info(f"Successfully loaded {len(self.centers)} LINAC centers")
            
        except Exception as e:
            logger.error(f"Error loading LINAC centers: {e}")
            raise
            
    def add_center(self, 
                  name: str, 
                  latitude: float, 
                  longitude: float, 
                  additional_data: Dict = None) -> None:
        """
        Add a new LINAC center to the dataset.
        
        Args:
            name: Name of the center
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            additional_data: Dictionary of additional center data
        """
        if self.centers is None:
            # Initialize with empty GeoDataFrame if no centers exist
            self.centers = gpd.GeoDataFrame(
                columns=['name', 'geometry'],
                geometry='geometry',
                crs="EPSG:4326"
            )
            
        # Create new center data
        new_center = {
            'name': name,
            'geometry': Point(longitude, latitude)
        }
        
        # Add any additional data
        if additional_data:
            new_center.update(additional_data)
            
        # Add to centers
        self.centers = pd.concat([
            self.centers,
            gpd.GeoDataFrame([new_center], crs="EPSG:4326")
        ], ignore_index=True)
        
        logger.info(f"Added new LINAC center: {name}")
        
    def calculate_accessibility(self,
                              population_data: np.ndarray,
                              transform: tuple,
                              decay_rate: float = 0.001,  # 0.1% per km
                              max_distance: float = 100.0) -> np.ndarray:
        """
        Calculate accessibility to radiotherapy treatment.
        
        Args:
            population_data: 2D array of population density
            transform: Affine transform for the population data
            decay_rate: Distance decay rate per kilometer
            max_distance: Maximum distance to consider (in kilometers)
            
        Returns:
            2D array of accessibility scores
        """
        if self.centers is None or len(self.centers) == 0:
            raise ValueError("No LINAC centers available for accessibility calculation")
            
        # Convert centers to the same CRS as population data if needed
        # (This would need to be implemented based on the actual CRS of the population data)
        
        # Create output array
        accessibility = np.zeros_like(population_data)
        
        # For each pixel in the population data:
        # 1. Calculate distance to each LINAC center
        # 2. Apply distance decay
        # 3. Sum the accessibility from all centers
        
        # TODO: Implement the actual distance calculation and decay
        # This will need to be optimized for performance, possibly using
        # vectorized operations or parallel processing
        
        return accessibility
        
    def save_centers(self, output_path: Union[str, Path]) -> None:
        """
        Save the current LINAC centers to a CSV file.
        
        Args:
            output_path: Path to save the centers data
        """
        if self.centers is None:
            raise ValueError("No LINAC centers to save")
            
        try:
            # Convert geometry to lat/lon columns
            centers_df = self.centers.copy()
            centers_df['latitude'] = centers_df.geometry.y
            centers_df['longitude'] = centers_df.geometry.x
            centers_df = centers_df.drop(columns=['geometry'])
            
            # Save to CSV
            centers_df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved {len(centers_df)} LINAC centers to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving LINAC centers: {e}")
            raise 