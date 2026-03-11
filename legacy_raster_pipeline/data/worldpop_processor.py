import rasterio
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Union, Tuple, Optional
import logging
import rasterio.warp
from rasterio.enums import Resampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorldPopProcessor:
    """Class for processing WorldPop population density data."""
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the WorldPop processor.
        
        Args:
            data_path: Path to the WorldPop GeoTIFF file
        """
        self.data_path = Path(data_path)
        self.dataset = None
        self.population_data = None
        
    def load_data(self) -> None:
        """Load the WorldPop dataset."""
        try:
            self.dataset = rasterio.open(self.data_path)
            logger.info(f"Successfully loaded WorldPop data from {self.data_path}")
        except Exception as e:
            logger.error(f"Error loading WorldPop data: {e}")
            raise
            
    def extract_country(self, country_boundary: gpd.GeoDataFrame) -> Tuple[np.ndarray, dict]:
        """
        Extract population data for a specific country.
        
        Args:
            country_boundary: GeoDataFrame containing the country's boundary
            
        Returns:
            Tuple of (population_data, metadata)
        """
        if self.dataset is None:
            self.load_data()
            
        try:
            # Get the bounding box of the country
            minx, miny, maxx, maxy = country_boundary.total_bounds
            
            # Read the data within the bounding box
            window = self.dataset.window(minx, miny, maxx, maxy)
            population_data = self.dataset.read(1, window=window)
            
            # Update the transform for the cropped data
            transform = rasterio.windows.transform(window, self.dataset.transform)
            metadata = self.dataset.meta.copy()
            metadata.update({
                'height': window.height,
                'width': window.width,
                'transform': transform
            })
            
            logger.info(f"Successfully extracted country data with shape {population_data.shape}")
            return population_data, metadata
            
        except Exception as e:
            logger.error(f"Error extracting country data: {e}")
            raise
            
    def convert_to_cancer_density(self, 
                                population_data: np.ndarray, 
                                scaling_factor: float = 0.001) -> np.ndarray:
        """
        Convert population density to cancer density using a scaling factor.
        
        Args:
            population_data: Population density data
            scaling_factor: Factor to convert population to cancer cases
            
        Returns:
            Cancer density data
        """
        return population_data * scaling_factor
        
    def save_processed_data(self, 
                          data: np.ndarray, 
                          metadata: dict, 
                          output_path: Union[str, Path]) -> None:
        """
        Save processed data to a new GeoTIFF file.
        
        Args:
            data: Processed data array
            metadata: Raster metadata
            output_path: Path to save the output file
        """
        output_path = Path(output_path)
        try:
            with rasterio.open(output_path, 'w', **metadata) as dst:
                dst.write(data, 1)
            logger.info(f"Successfully saved processed data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
            
    def resample_to_coarse_resolution(self, 
                                    population_data: np.ndarray, 
                                    metadata: dict, 
                                    target_resolution_km: float = 10.0) -> Tuple[np.ndarray, dict]:
        """
        Resample population data to a coarser resolution using rasterio.warp.reproject with Resampling.sum.
        
        Args:
            population_data: Population density data
            metadata: Raster metadata
            target_resolution_km: Target resolution in kilometers
            
        Returns:
            Tuple of (resampled_data, updated_metadata)
        """
        try:
            # Calculate the scale factor for resampling
            # Convert degrees to kilometers (approximate: 1 degree ≈ 111.32 km at equator)
            current_resolution_degrees = abs(metadata['transform'][0])  # Pixel size in degrees
            current_resolution_km = current_resolution_degrees * 111.32  # Convert to km
            scale_factor = target_resolution_km / current_resolution_km
            
            # Calculate new dimensions
            new_height = int(metadata['height'] / scale_factor)
            new_width = int(metadata['width'] / scale_factor)
            
            # Create new transform for the coarser resolution
            new_transform = rasterio.Affine(
                metadata['transform'][0] * scale_factor,
                metadata['transform'][1],
                metadata['transform'][2],
                metadata['transform'][3],
                metadata['transform'][4] * scale_factor,
                metadata['transform'][5]
            )
            
            # Prepare destination array
            resampled_data = np.zeros((new_height, new_width), dtype=population_data.dtype)
            
            # Perform the resampling using sum aggregation
            rasterio.warp.reproject(
                source=population_data,
                destination=resampled_data,
                src_transform=metadata['transform'],
                src_crs=metadata['crs'],
                dst_transform=new_transform,
                dst_crs=metadata['crs'],
                resampling=Resampling.sum,
                src_nodata=metadata.get('nodata', -99999),
                dst_nodata=-99999
            )
            
            # Update metadata
            new_metadata = metadata.copy()
            new_metadata.update({
                'height': new_height,
                'width': new_width,
                'transform': new_transform,
                'nodata': -99999  # Ensure nodata is set
            })
            
            logger.info(f"Successfully resampled data from {current_resolution_km:.2f}km to {target_resolution_km}km resolution")
            logger.info(f"Original shape: {population_data.shape}, New shape: {resampled_data.shape}")
            
            return resampled_data, new_metadata
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            raise 