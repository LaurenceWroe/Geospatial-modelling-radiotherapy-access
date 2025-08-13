#!/usr/bin/env python3
"""
Population resampling using rioxarray for accurate coordinate handling.
"""

import rioxarray
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, Any

def resample_population_rioxarray(
    raw_population_file: str,
    output_dir: str,
    country_bounds: Dict[str, float],
    target_resolution_km: float = 1.0,
    output_path: str | None = None
) -> Dict[str, Any]:
    """
    Resample population data to target resolution using rioxarray.
    
    Args:
        raw_population_file: Path to raw population raster
        output_dir: Output directory
        country_bounds: Dictionary with 'left', 'bottom', 'right', 'top' bounds
        target_resolution_km: Target resolution in km
        output_path: Optional specific output path
    
    Returns:
        Dictionary with resampling results
    """
    print("=== RESAMPLING POPULATION DATA USING RIOXARRAY ===")
    print(f"Raw file: {raw_population_file}")
    print(f"Target resolution: {target_resolution_km} km")
    
    # Load raw data
    print("Loading raw data...")
    worldpop = rioxarray.open_rasterio(raw_population_file, masked=True)
    # Ensure we have a DataArray, not a Dataset or list
    if isinstance(worldpop, list):
        worldpop = worldpop[0]
    if hasattr(worldpop, 'squeeze'):
        worldpop = worldpop.squeeze()
    
    print(f"Raw data shape: {worldpop.shape}")
    print(f"Raw data CRS: {worldpop.rio.crs}")
    print(f"Raw data bounds: {worldpop.rio.bounds()}")
    print(f"Raw data range: {worldpop.min().values:.1f} to {worldpop.max().values:.1f}")
    
    # Crop to bounding box before resampling
    print("\n=== CROPPING TO BOUNDING BOX ===")
    left = country_bounds['left']
    bottom = country_bounds['bottom']
    right = country_bounds['right']
    top = country_bounds['top']
    worldpop = worldpop.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)
    print(f"Cropped data shape: {worldpop.shape}")
    print(f"Cropped data bounds: {worldpop.rio.bounds()}")
    
    # Calculate current pixel sizes
    x_coords = worldpop.x.values
    y_coords = worldpop.y.values
    
    # Calculate pixel sizes at different latitudes
    print(f"\n=== PIXEL SIZE ANALYSIS ===")
    pixel_sizes_km = []
    for lat in y_coords:
        lat_rad = np.radians(lat)
        cos_lat = np.cos(lat_rad)
        # Calculate pixel width and height in km
        pixel_width_km = abs(x_coords[1] - x_coords[0]) * 111.32 * cos_lat
        pixel_height_km = abs(y_coords[1] - y_coords[0]) * 111.32
        pixel_area_km2 = pixel_width_km * pixel_height_km
        pixel_sizes_km.append(pixel_area_km2)
    
    print(f"Current pixel area range: {min(pixel_sizes_km):.3f} to {max(pixel_sizes_km):.3f} km²")
    print(f"Average pixel area: {np.mean(pixel_sizes_km):.3f} km²")
    
    # Calculate current total population
    print(f"\n=== CURRENT POPULATION CALCULATION ===")
    valid_data = worldpop.where(worldpop > 0, drop=True)
    total_population = 0
    
    for i, lat in enumerate(y_coords):
        lat_data = worldpop.sel(y=lat)
        lat_valid = lat_data.where(lat_data > 0, drop=True)
        if len(lat_valid) > 0:
            lat_pop = (lat_valid * pixel_sizes_km[i]).sum().values
            total_population += lat_pop
    
    print(f"Current total population: {total_population:,.0f}")
    
    # Calculate target resolution for specified km pixels at average latitude
    avg_lat = (country_bounds['top'] + country_bounds['bottom']) / 2
    cos_lat = np.cos(np.radians(avg_lat))
    target_resolution_deg = target_resolution_km / (111.32 * cos_lat)
    
    print(f"\n=== TARGET RESOLUTION ===")
    print(f"Average latitude: {avg_lat:.1f}°")
    print(f"Target resolution: {target_resolution_deg:.6f}°")
    
    # Calculate target dimensions
    width_deg = country_bounds['right'] - country_bounds['left']
    height_deg = country_bounds['top'] - country_bounds['bottom']
    
    target_width = int(width_deg / target_resolution_deg)
    target_height = int(height_deg / target_resolution_deg)
    
    print(f"Target dimensions: {target_width} x {target_height}")
    
    # Reproject using rioxarray
    print(f"\n=== REPROJECTING TO {target_resolution_km}KM RESOLUTION ===")
    resampled = worldpop.rio.reproject(
        dst_crs=worldpop.rio.crs,
        resolution=target_resolution_deg,
        resampling=1,  # Average resampling
        nodata=-99999.0
    )
    
    print(f"Resampled shape: {resampled.shape}")
    print(f"Resampled range: {resampled.min().values:.1f} to {resampled.max().values:.1f}")
    
    # Calculate population using the resampled data
    print(f"\n=== RESAMPLED POPULATION CALCULATION ===")
    resampled_valid = resampled.where(resampled > 0, drop=True)
    total_population_resampled = 0
    
    # Calculate pixel areas for resampled data
    resampled_y_coords = resampled.y.values
    resampled_pixel_sizes_km = []
    
    for lat in resampled_y_coords:
        lat_rad = np.radians(lat)
        cos_lat = np.cos(lat_rad)
        pixel_width_km = target_resolution_deg * 111.32 * cos_lat
        pixel_height_km = target_resolution_deg * 111.32
        pixel_area_km2 = pixel_width_km * pixel_height_km
        resampled_pixel_sizes_km.append(pixel_area_km2)
    
    for i, lat in enumerate(resampled_y_coords):
        lat_data = resampled.sel(y=lat)
        lat_valid = lat_data.where(lat_data > 0, drop=True)
        if len(lat_valid) > 0:
            lat_pop = (lat_valid * resampled_pixel_sizes_km[i]).sum().values
            total_population_resampled += lat_pop
    
    print(f"Resampled total population: {total_population_resampled:,.0f}")
    
    # Save the resampled data
    if output_path is None:
        output_path = str(Path(output_dir) / f"population_density_{target_resolution_km}km.tif")
    else:
        output_path = str(Path(output_path))
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    resampled.rio.to_raster(output_path)
    
    print(f"\nResampled data saved to: {output_path}")
    print(f"Note: This file contains population density (per km²) at {target_resolution_km}km resolution")
    
    return {
        'original_population': total_population,
        'resampled_population': total_population_resampled,
        'output_path': str(output_path)
    } 