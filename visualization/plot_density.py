import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import LogNorm

def plot_density(raster_path: str, title: str = "Nigeria Population Density", vmin=None, vmax=None, save_path=None, use_log_scale=True):
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        bounds = src.bounds
        # Mask nodata values
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        
        # Remove zeros and negative values for log scale
        if use_log_scale:
            data = np.where(data <= 0, np.nan, data)
        
        # Calculate total population by accounting for actual pixel areas
        # Get pixel dimensions in degrees
        pixel_width_deg = (bounds.right - bounds.left) / data.shape[1]
        pixel_height_deg = (bounds.top - bounds.bottom) / data.shape[0]
        
        # Calculate pixel areas in km² (accounting for latitude)
        total_population = 0
        for i in range(data.shape[0]):
            # Calculate latitude for this row
            lat = bounds.top - (i + 0.5) * pixel_height_deg
            lat_rad = np.radians(lat)
            cos_lat = np.cos(lat_rad)
            
            # Calculate pixel area for this latitude
            pixel_width_km = pixel_width_deg * 111.32 * cos_lat
            pixel_height_km = pixel_height_deg * 111.32
            pixel_area_km2 = pixel_width_km * pixel_height_km
            
            # Sum population for this row
            row_data = data[i, :]
            valid_mask = ~np.isnan(row_data)
            if np.any(valid_mask):
                row_population = np.sum(row_data[valid_mask] * pixel_area_km2)
                total_population += row_population
        
        # Standardize colorbar range: 1 to 10,000
        standardized_vmin = 1.0
        standardized_vmax = 10000.0
        
        # Create plot data with standardized range
        plot_data = data.copy()
        # Values < 1 are set to 1 (same color as minimum)
        plot_data[(plot_data > 0) & (plot_data < 1)] = 1.0
        # Values > 10,000 are set to 10,000 (same color as maximum)
        plot_data[plot_data > 10000] = 10000.0
        
        plt.figure(figsize=(10, 8))
        
        if use_log_scale:
            img = plt.imshow(plot_data, cmap='viridis', norm=LogNorm(vmin=standardized_vmin, vmax=standardized_vmax),
                             extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), origin='upper')
        else:
            img = plt.imshow(plot_data, cmap='viridis', vmin=standardized_vmin, vmax=standardized_vmax,
                             extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), origin='upper')
        
        # Add colorbar with custom ticks and labels
        cbar = plt.colorbar(img, label='Population Density (persons/km²)')
        
        # Set custom ticks and labels
        ticks = [1, 10, 100, 1000, 10000]
        tick_labels = ['≤ 1', '10', '100', '1,000', '≥ 10,000']
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        
        plt.title(f'{title}\n(Total Population: {total_population:,.0f})')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        plt.close()

def plot_cancer_density(raster_path: str, title: str = "Nigeria Cancer Density", save_path=None):
    """Plot cancer density with log scale."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        # Mask nodata values
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        
        plt.figure(figsize=(10, 8))
        img = plt.imshow(data, cmap='viridis', norm=LogNorm(vmin=0.1, vmax=data.max()))
        plt.colorbar(img, label='Cancer Density (cases/km²)')
        plt.title(title)
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    # Generate population density map with log scale
    data_dir = Path(__file__).parent.parent.parent / "data"
    population_raster = data_dir / "raw" / "nga_pd_2020_1km_UNadj.tif"
    population_save_path = data_dir / "processed" / "nga_population_density_log.png"
    
    plot_density(
        str(population_raster), 
        title="Nigeria Population Density (2020, Log Scale)", 
        save_path=str(population_save_path),
        use_log_scale=True
    ) 