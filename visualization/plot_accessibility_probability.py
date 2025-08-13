import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pathlib import Path
from matplotlib import cm
from scipy.spatial.distance import cdist
from pyproj import Geod
from typing import Optional

def calculate_accessibility_probability(
    population_raster_path: str | Path,
    linac_excel_path: str | Path,
    lambda_km: float = 30.0,
    max_distance_km: Optional[float] = None
) -> np.ndarray:
    """
    Calculate the probability of access to cancer treatment for each grid cell.
    
    Parameters:
    -----------
    population_raster_path : str | Path
        Path to the population density raster file
    linac_excel_path : str | Path
        Path to the LINAC Excel file
    lambda_km : float
        Distance decay parameter in km
    max_distance_km : float, optional
        Maximum distance to consider (default: 5 * lambda_km)
    
    Returns:
    --------
    np.ndarray : Combined probability of access for each grid cell
    """
    if max_distance_km is None:
        max_distance_km = 5 * lambda_km
    
    # Load population data to get grid dimensions and bounds
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
    
    # Create mask for valid regions (where population > 0)
    valid_mask = population > 0
    height, width = population.shape
    
    # Load LINAC locations
    from analysis.excel_utils import read_linac_excel
    df = read_linac_excel(linac_excel_path)
    linac_locations = []
    
    # Extract LINAC locations from the Excel file
    for _, row in df.iterrows():
        coords = row.get('Coordinates')
        if coords is None or (hasattr(coords, '__bool__') and not bool(coords)):
            continue
        try:
            coords_str = str(coords)
            lat, lon = [float(x.strip()) for x in coords_str.split(',')]
            linacs = row.get('He Photon And Electron Beam Rt', 0)
            if linacs is None or (hasattr(linacs, '__bool__') and not bool(linacs)):
                linacs = 0
            if float(linacs) > 0:
                linac_locations.append((lat, lon))
        except (ValueError, AttributeError):
            continue
    
    print(f"Found {len(linac_locations)} LINAC facilities")
    
    # Calculate combined probability-based accessibility for each grid cell (vectorized)
    geod = Geod(ellps='WGS84')
    x = np.linspace(bounds.left, bounds.right, width)
    y = np.linspace(bounds.top, bounds.bottom, height)
    X, Y = np.meshgrid(x, y)
    
    # Initialize combined probability array
    combined_probability = np.ones((height, width))
    
    # Vectorized distance calculation for all LINACs
    for i, (lat, lon) in enumerate(linac_locations):
        print(f"Processing LINAC {i+1}/{len(linac_locations)} at ({lat:.4f}, {lon:.4f})", flush=True)
        
        # Calculate distances from this LINAC to all grid cells at once
        lons = X.flatten()
        lats = Y.flatten()
        
        # Calculate distances for all points at once
        _, _, distances = geod.inv(lon * np.ones_like(lons), lat * np.ones_like(lats), lons, lats)
        distances = distances.reshape(height, width) / 1000  # Convert to km
        
        # Apply valid mask and max distance filter
        valid_distances = np.where(valid_mask, distances, np.inf)
        within_range = valid_distances <= max_distance_km
        
        # Calculate probability of treatment from this LINAC
        prob_treatment = np.exp(-valid_distances / lambda_km)
        prob_treatment = np.where(within_range, prob_treatment, 0)
        
        # Update combined probability: P(not treated by any) = product of P(not treated by each)
        combined_probability *= (1 - prob_treatment)
    
    # Calculate final combined probability of being treated by at least one LINAC
    combined_probability = 1 - combined_probability
    combined_probability = np.where(valid_mask, combined_probability, np.nan)
    
    return combined_probability

def plot_accessibility_probability(
    population_raster_path: str | Path,
    linac_excel_path: str | Path,
    output_path: Optional[str | Path] = None,
    lambda_km: float = 30.0,
    max_distance_km: Optional[float] = None,
    dpi: int = 300
):
    """
    Plot the probability of access to cancer treatment.
    
    Parameters:
    -----------
    population_raster_path : str | Path
        Path to the population density raster file
    linac_excel_path : str | Path
        Path to the LINAC Excel file
    output_path : str | Path, optional
        Path to save the output PNG file
    lambda_km : float
        Distance decay parameter in km
    max_distance_km : float, optional
        Maximum distance to consider (default: 5 * lambda_km)
    dpi : int
        DPI for the output image
    """
    print("Calculating accessibility probability...")
    probability = calculate_accessibility_probability(
        population_raster_path, linac_excel_path, lambda_km, max_distance_km
    )
    
    # Load population data to get bounds and transform
    with rasterio.open(population_raster_path) as src:
        bounds = src.bounds
        transform = src.transform
        population = src.read(1)
    
    # Create mask for valid regions
    valid_mask = population > 0
    
    # Get grid dimensions for plotting
    height, width = population.shape
    
    # Create grid in km from bottom left (same as other plots)
    pixel_width_km = abs(transform[0]) * 111.32  # degrees to km (approx)
    pixel_height_km = abs(transform[4]) * 111.32
    x_km = np.arange(width) * pixel_width_km
    y_km = np.arange(height) * pixel_height_km
    X_km, Y_km = np.meshgrid(x_km, y_km)
    
    # Plot extent: [0, max_x, 0, max_y] (same as other plots)
    extent = (0, x_km[-1], 0, y_km[-1])
    
    # Calculate statistics
    valid_probability = probability[valid_mask]
    mean_probability = np.nanmean(valid_probability)
    median_probability = np.nanmedian(valid_probability)
    max_probability = np.nanmax(valid_probability)
    
    print(f"Probability statistics:")
    print(f"  Mean: {mean_probability:.3f}")
    print(f"  Median: {median_probability:.3f}")
    print(f"  Maximum: {max_probability:.3f}")
    print(f"  Cells with >50% probability: {np.sum(valid_probability > 0.5):,}")
    print(f"  Cells with >90% probability: {np.sum(valid_probability > 0.9):,}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    # Create custom colormap from blue (low probability) to red (high probability)
    colors = ['blue', 'cyan', 'yellow', 'orange', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('probability', colors, N=n_bins)
    cmap.set_bad(color='white')
    
    # Flip the data vertically to match other plots
    probability_plot = np.flipud(probability)
    
    # Plot probability using imshow
    im = ax.imshow(probability_plot,
                   extent=extent,
                   origin='lower',
                   cmap=cmap,
                   vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Probability of Access to Treatment')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Add title and labels
    ax.set_title(f'Probability of Access to Cancer Treatment\n(λ={lambda_km} km, Mean: {mean_probability:.1%})')
    ax.set_xlabel('Distance east from origin (km)')
    ax.set_ylabel('Distance north from origin (km)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    return probability

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot accessibility probability')
    parser.add_argument('--population', required=True, help='Path to population raster')
    parser.add_argument('--linac', required=True, help='Path to LINAC Excel file')
    parser.add_argument('--output', help='Output PNG file path')
    parser.add_argument('--lambda_km', type=float, default=30.0, help='Distance decay parameter (km)')
    parser.add_argument('--max_distance', type=float, help='Maximum distance to consider (km)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output image')
    
    args = parser.parse_args()
    
    plot_accessibility_probability(
        args.population,
        args.linac,
        output_path=args.output,
        lambda_km=args.lambda_km,
        max_distance_km=args.max_distance,
        dpi=args.dpi
    ) 