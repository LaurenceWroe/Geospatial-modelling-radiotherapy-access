import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, LinearSegmentedColormap
import pandas as pd
from pathlib import Path
from matplotlib import cm
from scipy.spatial.distance import cdist
from pyproj import Geod

def plot_unallocated_cases(population_raster_path, accessibility_raster_path, treated_cancer_raster_path, linac_excel_path=None, output_path=None, threshold=1.0, show_green_overlay=True, patients_per_linac_per_year=600, lambda_km=30.0, calculate_overlay=True):
    """
    Plot unallocated cancer cases, highlighting areas that have acceptable access to treatment.
    
    Parameters:
    -----------
    population_raster_path : str
        Path to the population density raster file
    accessibility_raster_path : str
        Path to the accessibility raster file (unallocated cases)
    treated_cancer_raster_path : str
        Path to the treated cancer cases raster file
    output_path : str, optional
        Path to save the output PNG file
    threshold : float, optional
        Threshold for considering an area "covered" (unallocated cases per km²)
    show_green_overlay : bool, optional
        Whether to show the green overlay
    """
    # Load population data to get the mask for valid regions
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
    
    # Create mask for valid regions (where population > 0)
    valid_mask = population > 0
    
    with rasterio.open(accessibility_raster_path) as src:
        unallocated = src.read(1)
    
    # Load treated cancer cases to identify service areas
    with rasterio.open(treated_cancer_raster_path) as src:
        treated_cases = src.read(1)
    
    # Replace all -99999 and NaN values with np.nan
    unallocated = np.where((unallocated == -99999) | np.isnan(unallocated), np.nan, unallocated)
    
    # Fix negative unallocated cases by taking absolute values
    unallocated = np.abs(unallocated)
    
    # Apply the valid mask from population data
    unallocated_masked = np.where(valid_mask, unallocated, np.nan)
    
    # Get grid dimensions for plotting
    height, width = population.shape
    
    # Load LINAC locations and calculate overlay (optional)
    transparency = None
    if calculate_overlay and linac_excel_path is not None:
        if linac_excel_path is None:
            # Default to Nigeria LINAC file for backward compatibility
            data_dir = Path(__file__).resolve().parents[2] / "data"
            linac_file = data_dir / "linac" / "Nigeria_DIRAC.xlsx"
        else:
            linac_file = Path(linac_excel_path)
        
        from analysis.excel_utils import read_linac_excel
        df = read_linac_excel(linac_file)
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
        
        # Calculate combined probability-based transparency for each grid cell (vectorized)
        geod = Geod(ellps='WGS84')
        x = np.linspace(bounds.left, bounds.right, width)
        y = np.linspace(bounds.top, bounds.bottom, height)
        X, Y = np.meshgrid(x, y)
        
        # Initialize combined probability array
        combined_probability = np.ones((height, width))
        
        # Vectorized distance calculation for all LINACs
        for lat, lon in linac_locations:
            # Calculate distances from this LINAC to all grid cells at once
            # Use vectorized geodesic calculation
            lons = X.flatten()
            lats = Y.flatten()
            
            # Calculate distances for all points at once
            _, _, distances = geod.inv(lon * np.ones_like(lons), lat * np.ones_like(lats), lons, lats)
            distances = distances.reshape(height, width) / 1000  # Convert to km
            
            # Apply valid mask and max distance filter
            valid_distances = np.where(valid_mask, distances, np.inf)
            within_range = valid_distances <= lambda_km * 5
            
            # Calculate probability of treatment from this LINAC
            prob_treatment = np.exp(-valid_distances / lambda_km)
            prob_treatment = np.where(within_range, prob_treatment, 0)
            
            # Update combined probability: P(not treated by any) = product of P(not treated by each)
            combined_probability *= (1 - prob_treatment)
        
        # Calculate final combined probability of being treated by at least one LINAC
        combined_probability = 1 - combined_probability
        transparency = combined_probability
        transparency = np.where(valid_mask, transparency, np.nan)
    
    # Debug: Find where the maximum transparency occurs
    if transparency is not None:
        max_transparency_idx = np.unravel_index(np.nanargmax(transparency), transparency.shape)
        max_transparency_lat = Y[max_transparency_idx]
        max_transparency_lon = X[max_transparency_idx]
        print(f"Debug - Max combined probability at: lat={max_transparency_lat:.6f}, lon={max_transparency_lon:.6f}")
        print(f"Debug - Max combined probability value: {np.nanmax(transparency):.6f}")
        
        # Explicitly set transparency to 1.0 at LINAC locations
        for lat, lon in linac_locations:
            # Find the closest grid cell to this LINAC
            lat_diff = np.abs(Y - lat)
            lon_diff = np.abs(X - lon)
            total_diff = lat_diff + lon_diff
            linac_idx = np.unravel_index(np.argmin(total_diff), total_diff.shape)
            transparency[linac_idx] = 1.0
            print(f"Debug - Set combined probability=1.0 at LINAC: lat={lat:.6f}, lon={lon:.6f}, grid_idx={linac_idx}")
    else:
        print("Debug - No overlay calculation performed (calculate_overlay=False or no LINAC file)")
    
    # Create mask for LINAC service areas with distance-based transparency
    service_area_mask = (treated_cases > 0) & valid_mask
    service_area_masked = np.where(service_area_mask, 1, np.nan)
    
    # Debug prints
    print("\nData Statistics:")
    print(f"Population range: {population.min():.1f} to {population.max():.1f}")
    print(f"Unallocated range: {unallocated.min():.1f} to {unallocated.max():.1f}")
    print(f"Treated cases range: {treated_cases.min():.1f} to {treated_cases.max():.1f}")
    print(f"Number of cells below threshold ({threshold}): {np.sum(unallocated <= threshold)}")
    print(f"Percentage of cells below threshold: {100 * np.sum(unallocated <= threshold) / np.sum(~np.isnan(unallocated)):.1f}%")
    print(f"Number of cells with treated cases: {np.sum(treated_cases > 0)}")
    
    # Standardize colorbar range: 0.01 to 20
    standardized_vmin = 0.01
    standardized_vmax = 20.0
    
    # Create plot data with standardized range
    unallocated_for_plot = unallocated_masked.copy()
    # Values < 0.01 are set to 0.01 (same color as minimum)
    unallocated_for_plot[(unallocated_for_plot > 0) & (unallocated_for_plot < 0.01)] = 0.01
    # Values > 20 are set to 20 (same color as maximum)
    unallocated_for_plot[unallocated_for_plot > 20] = 20.0
    
    # Ensure only valid data is plotted (exclude NaN and invalid regions)
    unallocated_plot_data = np.where(np.isnan(unallocated_for_plot), np.nan, unallocated_for_plot)
    
    # Check if we have valid data to plot
    if np.all(np.isnan(unallocated_plot_data)):
        print("Warning: All data is NaN, cannot create plot")
        return
    
    # Debug: Print data statistics
    print(f"Debug - Plot data range: {np.nanmin(unallocated_plot_data)} to {np.nanmax(unallocated_plot_data)}")
    print(f"Debug - Number of NaN values: {np.sum(np.isnan(unallocated_plot_data))}")
    print(f"Debug - Number of values <= 0: {np.sum(unallocated_plot_data <= 0)}")
    print(f"Debug - Number of values > 0: {np.sum(unallocated_plot_data > 0)}")
    print(f"Debug - Number of values > 0.1: {np.sum(unallocated_plot_data > 0.1)}")
    print(f"Debug - Number of values > 1: {np.sum(unallocated_plot_data > 1)}")
    print(f"Debug - Number of values > 10: {np.sum(unallocated_plot_data > 10)}")
    
    # Ensure we don't have any negative or zero values that could cause issues
    unallocated_plot_data = np.where((unallocated_plot_data <= 0) | np.isnan(unallocated_plot_data), np.nan, unallocated_plot_data)
    
    # Calculate total unallocated cases from the raster data
    total_unallocated = np.nansum(unallocated_masked)
    
    # Calculate total treated cases from the treated raster
    total_treated = np.nansum(treated_cases)
    
    # Calculate total treatable cases (treated + unallocated)
    total_treatable = total_treated + total_unallocated
    
    print(f"Debug - Total treated cases: {total_treated:,.0f}")
    print(f"Debug - Total unallocated cases: {total_unallocated:,.0f}")
    print(f"Debug - Total treatable cases: {total_treatable:,.0f}")
    
    # Create grid in km from bottom left (same as treated density plots)
    pixel_width_km = abs(transform[0]) * 111.32  # degrees to km (approx)
    pixel_height_km = abs(transform[4]) * 111.32
    x_km = np.arange(width) * pixel_width_km
    y_km = np.arange(height) * pixel_height_km
    X_km, Y_km = np.meshgrid(x_km, y_km)
    
    # Plot extent: [0, max_x, 0, max_y] (same as treated density plots)
    extent = (0, x_km[-1], 0, y_km[-1])
    
    # Create a mask for valid data in kilometer coordinates
    valid_km_mask = np.where(valid_mask, True, False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')  # Set background to white
    
    # Set colormap bad color to white
    cmap_viridis = cm.get_cmap('viridis').copy()
    cmap_viridis.set_bad(color='white')
    
    # Flip the data vertically to match treated cancer plot orientation
    unallocated_plot_data = np.flipud(unallocated_plot_data)
    
    # Plot unallocated cases using imshow with LogNorm, standardized range
    im1 = ax.imshow(unallocated_plot_data,
                    extent=extent,
                    origin='lower',
                    cmap=cmap_viridis,
                    norm=LogNorm(vmin=standardized_vmin, vmax=standardized_vmax))
    
    # Overlay LINAC service areas in bright green
    if show_green_overlay:
        # Debug: Print service area statistics
        print(f"Debug - Service area mask: {np.sum(service_area_mask)} cells")
        if transparency is not None:
            print(f"Debug - Combined probability range: {np.nanmin(transparency)} to {np.nanmax(transparency)}")
            print(f"Debug - Combined probability in service areas: {np.nanmin(transparency[service_area_mask])} to {np.nanmax(transparency[service_area_mask])}")
            print(f"Debug - Example combined probability values (first 10): {transparency[service_area_mask][:10]}")
            
            # Flip the service area mask and transparency to match the data orientation
            service_area_mask_flipped = np.flipud(service_area_mask)
            transparency_flipped = np.flipud(transparency)
        else:
            print("Debug - No transparency data available for overlay")
            service_area_mask_flipped = np.flipud(service_area_mask)
            transparency_flipped = None
        
        # Create an RGBA image for the green overlay
        green_overlay = np.zeros((height, width, 4), dtype=np.float32)
        green_overlay[..., 1] = 1.0  # G channel
        # Use combined probability directly for alpha (no additional scaling)
        if transparency_flipped is not None:
            green_overlay[..., 3] = np.where(service_area_mask_flipped, transparency_flipped, 0)
        else:
            # Use uniform transparency for service areas
            green_overlay[..., 3] = np.where(service_area_mask_flipped, 0.5, 0)

        print(f"Debug - Example green overlay alpha values (first 10): {green_overlay[..., 3][service_area_mask_flipped][:10]}")
        print(f"Debug - Green overlay alpha min/max in service area: {np.nanmin(green_overlay[..., 3][service_area_mask_flipped])} to {np.nanmax(green_overlay[..., 3][service_area_mask_flipped])}")

        im2 = ax.imshow(green_overlay,
                        extent=extent,
                        origin='lower')
    
    # Add colorbar with custom ticks and labels
    cbar = plt.colorbar(im1, label='Unallocated Cancer Cases (cases/km²)')
    
    # Set custom ticks and labels
    ticks = [0.01, 0.1, 1, 10, 20]
    tick_labels = ['≤ 0.01', '0.1', '1', '10', '≥ 20']
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    
    ax.set_title(f'Unallocated Cancer Cases\n(Total: {total_unallocated:,.0f} cases, λ={lambda_km} km, Capacity={patients_per_linac_per_year} patients/year)')
    ax.set_xlabel('Distance east from origin (km)')
    ax.set_ylabel('Distance north from origin (km)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

def plot_patients_treated(population_raster_path, accessibility_raster_path, output_path=None):
    """
    Plot the number of patients treated by LINACs in each grid cell.
    patients_treated = treatable_cancer - unallocated
    """
    # Parameters for cancer incidence
    cancer_incidence_per_year = 128000
    fraction_treatable = 0.5

    # Load population data
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
        bounds = src.bounds
        transform = src.transform
    # Debug: Print population density stats
    print("Population density (min, max):", population.min(), population.max())
    print("Population density (mean, median):", np.mean(population[population > 0]), np.median(population[population > 0]))
    print("Population density (number of cells > 1):", (population > 1).sum())

    # Calculate treatable cancer cases per cell
    total_population = np.sum(population[population > 0])
    cancer_rate_per_capita = cancer_incidence_per_year / total_population
    treatable_cancer = population * cancer_rate_per_capita * fraction_treatable

    # Load unallocated raster
    with rasterio.open(accessibility_raster_path) as src:
        unallocated = src.read(1)

    # Calculate patients treated
    patients_treated = treatable_cancer - unallocated
    patients_treated = np.where(patients_treated < 0, 0, patients_treated)

    # Create masks
    valid_mask = ~np.isnan(patients_treated) & (patients_treated > 0)
    patients_treated_masked = np.ma.masked_where(~valid_mask, patients_treated)
    
    # Create mask for cells below threshold (considered but not allocated)
    below_threshold = (patients_treated > 0) & (patients_treated < 0.1)
    below_threshold_masked = np.ma.masked_where(~below_threshold, patients_treated)
    
    # Debug prints about below-threshold cells
    print("\nBelow threshold statistics:")
    print(f"Number of cells below threshold (0-0.1): {np.sum(below_threshold)}")
    print(f"Range of values in below-threshold cells: {patients_treated[below_threshold].min():.3f} to {patients_treated[below_threshold].max():.3f}")
    print(f"Mean value in below-threshold cells: {np.mean(patients_treated[below_threshold]):.3f}")

    # Plot
    plt.figure(figsize=(12, 8))
    height, width = population.shape
    x = np.linspace(bounds.left, bounds.right, width)
    y = np.linspace(bounds.top, bounds.bottom, height)
    X, Y = np.meshgrid(x, y)
    
    # Plot main patients treated data
    mesh = plt.pcolormesh(X, Y, patients_treated_masked, 
                         cmap='viridis', 
                         shading='auto')
    
    # Overlay cells below threshold in red with higher alpha
    plt.pcolormesh(X, Y, below_threshold_masked,
                  cmap='Reds',  # Changed from 'gray' to 'Reds'
                  alpha=0.7,    # Increased alpha
                  shading='auto')
    
    cbar = plt.colorbar(mesh, label='Patients Treated by LINACs (cases/km²)')
    plt.title('Patients Treated by LINACs per Grid Cell\n(Red = Below Allocation Threshold)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot unallocated cancer cases')
    parser.add_argument('--population', required=True, help='Path to population raster')
    parser.add_argument('--accessibility', required=True, help='Path to accessibility raster')
    parser.add_argument('--treated', required=True, help='Path to treated cancer raster')
    parser.add_argument('--linac', help='Path to LINAC Excel file')
    parser.add_argument('--output', help='Output PNG file path')
    parser.add_argument('--lambda_km', type=float, default=30.0, help='Distance decay parameter (km)')
    parser.add_argument('--capacity', type=int, default=600, help='LINAC capacity (patients/year)')
    parser.add_argument('--no-overlay', action='store_true', help='Disable green overlay')
    parser.add_argument('--calculate-overlay', action='store_true', help='Calculate probability overlay (slow)')
    
    args = parser.parse_args()
    
    plot_unallocated_cases(
        args.population,
        args.accessibility,
        args.treated,
        linac_excel_path=args.linac,
        output_path=args.output,
        threshold=1.0,
        show_green_overlay=not args.no_overlay,
        patients_per_linac_per_year=args.capacity,
        lambda_km=args.lambda_km,
        calculate_overlay=args.calculate_overlay
    ) 