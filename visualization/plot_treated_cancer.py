import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

def plot_treated_cancer_density(
    treated_cancer_raster_path: str | Path,
    population_raster_path: str | Path | None = None,
    linac_excel_path: str | Path | None = None,
    output_path: str | Path = "",
    dpi: int = 300,
    lambda_km: float = 30.0,
    patients_per_linac_per_year: int = 600
):
    """
    Create a map showing treated cancer density from the treated cases raster.
    Plots in kilometers with the bottom left at [0, 0].
    """
    # Load treated cancer data
    with rasterio.open(treated_cancer_raster_path) as src:
        treated_cancer_density = src.read(1)
        bounds = src.bounds
        transform = src.transform
        pixel_width_km = abs(transform[0]) * 111.32  # degrees to km (approx)
        pixel_height_km = abs(transform[4]) * 111.32
        width = src.width
        height = src.height

    # Load population data to get the mask for valid regions
    if population_raster_path:
        with rasterio.open(population_raster_path) as src:
            population = src.read(1)
            bounds = src.bounds
        # Create mask for areas with population (country boundary)
        # For UK, negative values indicate areas outside the country
        valid_mask = population > 0
        # Also mask out nodata values from population raster
        population_nodata_mask = population != -9999
        valid_mask = valid_mask & population_nodata_mask
    else:
        valid_mask = np.ones_like(treated_cancer_density, dtype=bool)
        bounds = None

    # Mask nodata values
    treated_cancer_density = np.where(treated_cancer_density == -9999, np.nan, treated_cancer_density)

    # Calculate total patients treated (excluding nodata)
    total_patients_treated = np.nansum(treated_cancer_density)

    # Load LINAC locations and calculate combined-probability transparency (optional)
    transparency = None
    if linac_excel_path and bounds is not None and population_raster_path is not None:
        import pandas as pd
        from analysis.calculate_accessibility import calculate_combined_probability
        from analysis.excel_utils import read_linac_excel
        df = read_linac_excel(linac_excel_path)
        linac_facilities = []
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
                    from collections import namedtuple
                    LinacFacility = namedtuple('LinacFacility', ['name', 'lon', 'lat', 'capacity', 'remaining_capacity', 'treated_patients'])
                    linac_facilities.append(LinacFacility(
                        name=row.get('Operator Name', 'Unknown'),
                        lon=lon,
                        lat=lat,
                        capacity=linacs * patients_per_linac_per_year,
                        remaining_capacity=linacs * patients_per_linac_per_year,
                        treated_patients={}
                    ))
            except (ValueError, AttributeError):
                continue
        # Calculate combined probability for transparency
        treatable_cancer_shape = treated_cancer_density.shape
        combined_probability, _ = calculate_combined_probability(
            np.ones(treatable_cancer_shape), linac_facilities, population_raster_path, lambda_km, 5 * lambda_km
        )
        transparency = np.where(valid_mask, combined_probability, np.nan)
        transparency = np.flipud(transparency)

    # Create grid in km from bottom left
    x_km = np.arange(width) * pixel_width_km
    y_km = np.arange(height) * pixel_height_km
    X_km, Y_km = np.meshgrid(x_km, y_km)

    # Plot extent: [0, max_x, 0, max_y]
    extent = (0, x_km[-1], 0, y_km[-1])

    # Prepare data for plotting
    treated_cancer_for_plot = treated_cancer_density.copy()

    if population_raster_path:
        with rasterio.open(population_raster_path) as src:
            population = src.read(1)
            bounds = src.bounds
        # Mask all areas where population <= 0 (including negative values and zero)
        mask_outside = np.array(population <= 0)
        treated_cancer_for_plot[mask_outside] = np.nan
        # Now create a mask for valid (non-NaN) data
        valid_data_mask = ~np.isnan(treated_cancer_for_plot)
        # Set all values < 0.01 to 0.01 for consistent coloring (only for valid data)
        treated_cancer_for_plot[(treated_cancer_for_plot > 0) & (treated_cancer_for_plot < 0.01) & valid_data_mask] = 0.01
        # Set all values > 20 to 20 for consistent coloring (only for valid data)
        treated_cancer_for_plot[(treated_cancer_for_plot > 20) & valid_data_mask] = 20.0
        # Set zero values to 0.01 so they get the same color as ≤ 0.01 (only for valid data)
        treated_cancer_for_plot[(treated_cancer_for_plot == 0) & valid_data_mask] = 0.01
    else:
        # Now create a mask for valid (non-NaN) data
        valid_data_mask = ~np.isnan(treated_cancer_for_plot)
        # Set all values < 0.01 to 0.01 for consistent coloring (only for valid data)
        treated_cancer_for_plot[(treated_cancer_for_plot > 0) & (treated_cancer_for_plot < 0.01) & valid_data_mask] = 0.01
        # Set all values > 20 to 20 for consistent coloring (only for valid data)
        treated_cancer_for_plot[(treated_cancer_for_plot > 20) & valid_data_mask] = 20.0
        # Set zero values to 0.01 so they get the same color as ≤ 0.01 (only for valid data)
        treated_cancer_for_plot[(treated_cancer_for_plot == 0) & valid_data_mask] = 0.01

    standardized_vmin = 0.01
    standardized_vmax = 20.0

    # Use the treated cancer data directly (zero values already set to 0.01)
    plot_data = treated_cancer_for_plot.copy()
    # Ensure NaN values remain NaN for proper masking
    plot_data[~valid_mask] = np.nan
    
    # Flip the data vertically so that the first row (Y=0) appears at the bottom
    plot_data = np.flipud(plot_data)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 15))
    
    norm = LogNorm(vmin=standardized_vmin, vmax=standardized_vmax)
    cmap = cm.get_cmap('viridis')
    
    # Set colormap bad color to white for NaN values (no population areas)
    cmap.set_bad(color='white')
    

    
    if transparency is not None:
        # Create RGBA image with full opacity (no transparency)
        rgba_image = cmap(norm(plot_data))
        # Set full opacity (alpha = 1.0) for all cells
        rgba_image[..., 3] = 1.0
        im1 = ax.imshow(
            rgba_image,
            extent=extent,
            origin='lower'  # bottom left is [0, 0]
        )
    else:
        im1 = ax.imshow(
            plot_data,
            extent=extent,
            origin='lower',  # bottom left is [0, 0]
            cmap=cmap,
            norm=norm
        )

    # Always create a ScalarMappable for the colorbar
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Treated Cancer Cases per km²')
    
    # Set custom ticks and labels (only those within range)
    ticks = [tick for tick in [0.01, 0.1, 1, 10, 20] if standardized_vmin <= tick <= standardized_vmax]
    labels = ['≤ 0.01', '0.1', '1', '10', '≥ 20'][:len(ticks)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)

    # Customize the plot
    ax.set_title(f'Treated Cancer Cases Density\n(Total: {total_patients_treated:,.0f} patients treated, λ={lambda_km} km, Capacity={patients_per_linac_per_year} patients/year)', fontsize=16, pad=20)
    ax.set_xlabel('Distance east from origin (km)')
    ax.set_ylabel('Distance north from origin (km)')

    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved map to {output_path}")
    plt.close()

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data"
    treated_cancer_raster = data_dir / "processed/1km_grid/gbr_cancer_accessibility_treated.tif"
    population_raster = data_dir / "processed/1km_grid/gbr_pd_2020_1km_UNadj.tif"
    output_map = data_dir / "processed/1km_grid/gbr_treated_cancer_density_km.png"
    plot_treated_cancer_density(
        treated_cancer_raster,
        population_raster_path=population_raster,
        output_path=output_map,
        dpi=300
    ) 