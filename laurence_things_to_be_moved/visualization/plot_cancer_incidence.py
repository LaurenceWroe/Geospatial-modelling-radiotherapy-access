import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import LogNorm

def plot_cancer_incidence_density(
    population_raster_path: str,
    output_path: str = "",
    cancer_incidence_per_year: int = 128000,
    fraction_treatable: float = 0.5,
    dpi: int = 300
):
    """
    Create a map showing cancer incidence density (population density × cancer rate).
    """
    # Load population data
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    
    # Debug: check population data
    print(f"Population data range: {population.min()} to {population.max()}")
    print(f"Population data shape: {population.shape}")
    print(f"Number of negative values: {np.sum(population < 0)}")
    print(f"Number of zero values: {np.sum(population == 0)}")
    print(f"Number of positive values: {np.sum(population > 0)}")
    
    # Handle negative and zero values
    population_clean = np.where(population <= 0, 0, population)
    
    # Calculate cancer incidence density
    total_population = np.sum(population_clean)
    cancer_rate_per_capita = cancer_incidence_per_year / total_population
    cancer_incidence_density = population_clean * cancer_rate_per_capita
    
    # Debug: print the actual calculated total
    calculated_total = np.sum(cancer_incidence_density)
    print(f"Calculated total cancer incidence: {calculated_total:,.0f}")
    print(f"Expected total: {cancer_incidence_per_year:,.0f}")
    
    # Use the expected total for the title
    total_cancer_incidence = cancer_incidence_per_year
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot cancer incidence density (log scale)
    im = ax.imshow(
        cancer_incidence_density,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin='upper',
        cmap='viridis',
        norm=LogNorm(vmin=0.1, vmax=cancer_incidence_density.max())
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cancer Incidence Density (cases/km²)')
    
    # Customize the plot
    ax.set_title(f'Cancer Incidence Density in Nigeria\n(Total: {total_cancer_incidence:,.0f} cases/year)', fontsize=16, pad=20)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved map to {output_path}")
    plt.close()

def plot_treatable_cancer_density(
    population_raster_path: str,
    output_path: str = "",
    cancer_incidence_per_year: int = 128000,
    fraction_treatable: float = 0.5,
    dpi: int = 300
):
    """
    Create a map showing treatable cancer density (population density × cancer rate × fraction_treatable).
    """
    # Load population data
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    
    # Handle negative and zero values - set to NaN for proper masking
    population_clean = np.where(population <= 0, np.nan, population)
    
    # Calculate treatable cancer density
    total_population = np.nansum(population_clean)
    cancer_rate_per_capita = cancer_incidence_per_year / total_population
    treatable_cancer_density = population_clean * cancer_rate_per_capita * fraction_treatable
    
    # Debug: print the actual calculated total
    calculated_total = np.sum(treatable_cancer_density)
    print(f"Calculated total treatable cases: {calculated_total:,.0f}")
    print(f"Expected total: {cancer_incidence_per_year * fraction_treatable:,.0f}")
    
    # Use the expected total for the title
    total_treatable_cases = cancer_incidence_per_year * fraction_treatable
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot treatable cancer density (log scale)
    im = ax.imshow(
        treatable_cancer_density,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin='upper',
        cmap='viridis',
        norm=LogNorm(vmin=0.1, vmax=treatable_cancer_density.max())
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Treatable Cancer Cases Density (cases/km²)')
    
    # Customize the plot
    ax.set_title(f'Treatable Cancer Cases Density in Nigeria\n(Total: {total_treatable_cases:,.0f} cases/year requiring radiotherapy)', fontsize=16, pad=20)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved map to {output_path}")
    plt.close()

def plot_treatable_cancer_density_km(
    population_raster_path: str,
    output_path: str = "",
    cancer_incidence_per_year: int = 128000,
    fraction_treatable: float = 0.5,
    dpi: int = 300,
    country_name: str = "Country"
):
    """
    Create a map showing treatable cancer density (population density × cancer rate × fraction_treatable),
    with axes in kilometers from the lower-left corner.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from matplotlib.colors import LogNorm

    # Load population data
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
        transform = src.transform
        bounds = src.bounds
        pixel_width_km = abs(transform[0]) * 111.32  # degrees to km (approx)
        pixel_height_km = abs(transform[4]) * 111.32
        width = src.width
        height = src.height

    # Handle negative and zero values - set to NaN for proper masking
    population_clean = np.where(population <= 0, np.nan, population)

    # Calculate treatable cancer density
    total_population = np.nansum(population_clean)
    cancer_rate_per_capita = cancer_incidence_per_year / total_population
    treatable_cancer_density = population_clean * cancer_rate_per_capita * fraction_treatable

    # Flip the data vertically so that the first row (Y=0) appears at the bottom
    treatable_cancer_density = np.flipud(treatable_cancer_density)

    # Create grid in km from bottom left
    x_km = np.arange(width) * pixel_width_km
    y_km = np.arange(height) * pixel_height_km
    extent = (0, x_km[-1], 0, y_km[-1])

    # Use the expected total for the title
    total_treatable_cases = cancer_incidence_per_year * fraction_treatable

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))

    # Standardize colorbar range: 0.01 to 20
    standardized_vmin = 0.01
    standardized_vmax = 20.0
    
    # Create plot data with standardized range
    plot_data = treatable_cancer_density.copy()
    # Values < 0.01 are set to 0.01 (same color as minimum)
    plot_data[(plot_data > 0) & (plot_data < 0.01)] = 0.01
    # Values > 20 are set to 20 (same color as maximum)
    plot_data[plot_data > 20] = 20.0
    
    # Plot treatable cancer density (log scale)
    im = ax.imshow(
        plot_data,
        extent=extent,
        origin='lower',
        cmap='viridis',
        norm=LogNorm(vmin=standardized_vmin, vmax=standardized_vmax)
    )

    # Add colorbar with custom ticks and labels
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cancer Cases Requiring Radiotherapy Density (cases/km²)')
    
    # Set custom ticks and labels
    ticks = [0.01, 0.1, 1, 10, 20]
    tick_labels = ['≤ 0.01', '0.1', '1', '10', '≥ 20']
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)

    # Customize the plot
    ax.set_title(f'Cancer Cases Requiring Radiotherapy Density in {country_name}\n(Total: {total_treatable_cases:,.0f} cases/year)', fontsize=16, pad=20)
    ax.set_xlabel('Distance east from origin (km)')
    ax.set_ylabel('Distance north from origin (km)')

    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved map to {output_path}")
    plt.close()

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data"
    population_raster = data_dir / "raw" / "nga_pd_2020_1km_UNadj.tif"
    
    # Generate cancer incidence density map
    cancer_incidence_output = data_dir / "processed" / "nga_cancer_incidence_density.png"
    plot_cancer_incidence_density(
        population_raster,
        output_path=cancer_incidence_output,
        cancer_incidence_per_year=128000,
        fraction_treatable=0.5,
        dpi=300
    )
    
    # Generate treatable cancer density map
    treatable_cancer_output = data_dir / "processed" / "nga_treatable_cancer_density.png"
    plot_treatable_cancer_density(
        population_raster,
        output_path=treatable_cancer_output,
        cancer_incidence_per_year=128000,
        fraction_treatable=0.5,
        dpi=300
    ) 