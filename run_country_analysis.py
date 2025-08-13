#!/usr/bin/env python3
"""
Generalized cancer accessibility analysis for any country.
"""

import argparse
import logging
import rasterio
from pathlib import Path
from typing import Optional

from config.country_config import load_country_config, CountryConfig
from data.worldpop_processor import WorldPopProcessor
from analysis.calculate_accessibility import calculate_accessibility
from visualization.plot_treated_cancer import plot_treated_cancer_density
from visualization.plot_unallocated_cases import plot_unallocated_cases
from visualization.plot_density import plot_density
from visualization.plot_cancer_incidence import plot_treatable_cancer_density_km

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_country_analysis(
    country_code: str,
    lambda_km: Optional[float] = None,
    target_resolution_km: Optional[float] = None,
    generate_maps: bool = True,
    output_dir: Optional[str] = None,
    generate_unallocated_plot: bool = True,
    generate_unallocated_plot_with_overlay: bool = False
) -> None:
    """
    Run complete cancer accessibility analysis for a country.
    
    Args:
        country_code: ISO country code (e.g., 'NGA', 'GBR')
        lambda_km: Override distance decay parameter
        target_resolution_km: Override grid resolution
        generate_maps: Whether to generate visualization maps
        generate_unallocated_plot: Whether to generate unallocated cases plot (without overlay)
        generate_unallocated_plot_with_overlay: Whether to generate unallocated cases plot (with overlay)
    """
    # Load country configuration
    config = load_country_config(country_code)
    
    # Override parameters if provided
    if lambda_km is not None:
        config.lambda_km = lambda_km
        config.max_distance_km = 5 * lambda_km
    
    if target_resolution_km is not None:
        config.target_resolution_km = target_resolution_km
    
    # Override output directory if provided
    if output_dir is not None:
        config.output_dir = Path(output_dir)
        # Update all output file paths to use the new directory
        for key in config.output_files:
            if config.output_files[key]:
                filename = Path(config.output_files[key]).name
                config.output_files[key] = config.output_dir / filename
    
    logger.info(f"Starting analysis for {config.name} ({config.code})")
    logger.info(f"Parameters: λ={config.lambda_km} km, resolution={config.target_resolution_km} km")
    
    # Step 1: Use pre-resampled population data
    logger.info("Step 1: Using pre-resampled population data...")
    
    # Copy the resampled population data to the output directory
    import shutil
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.population_raster_path, config.output_files['population_coarse'])
    
    logger.info(f"Using resampled population data: {config.population_raster_path}")
    logger.info(f"Copied to: {config.output_files['population_coarse']}")
    
    # Plot population density (lat/lon)
    pop_density_map_path = config.output_files.get('population_density_map', None)
    if pop_density_map_path:
        plot_density(
            raster_path=str(config.output_files['population_coarse']),
            title=f"{config.name} Population Density",
            save_path=pop_density_map_path,
            use_log_scale=True
        )

    # Plot treatable cancer density (km axes)
    treatable_cancer_map_path = config.output_files.get('treatable_cancer_density_map', None)
    if treatable_cancer_map_path:
        plot_treatable_cancer_density_km(
            population_raster_path=config.output_files['population_coarse'],
            output_path=treatable_cancer_map_path,
            cancer_incidence_per_year=config.cancer_incidence_per_year,
            fraction_treatable=config.fraction_treatable,
            dpi=300,
            country_name=config.name
        )
    
    # Step 2: Run accessibility analysis
    logger.info("Step 2: Running accessibility analysis...")
    accessibility, stats = calculate_accessibility(
        population_raster_path=config.output_files['population_coarse'],
        linac_excel_path=config.linac_data_path,
        output_path=config.output_files['accessibility'],
        cancer_incidence_per_year=config.cancer_incidence_per_year,
        fraction_treatable=config.fraction_treatable,
        patients_per_linac_per_year=config.patients_per_linac_per_year,
        lambda_km=config.lambda_km
    )
    
    logger.info(f"Analysis complete. Results saved to: {config.output_files['accessibility']}")
    
    # Print summary statistics
    logger.info("\n" + "="*50)
    logger.info(f"ANALYSIS SUMMARY FOR {config.name.upper()}")
    logger.info("="*50)
    logger.info(f"Total population: {stats['total_population']:,.0f}")
    logger.info(f"Total LINACs: {stats['total_linacs']}")
    logger.info(f"Total treatment capacity: {stats['total_capacity']:,.0f} patients/year")
    logger.info(f"Needed capacity: {stats['needed_capacity']:,.0f} patients/year")
    logger.info(f"Capacity gap: {stats['capacity_gap']:,.0f} patients/year")
    logger.info(f"Capacity utilization: {stats['capacity_utilization']:.1%}")
    logger.info(f"Total patients treated: {stats['total_treated']:,.0f}")
    logger.info(f"Total unallocated patients: {stats['total_unallocated']:,.0f}")
    logger.info(f"Distance decay parameter (λ): {stats['lambda']} km")
    
    # Step 3: Generate visualization maps
    if generate_maps:
        logger.info("Step 3: Generating visualization maps...")
        
        # Treated cancer cases map
        plot_treated_cancer_density(
            treated_cancer_raster_path=config.output_files['treated_cases'],
            population_raster_path=config.output_files['population_coarse'],
            linac_excel_path=config.linac_data_path,
            output_path=config.output_files['treated_map'],
            lambda_km=config.lambda_km,
            patients_per_linac_per_year=config.patients_per_linac_per_year
        )
        
        # Unallocated cases map with overlay (fast version - no probability calculation)
        if generate_unallocated_plot_with_overlay:
            plot_unallocated_cases(
                population_raster_path=config.output_files['population_coarse'],
                accessibility_raster_path=config.output_files['accessibility'],
                treated_cancer_raster_path=config.output_files['treated_cases'],
                linac_excel_path=config.linac_data_path,
                output_path=config.output_files['unallocated_map'],
                show_green_overlay=True,
                patients_per_linac_per_year=config.patients_per_linac_per_year,
                lambda_km=config.lambda_km,
                calculate_overlay=False  # Fast version - no probability calculation
            )
        
        # Unallocated cases map without overlay
        if generate_unallocated_plot:
            plot_unallocated_cases(
                population_raster_path=config.output_files['population_coarse'],
                accessibility_raster_path=config.output_files['accessibility'],
                treated_cancer_raster_path=config.output_files['treated_cases'],
                linac_excel_path=config.linac_data_path,
                output_path=config.output_files['unallocated_no_overlay'],
                show_green_overlay=False,
                patients_per_linac_per_year=config.patients_per_linac_per_year,
                lambda_km=config.lambda_km,
                calculate_overlay=False
            )
        
        logger.info("All visualization maps generated successfully!")
    
    logger.info(f"\nAnalysis complete for {config.name}!")
    logger.info(f"Output files saved in: {config.output_dir}")

def main():
    """Command-line interface for country analysis."""
    parser = argparse.ArgumentParser(
        description="Run cancer accessibility analysis for a country"
    )
    parser.add_argument(
        "country_code",
        help="ISO country code (e.g., NGA, GBR)"
    )
    parser.add_argument(
        "--lambda-param",
        type=float,
        help="Distance decay parameter in km (default: from config)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        help="Grid resolution in km (default: from config)"
    )
    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Skip generating visualization maps"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory (default: from config)"
    )
    parser.add_argument(
        "--no-unallocated-plot",
        action="store_true",
        help="Skip generating unallocated cases plot without overlay (default: True)"
    )
    parser.add_argument(
        "--generate-unallocated-plot-with-overlay",
        action="store_true",
        help="Generate unallocated cases plot with overlay (default: False)"
    )
    
    args = parser.parse_args()
    
    try:
        run_country_analysis(
            country_code=args.country_code,
            lambda_km=args.lambda_param,
            target_resolution_km=args.resolution,
            generate_maps=not args.no_maps,
            output_dir=args.output_dir,
            generate_unallocated_plot=not args.no_unallocated_plot,
            generate_unallocated_plot_with_overlay=args.generate_unallocated_plot_with_overlay
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main() 