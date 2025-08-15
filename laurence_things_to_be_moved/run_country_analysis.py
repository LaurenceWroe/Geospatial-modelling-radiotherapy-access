#!/usr/bin/env python3
"""
Generalized cancer accessibility analysis for any country.
"""

import argparse
import logging
import rasterio
import sys
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


def get_user_input():
    """
    Interactive user input for country and resolution selection.
    
    Returns:
        tuple: (country_code, target_resolution_km)
    """
    print("=== CANCER ACCESSIBILITY ANALYSIS TOOL ===\n")
    print("This tool analyzes cancer treatment accessibility for different countries.")
    print("It will use population data at your chosen resolution and calculate")
    print("accessibility to LINAC facilities.\n")
    
    # Available countries
    available_countries = ['NGA', 'GBR', 'USA']
    country_names = {
        'NGA': 'Nigeria',
        'GBR': 'United Kingdom', 
        'USA': 'United States'
    }
    
    # Display available countries
    print("Available countries:")
    for i, country_code in enumerate(available_countries, 1):
        print(f"  {i}. {country_code} - {country_names.get(country_code, country_code)}")
    
    # Get country selection
    while True:
        try:
            choice = input(f"\nSelect country (1-{len(available_countries)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(available_countries):
                country_code = available_countries[int(choice) - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(available_countries)}")
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
    
    print(f"\nSelected country: {country_code}")
    
    # Get target resolution
    print("\nResolution options:")
    print("  1. 0.5 km (high detail - slower analysis)")
    print("  2. 1.0 km (standard - balanced)")
    print("  3. 2.0 km (medium detail - faster)")
    print("  4. 5.0 km (low detail - much faster)")
    print("  5. 10.0 km (very low detail - fastest)")
    print("  6. Custom resolution")
    
    while True:
        try:
            res_choice = input("\nSelect resolution (1-6): ").strip()
            if res_choice == "1":
                target_resolution_km = 0.5
            elif res_choice == "2":
                target_resolution_km = 1.0
            elif res_choice == "3":
                target_resolution_km = 2.0
            elif res_choice == "4":
                target_resolution_km = 5.0
            elif res_choice == "5":
                target_resolution_km = 10.0
            elif res_choice == "6":
                while True:
                    try:
                        custom_res = input("Enter custom resolution in km (e.g., 0.25, 15.0): ").strip()
                        target_resolution_km = float(custom_res)
                        if target_resolution_km > 0:
                            break
                        else:
                            print("Resolution must be positive. Please try again.")
                    except ValueError:
                        print("Invalid number. Please try again.")
                break
            else:
                print("Please enter a number between 1 and 6")
                continue
            break
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
    
    print(f"Selected resolution: {target_resolution_km} km")
    
    # Check if population data exists at requested resolution
    data_dir = Path(__file__).resolve().parents[1] / "data"
    expected_population_path = data_dir / "raw" / "resampled" / f"{country_code.lower()}_population_{target_resolution_km}km.tif"
    
    if not expected_population_path.exists():
        print(f"\n⚠️  Note: Population data at {target_resolution_km}km resolution not found.")
        print(f"   Expected path: {expected_population_path}")
        print(f"   The analysis will use available data and generate {target_resolution_km}km output.")
        print(f"   To prepare data at this resolution, run: python3 resample_population.py {country_code} --resolution {target_resolution_km}")
    
    # Get analysis options
    print("\nAnalysis options:")
    print("  1. Generate all maps and plots (recommended)")
    print("  2. Skip map generation (faster)")
    
    while True:
        try:
            maps_choice = input("Select option (1-2): ").strip()
            if maps_choice == "1":
                generate_maps = True
                break
            elif maps_choice == "2":
                generate_maps = False
                break
            else:
                print("Please enter 1 or 2")
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
    
    return country_code, target_resolution_km, generate_maps


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
    
    # Step 1: Check and use population data at requested resolution
    logger.info("Step 1: Checking population data availability...")
    
    # Check if population data exists at the requested resolution
    import shutil
    from pathlib import Path
    
    # Define expected population file paths for different resolutions
    data_dir = Path(__file__).resolve().parents[1] / "data"
    expected_population_path = data_dir / "raw" / "resampled" / f"{config.code.lower()}_population_{config.target_resolution_km}km.tif"
    
    if expected_population_path.exists():
        # Use the exact resolution requested
        population_source = expected_population_path
        logger.info(f"Found population data at requested resolution: {population_source}")
    else:
        # Fall back to available resolution (usually 1km)
        fallback_path = config.population_raster_path
        logger.warning(f"Population data at {config.target_resolution_km}km resolution not found.")
        logger.warning(f"Using available data: {fallback_path}")
        logger.warning(f"Note: Output will still be generated at {config.target_resolution_km}km resolution")
        population_source = fallback_path
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the population data to the output directory
    shutil.copy2(population_source, config.output_files['population_coarse'])
    
    logger.info(f"Using population data: {population_source}")
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
        nargs='?',
        help="ISO country code (e.g., NGA, GBR). If not provided, interactive mode will be used."
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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode even if arguments are provided"
    )
    
    args = parser.parse_args()
    
    # Check if we should use interactive mode
    if args.interactive or not args.country_code:
        print("Starting interactive mode...\n")
        try:
            country_code, target_resolution_km, generate_maps = get_user_input()
            
            # Confirm before proceeding
            print(f"\n=== CONFIRMATION ===")
            print(f"Country: {country_code}")
            print(f"Resolution: {target_resolution_km} km")
            print(f"Generate maps: {'Yes' if generate_maps else 'No'}")
            
            confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_country_analysis(
                    country_code=country_code,
                    lambda_km=None,
                    target_resolution_km=target_resolution_km,
                    generate_maps=generate_maps,
                    output_dir=None,
                    generate_unallocated_plot=True,
                    generate_unallocated_plot_with_overlay=False
                )
            else:
                print("Analysis cancelled.")
                return
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return
        except Exception as e:
            print(f"Error in interactive mode: {e}")
            raise
    else:
        # Command-line mode
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