#!/usr/bin/env python3
"""
Generalized population resampling to target resolution for any country.
"""

import argparse
import sys
from data.resample_population import resample_population_rioxarray
from pathlib import Path

# Predefined country bounds
COUNTRY_BOUNDS = {
    'GBR': {
        'left': -8.6,
        'bottom': 49.5,
        'right': 1.8,
        'top': 60.9
    },
    'NGA': {
        'left': 2.7,
        'bottom': 4.3,
        'right': 14.7,
        'top': 13.9
    },
    'USA': {
        'left': -125.0,
        'bottom': 24.0,
        'right': -66.9,
        'top': 49.4
    }
}

def resample_country_population(
    country_code: str,
    raw_population_file: str,
    output_file: str,
    target_resolution_km: float = 1.0
):
    """
    Resample population data for any country to 1km resolution.
    
    Args:
        country_code: ISO country code (e.g., 'GBR', 'NGA')
        raw_population_file: Path to raw population raster
        output_file: Path to save resampled data
    """
    
    # Get country bounds
    if country_code.upper() not in COUNTRY_BOUNDS:
        raise ValueError(f"Country code '{country_code}' not supported. Available: {list(COUNTRY_BOUNDS.keys())}")
    
    country_bounds = COUNTRY_BOUNDS[country_code.upper()]
    
    print(f"=== RESAMPLING {country_code.upper()} POPULATION DATA ===")
    print(f"Raw file: {raw_population_file}")
    print(f"Output file: {output_file}")
    print(f"Country bounds: {country_bounds}")
    
    # Resample population data
    results = resample_population_rioxarray(
        raw_population_file=raw_population_file,
        output_dir=str(Path(output_file).parent),
        country_bounds=country_bounds,
        target_resolution_km=target_resolution_km,
        output_path=output_file
    )
    
    print(f"\n=== RESULTS ===")
    print(f"Original population: {results['original_population']:,.0f}")
    print(f"Resampled population: {results['resampled_population']:,.0f}")
    print(f"Output file: {results['output_path']}")
    
    print(f"\nResampling complete! Use {output_file} for subsequent analysis.")
    
    return results


def get_user_input():
    """
    Interactive user input for country and resolution selection.
    
    Returns:
        tuple: (country_code, target_resolution_km, raw_file, output_file)
    """
    print("=== POPULATION RESAMPLING TOOL ===")
    print("This tool resamples population raster data to your chosen resolution.")
    print("It will crop the data to the selected country's boundaries and")
    print("resample to the specified grid resolution.\n")
    
    # Display available countries
    print("Available countries:")
    for i, country_code in enumerate(COUNTRY_BOUNDS.keys(), 1):
        country_names = {
            'GBR': 'United Kingdom',
            'NGA': 'Nigeria', 
            'USA': 'United States'
        }
        print(f"  {i}. {country_code} - {country_names.get(country_code, country_code)}")
    
    # Get country selection
    while True:
        try:
            choice = input(f"\nSelect country (1-{len(COUNTRY_BOUNDS)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(COUNTRY_BOUNDS):
                country_code = list(COUNTRY_BOUNDS.keys())[int(choice) - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(COUNTRY_BOUNDS)}")
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
    
    print(f"\nSelected country: {country_code}")
    
    # Get target resolution
    print("\nCommon resolution options:")
    print("  1. 0.5 km (high detail)")
    print("  2. 1.0 km (standard)")
    print("  3. 2.0 km (medium detail)")
    print("  4. 5.0 km (low detail)")
    print("  5. 10.0 km (very low detail)")
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
    
    # Get input file path
    print("\nInput file options:")
    print("  1. Use default path for selected country")
    print("  2. Enter custom path")
    
    default_files = {
        'GBR': 'data/raw/gbr_pd_2020_1km.tif',
        'NGA': 'data/raw/nga_pd_2020_1km_UNadj.tif',
        'USA': 'data/raw/usa_pd_2020_1km_UNadj.tif'
    }
    
    while True:
        try:
            file_choice = input("Select option (1-2): ").strip()
            if file_choice == "1":
                raw_file = default_files[country_code]
                if not Path(raw_file).exists():
                    print(f"Warning: Default file {raw_file} not found.")
                    raw_file = input("Please enter the path to your population raster file: ").strip()
                break
            elif file_choice == "2":
                raw_file = input("Enter the path to your population raster file: ").strip()
                break
            else:
                print("Please enter 1 or 2")
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
    
    # Generate output file path
    output_dir = Path("data/raw/resampled")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{country_code.lower()}_population_{target_resolution_km}km.tif"
    
    print(f"\nOutput will be saved to: {output_file}")
    
    return country_code, target_resolution_km, str(raw_file), str(output_file)


def main():
    """Command-line interface for population resampling."""
    parser = argparse.ArgumentParser(
        description="Resample population data to target resolution"
    )
    parser.add_argument(
        "country_code",
        nargs='?',
        help="ISO country code (e.g., GBR, NGA). If not provided, interactive mode will be used."
    )
    parser.add_argument(
        "raw_file",
        nargs='?',
        help="Path to raw population raster file. Required if country_code is provided."
    )
    parser.add_argument(
        "output_file",
        nargs='?',
        help="Path to save resampled data. Required if country_code is provided."
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Target grid resolution in km (default: 1.0)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode even if arguments are provided"
    )
    
    args = parser.parse_args()
    
    # Check if we should use interactive mode
    if args.interactive or not all([args.country_code, args.raw_file, args.output_file]):
        print("Starting interactive mode...\n")
        try:
            country_code, target_resolution_km, raw_file, output_file = get_user_input()
            
            # Confirm before proceeding
            print(f"\n=== CONFIRMATION ===")
            print(f"Country: {country_code}")
            print(f"Resolution: {target_resolution_km} km")
            print(f"Input file: {raw_file}")
            print(f"Output file: {output_file}")
            
            confirm = input("\nProceed with resampling? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                resample_country_population(
                    country_code=country_code,
                    raw_population_file=raw_file,
                    output_file=output_file,
                    target_resolution_km=target_resolution_km
                )
            else:
                print("Resampling cancelled.")
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
            resample_country_population(
                country_code=args.country_code,
                raw_population_file=args.raw_file,
                output_file=args.output_file,
                target_resolution_km=args.resolution
            )
        except Exception as e:
            print(f"Error: {e}")
            raise

if __name__ == "__main__":
    main() 