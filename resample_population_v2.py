#!/usr/bin/env python3
"""
User-friendly population data resampling tool for country-level analysis.
"""

import argparse
import sys
from pathlib import Path
from data.resample_population import resample_population_rioxarray

# ======================
# CONSTANTS & SETTINGS
# ======================

COUNTRY_DATA = {
    'GBR': {
        'name': 'United Kingdom',
        'bounds': {'left': -8.6, 'bottom': 49.5, 'right': 1.8, 'top': 60.9},
        'default_file': 'actual_data/raw_from_worldpop/gbr_population.tif'
    },
    'NGA': {
        'name': 'Nigeria',
        'bounds': {'left': 2.7, 'bottom': 4.3, 'right': 14.7, 'top': 13.9},
        'default_file': 'actual_data/raw_from_worldpop/nga_population.tif'
    },
    'USA': {
        'name': 'United States',
        'bounds': {'left': -125.0, 'bottom': 24.0, 'right': -66.9, 'top': 49.4},
        'default_file': 'actual_data/raw_from_worldpop/usa_population.tif'
    }
}

RESOLUTION_OPTIONS = {
    '1': 0.5,   # High detail
    '2': 1.0,   # Standard
    '3': 2.0,   # Medium detail
    '4': 5.0,   # Low detail
    '5': 10.0   # Very low detail
}

# ======================
# CORE FUNCTIONS
# ======================

def resample_country_data(country_code, input_file, output_file, resolution_km=1.0):
    """
    Process population data for a specific country at given resolution.
    
    Args:
        country_code: 3-letter country code (GBR, NGA, USA)
        input_file: Path to source population data
        output_file: Path to save processed data
        resolution_km: Target resolution in kilometers
    """
    # Validate country code
    country_code = country_code.upper()
    if country_code not in COUNTRY_DATA:
        available = list(COUNTRY_DATA.keys())
        raise ValueError(f"Unsupported country. Available: {available}")

    country = COUNTRY_DATA[country_code]
    
    print("\n" + "="*40)
    print(f"PROCESSING {country['name'].upper()} POPULATION DATA")
    print("="*40)
    print(f"• Input file: {input_file}")
    print(f"• Output file: {output_file}")
    print(f"• Resolution: {resolution_km} km")
    print(f"• Country bounds: {country['bounds']}")

    # Perform the resampling
    results = resample_population_rioxarray(
        raw_population_file=input_file,
        output_dir=str(Path(output_file).parent),
        country_bounds=country['bounds'],
        target_resolution_km=resolution_km,
        output_path=output_file
    )

    # Display results
    print("\nRESULTS:")
    print(f"• Original population: {results['original_population']:,.0f}")
    print(f"• Resampled population: {results['resampled_population']:,.0f}")
    print(f"• Output saved to: {results['output_path']}")
    
    return results

# ======================
# USER INTERFACE
# ======================

def show_menu(title, options):
    """Display a simple menu and get valid user selection."""
    print(f"\n{title}:")
    for key, option in options.items():
        print(f"  {key}. {option}")
    
    while True:
        choice = input(f"Select option (1-{len(options)}): ").strip()
        if choice in options:
            return choice
        print(f"Please enter a number between 1-{len(options)}")

def get_custom_resolution():
    """Get valid custom resolution from user."""
    while True:
        try:
            res = float(input("Enter resolution in km (e.g., 0.25, 15.0): ").strip())
            if res > 0:
                return res
            print("Resolution must be positive.")
        except ValueError:
            print("Please enter a valid number.")

def interactive_mode():
    """Guide user through interactive processing."""
    print("\n" + "="*40)
    print("POPULATION DATA RESAMPLING TOOL")
    print("="*40)
    
    # Country selection
    country_choices = {str(i+1): code for i, code in enumerate(COUNTRY_DATA)}
    country_code = country_choices[show_menu(
        "SELECT COUNTRY", 
        {k: f"{COUNTRY_DATA[v]['name']} ({v})" for k, v in country_choices.items()}
    )]
    
    # Resolution selection
    res_choice = show_menu(
        "SELECT RESOLUTION (km)", 
        {**RESOLUTION_OPTIONS, '6': 'Custom resolution'}
    )
    resolution = RESOLUTION_OPTIONS[res_choice] if res_choice != '6' else get_custom_resolution()
    
    # File handling
    default_file = COUNTRY_DATA[country_code]['default_file']
    if show_menu("INPUT FILE", {'1': f"Use default ({Path(default_file).name})", '2': "Custom path"}) == '1':
        input_file = default_file
        if not Path(input_file).exists():
            print(f"Warning: Default file not found at {input_file}")
            input_file = input("Enter path to population data: ").strip()
    else:
        input_file = input("Enter path to population data: ").strip()
    
    # Prepare output
    output_dir = Path("actual_data/resampled")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{country_code}_population_{resolution}km.tif"
    
    # Confirmation
    print("\n" + "="*40)
    print("CONFIRM SETTINGS")
    print("="*40)
    print(f"Country: {COUNTRY_DATA[country_code]['name']} ({country_code})")
    print(f"Resolution: {resolution} km")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    if input("\nProceed with processing? (y/n): ").lower() != 'y':
        print("Processing cancelled.")
        return
    
    # Process the data
    resample_country_data(country_code, input_file, output_file, resolution)

# ======================
# COMMAND LINE INTERFACE
# ======================

def parse_arguments():
    """Configure and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Resample population data to target resolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "country_code",
        nargs='?',
        help="3-letter country code (GBR, NGA, USA)"
    )
    parser.add_argument(
        "input_file",
        nargs='?',
        help="Path to source population data"
    )
    parser.add_argument(
        "output_file",
        nargs='?',
        help="Path to save processed data"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Target resolution in kilometers"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    if args.interactive or not all([args.country_code, args.input_file, args.output_file]):
        try:
            interactive_mode()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
    else:
        try:
            resample_country_data(
                args.country_code,
                args.input_file,
                args.output_file,
                args.resolution
            )
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()