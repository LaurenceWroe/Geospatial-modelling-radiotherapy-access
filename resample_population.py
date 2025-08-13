#!/usr/bin/env python3
"""
Generalized population resampling to 1km resolution for any country.
"""

import argparse
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

def main():
    """Command-line interface for population resampling."""
    parser = argparse.ArgumentParser(
        description="Resample population data to 1km resolution"
    )
    parser.add_argument(
        "country_code",
        help="ISO country code (e.g., GBR, NGA)"
    )
    parser.add_argument(
        "raw_file",
        help="Path to raw population raster file"
    )
    parser.add_argument(
        "output_file",
        help="Path to save resampled data"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Target grid resolution in km (default: 1.0)"
    )
    
    args = parser.parse_args()
    
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