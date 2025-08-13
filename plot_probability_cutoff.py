#!/usr/bin/env python3
"""
Generate probability of access plot for any country with a specified lambda value and a distance cutoff (e.g., 150 km).
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from visualization.plot_accessibility_probability import plot_accessibility_probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate probability of access plot for any country with a distance cutoff.")
    parser.add_argument("--country", required=True, help="Country code (e.g., NGA, GBR, USA)")
    parser.add_argument("--population_raster", required=True, help="Path to population raster file")
    parser.add_argument("--linac_excel", required=True, help="Path to LINAC Excel file")
    parser.add_argument("--output_dir", required=True, help="Directory to save output plot")
    parser.add_argument("--lambda_km", type=float, default=30.0, help="Lambda value in km (default: 30.0)")
    parser.add_argument("--cutoff_km", type=float, default=150.0, help="Distance cutoff in km (default: 150.0)")
    parser.add_argument("--output_name", default=None, help="Output PNG filename (default: <country>_accessibility_probability_cutoff_<lambda>km.png)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output PNG (default: 300)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_path = output_dir / args.output_name
    else:
        output_path = output_dir / f"{args.country.lower()}_accessibility_probability_cutoff_{int(args.lambda_km)}km.png"

    print(f"Generating probability plot for {args.country} with λ={args.lambda_km} km and cutoff {args.cutoff_km} km...")
    plot_accessibility_probability(
        population_raster_path=args.population_raster,
        linac_excel_path=args.linac_excel,
        output_path=output_path,
        lambda_km=args.lambda_km,
        max_distance_km=args.cutoff_km,
        dpi=args.dpi
    )
    print(f"\nProbability plot generated successfully!\nOutput: {output_path}") 