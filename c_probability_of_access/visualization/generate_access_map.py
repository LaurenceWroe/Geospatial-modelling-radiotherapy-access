#!/usr/bin/env python3
"""
Generate probability of access to cancer treatment:
 - Implements accessibility calculations (probability + population weighting).
 - Combines calculation + plotting into a single function for easy integration with GUIs.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from pyproj import Geod
from typing import Optional
import argparse


def generate_accessibility_plot(
    population_raster_path: str,
    linac_excel_path: str,
    country: str = "",
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    lambda_km: float = 30.0,
    max_distance_km: Optional[float] = None,
    dpi: int = 300,
    show_plot: bool = False
):
    """
    Calculate and plot accessibility probability.

    Parameters
    ----------
    population_raster_path : str
        Path to the population density raster file.
    linac_excel_path : str
        Path to the LINAC Excel file.
    country : str, optional
        Country code (used for default naming if provided).
    output_dir : str, optional
        Directory to save output plot.
    output_name : str, optional
        Custom output file name.
    lambda_km : float
        Distance decay parameter in km.
    max_distance_km : float, optional
        Distance cutoff in km (default: 5 * lambda_km).
    dpi : int
        Resolution for output plot.
    show_plot : bool
        If True, show interactive plot instead of saving.

    Returns
    -------
    np.ndarray
        Population-weighted accessibility probability array.
    """

    if max_distance_km is None:
        max_distance_km = 5 * lambda_km

    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
        bounds = src.bounds
        transform = src.transform
        crs = src.crs

    valid_mask = population > 0
    height, width = population.shape

    # Load LINAC locations
    from c_probability_of_access.analysis.excel_utils import read_linac_excel
    df = read_linac_excel(linac_excel_path)
    linac_locations = []
    for _, row in df.iterrows():
        coords = row.get("Coordinates")
        if not coords:
            continue
        try:
            lat, lon = [float(x.strip()) for x in str(coords).split(",")]
            linacs = float(row.get("He Photon And Electron Beam Rt", 0) or 0)
            if linacs > 0:
                linac_locations.append((lat, lon))
        except Exception:
            continue

    print(f"Found {len(linac_locations)} LINAC facilities")

    geod = Geod(ellps="WGS84")
    x = np.linspace(bounds.left, bounds.right, width)
    y = np.linspace(bounds.top, bounds.bottom, height)
    X, Y = np.meshgrid(x, y)

    combined_probability = np.ones((height, width))

    for i, (lat, lon) in enumerate(linac_locations):
        print(f"Processing LINAC {i+1}/{len(linac_locations)} at ({lat:.4f}, {lon:.4f})")

        lons = X.flatten()
        lats = Y.flatten()
        _, _, distances = geod.inv(lon * np.ones_like(lons), lat * np.ones_like(lats), lons, lats)
        distances = distances.reshape(height, width) / 1000.0

        valid_distances = np.where(valid_mask, distances, np.inf)
        within_range = valid_distances <= max_distance_km

        prob_treatment = np.exp(-valid_distances / lambda_km)
        prob_treatment = np.where(within_range, prob_treatment, 0)

        combined_probability *= (1 - prob_treatment)

    combined_probability = 1 - combined_probability
    population_weighted_probability = combined_probability * population
    probability = np.where(valid_mask, population_weighted_probability, np.nan)

    # Plotting
    pixel_width_km = abs(transform[0]) * 111.32
    pixel_height_km = abs(transform[4]) * 111.32
    x_km = np.arange(width) * pixel_width_km
    y_km = np.arange(height) * pixel_height_km
    extent = (0, x_km[-1], 0, y_km[-1])

    valid_probability = probability[valid_mask]
    mean_probability = np.nanmean(valid_probability)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")

    colors = ["blue", "cyan", "yellow", "orange", "red"]
    cmap = LinearSegmentedColormap.from_list("probability", colors, N=100)
    cmap.set_bad(color="white")

    probability_plot = np.flipud(probability)
    im = ax.imshow(probability_plot, extent=extent, origin="lower", cmap=cmap, vmin=0, vmax=1)

    cbar = plt.colorbar(im, label="Probability of Access to Treatment")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.set_title(f"Probability of Access to Cancer Treatment\n(λ={lambda_km} km, Mean: {mean_probability:.1%})")
    ax.set_xlabel("Distance east from origin (km)")
    ax.set_ylabel("Distance north from origin (km)")
    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if output_name:
                output_path = output_dir / output_name
            else:
                base_name = f"{country.lower()}_accessibility_probability_cutoff_{int(lambda_km)}km.png" if country else "accessibility_probability.png"
                output_path = output_dir / base_name
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved plot to {output_path}")

    return probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate probability of access plot for any country with a distance cutoff.")
    parser.add_argument("--country", required=True, help="Country code (e.g., NGA, GBR, USA)")
    parser.add_argument("--population_raster", required=True, help="Path to population raster file")
    parser.add_argument("--linac_excel", required=True, help="Path to LINAC Excel file")
    parser.add_argument("--output_dir", required=True, help="Directory to save output plot")
    parser.add_argument("--lambda_km", type=float, default=30.0, help="Lambda value in km (default: 30.0)")
    parser.add_argument("--cutoff_km", type=float, default=150.0, help="Distance cutoff in km (default: 150.0)")
    parser.add_argument("--output_name", default=None, help="Output PNG filename")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output PNG (default: 300)")
    parser.add_argument("--show_plot", action="store_true", help="Show interactive plot instead of saving")
    args = parser.parse_args()

    generate_accessibility_plot(
        population_raster_path=args.population_raster,
        linac_excel_path=args.linac_excel,
        country=args.country,
        output_dir=args.output_dir,
        output_name=args.output_name,
        lambda_km=args.lambda_km,
        max_distance_km=args.cutoff_km,
        dpi=args.dpi,
        show_plot=args.show_plot
    )
