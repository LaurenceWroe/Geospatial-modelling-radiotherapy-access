import numpy as np
import rasterio
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cdist
from pyproj import Geod
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict
import heapq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LinacFacility:
    name: str
    lon: float
    lat: float
    capacity: int  # patients per year
    remaining_capacity: int
    treated_patients: Dict[Tuple[int, int], float]  # (row, col) -> number of patients treated

def distance_decay(distance_km: float, lambda_km: float) -> float:
    """
    Calculate the distance decay factor using P(d) = exp(-d/λ)
    where d is distance in km and λ is the decay parameter
    """
    return np.exp(-distance_km / lambda_km)

def calculate_treatable_cancer_density(
    population_raster_path: str | Path,
    cancer_incidence_per_year: int,
    fraction_treatable: float
) -> Tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Calculate the density of treatable cancer cases per grid cell."""
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
        transform = src.transform
        crs = src.crs
        
    # Handle negative and zero values - set to 0 for calculations (plotting will mask them)
    population_clean = np.where(population <= 0, 0, population)
    
    # Calculate total population and cancer rate per capita
    total_population = np.sum(population_clean)
    cancer_rate_per_capita = cancer_incidence_per_year / total_population
    
    # Calculate treatable cancer density
    treatable_cancer = population_clean * cancer_rate_per_capita * fraction_treatable
    
    return treatable_cancer, transform, crs

def get_grid_coordinates(
    transform: rasterio.transform.Affine,
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the geographic coordinates for each grid cell center and corners.
    Returns (center_lons, center_lats, corner_lons, corner_lats)
    """
    rows, cols = shape
    x_coords = np.arange(cols)
    y_coords = np.arange(rows)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Get center coordinates
    center_lons, center_lats = rasterio.transform.xy(transform, yy.flatten(), xx.flatten())
    
    # Get corner coordinates (for distance calculations)
    corner_lons = []
    corner_lats = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            lon, lat = rasterio.transform.xy(transform, i, j)
            corner_lons.append(lon)
            corner_lats.append(lat)
    
    return (np.array(center_lons), np.array(center_lats),
            np.array(corner_lons), np.array(corner_lats))

def calculate_distances(
    linac: LinacFacility,
    center_lons: np.ndarray,
    center_lats: np.ndarray,
    corner_lons: np.ndarray,
    corner_lats: np.ndarray,
    shape: Tuple[int, int]
) -> np.ndarray:
    """
    Calculate distances from LINAC to all grid cells.
    Uses the minimum distance from LINAC to any point in the grid cell.
    """
    geod = Geod(ellps='WGS84')
    rows, cols = shape
    
    # Calculate distances to cell centers
    center_distances = np.array([
        geod.inv(linac.lon, linac.lat, lon, lat)[2] / 1000  # Convert to km
        for lon, lat in zip(center_lons, center_lats)
    ]).reshape(shape)
    
    # Calculate distances to cell corners
    corner_distances = np.array([
        geod.inv(linac.lon, linac.lat, lon, lat)[2] / 1000  # Convert to km
        for lon, lat in zip(corner_lons, corner_lats)
    ]).reshape((rows + 1, cols + 1))
    
    # For each cell, use the minimum distance to any corner or center
    min_distances = np.zeros_like(center_distances)
    for i in range(rows):
        for j in range(cols):
            cell_corners = [
                corner_distances[i, j],
                corner_distances[i, j+1],
                corner_distances[i+1, j],
                corner_distances[i+1, j+1]
            ]
            min_distances[i, j] = min(center_distances[i, j], *cell_corners)
    
    return min_distances

def calculate_combined_probability(
    treatable_cancer: np.ndarray,
    linac_facilities: List[LinacFacility],
    population_raster_path: str | Path,
    lambda_km: float,
    max_distance_km: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate combined probability of treatment from all LINACs.
    Uses P_total = 1 - ∏(1 - P_i) where P_i = exp(-d_i/λ) for each LINAC i.
    Returns both the combined probability and individual probabilities per LINAC.
    """
    rows, cols = treatable_cancer.shape
    
    # Get grid coordinates
    with rasterio.open(population_raster_path) as src:
        transform = src.transform
    center_lons, center_lats, corner_lons, corner_lats = get_grid_coordinates(transform, (rows, cols))
    
    # Initialize combined probability array
    combined_probability = np.zeros((rows, cols))
    individual_probabilities = np.zeros((len(linac_facilities), rows, cols))
    
    # Calculate individual probabilities for each LINAC
    for linac_idx, linac in enumerate(linac_facilities):
        # Calculate distances to all grid cells
        distances = calculate_distances(linac, center_lons, center_lats, corner_lons, corner_lats, (rows, cols))
        print(f"Processing LINAC  {linac_idx} of {len(linac_facilities)}", flush=True)
        
        # Calculate individual probability for this LINAC
        for i in range(rows):
            for j in range(cols):
                if distances[i, j] <= max_distance_km:
                    # P_i = exp(-d_i/λ)
                    prob = distance_decay(distances[i, j], lambda_km)
                    individual_probabilities[linac_idx, i, j] = prob
    
    # Calculate combined probability: P_total = 1 - ∏(1 - P_i)
    for i in range(rows):
        for j in range(cols):
            # Product of (1 - P_i) for all LINACs
            product = 1.0
            for linac_idx in range(len(linac_facilities)):
                prob = individual_probabilities[linac_idx, i, j]
                if prob > 0:
                    product *= (1 - prob)
            # Combined probability
            combined_probability[i, j] = 1 - product
    
    return combined_probability, individual_probabilities

def allocate_patients(
    treatable_cancer: np.ndarray,
    linac_facilities: List[LinacFacility],
    population_raster_path: str | Path,
    lambda_km: float,
    max_distance_km: float = 100.0
) -> Tuple[np.ndarray, List[LinacFacility]]:
    """
    Allocate patients to LINACs based on distance and capacity constraints.
    Uses improved model: P_total = 1 - ∏(1 - P_i) where P_i = exp(-d_i/λ) for each LINAC i.
    """
    rows, cols = treatable_cancer.shape
    unallocated = treatable_cancer.copy()
    allocated = np.zeros_like(treatable_cancer)
    
    # Calculate combined probability from all LINACs
    combined_probability, individual_probabilities = calculate_combined_probability(
        treatable_cancer, linac_facilities, population_raster_path, lambda_km, max_distance_km
    )
    
    # Get grid coordinates
    with rasterio.open(population_raster_path) as src:
        transform = src.transform
    center_lons, center_lats, corner_lons, corner_lats = get_grid_coordinates(transform, (rows, cols))
    
    # Process each LINAC
    total_linacs = len(linac_facilities)
    for linac_idx, linac in enumerate(linac_facilities):
        print(f"Processing LINAC {linac_idx+1}/{total_linacs} at ({linac.lat:.4f}, {linac.lon:.4f})")
        if linac.remaining_capacity <= 0:
            continue
            
        # Calculate distances to all grid cells
        distances = calculate_distances(linac, center_lons, center_lats, corner_lons, corner_lats, (rows, cols))
        
        # Create priority queue of grid cells by distance
        # (distance, row, col, patients)
        queue = []
        for i in range(rows):
            for j in range(cols):
                if distances[i, j] <= max_distance_km and unallocated[i, j] > 0:
                    # Use individual probability for this LINAC
                    prob = individual_probabilities[linac_idx, i, j]
                    # Number of patients that would access this LINAC
                    patients = unallocated[i, j] * prob
                    if patients > 0:  # Include any cells with cancer cases
                        heapq.heappush(queue, (distances[i, j], i, j, patients))
        
        # Track processed cells to avoid infinite loops
        processed_cells = set()
        
        # Allocate patients to this LINAC
        while queue and linac.remaining_capacity > 0:
            _, i, j, patients = heapq.heappop(queue)
            
            # Skip if we've already processed this cell for this LINAC
            if (i, j) in processed_cells:
                continue
                
            processed_cells.add((i, j))
            # hello there
            # Calculate how many patients this LINAC can take
            patients_to_treat = min(patients, linac.remaining_capacity)
            
            if patients_to_treat > 0:
                # Update allocations
                allocated[i, j] += patients_to_treat
                unallocated[i, j] -= patients_to_treat
                linac.remaining_capacity -= patients_to_treat
                linac.treated_patients[(i, j)] = patients_to_treat
                
                # If there are still unallocated patients, only put back in queue
                # if the remaining number is significant
                if unallocated[i, j] > 0:  # Lower threshold to be consistent
                    prob = individual_probabilities[linac_idx, i, j]
                    remaining_patients = unallocated[i, j] * prob
                    if remaining_patients > 0:  # Lower threshold to be consistent
                        heapq.heappush(queue, (distances[i, j], i, j, remaining_patients))
    
    return unallocated, linac_facilities

def calculate_accessibility(
    population_raster_path: str | Path,
    linac_excel_path: str | Path,
    output_path: str | Path = None,
    cancer_incidence_per_year: int = 128000,
    fraction_treatable: float = 0.5,
    patients_per_linac_per_year: int = 600,
    lambda_km: float = 30.0
) -> Tuple[np.ndarray, Dict]:
    """
    Calculate accessibility to cancer treatment facilities using a capacity-constrained
    allocation model with P(d) = exp(-d/λ) distance decay.
    """
    # Calculate treatable cancer density
    treatable_cancer, transform, crs = calculate_treatable_cancer_density(
        population_raster_path,
        cancer_incidence_per_year,
        fraction_treatable
    )
    
    # Calculate total population from the original population raster
    with rasterio.open(population_raster_path) as src:
        population = src.read(1)
    total_population = float(np.sum(population[population > 0]))
    
    # Load and initialize LINAC facilities
    from .excel_utils import read_linac_excel
    df = read_linac_excel(linac_excel_path)
    linac_facilities = []
    for _, row in df.iterrows():
        if pd.isna(row['Coordinates']):
            continue
        try:
            lat, lon = [float(x.strip()) for x in row['Coordinates'].split(',')]
            linacs = row.get('He Photon And Electron Beam Rt', 0)
            if pd.isna(linacs):
                linacs = 0
            if linacs > 0:
                facility = LinacFacility(
                    name=row.get('Operator Name', 'Unknown'),
                    lon=lon,
                    lat=lat,
                    capacity=linacs * patients_per_linac_per_year,
                    remaining_capacity=linacs * patients_per_linac_per_year,
                    treated_patients={}
                )
                linac_facilities.append(facility)
        except (ValueError, AttributeError):
            continue
    print(f"Loaded {len(linac_facilities)} LINACs from {linac_excel_path}")
    
    # Calculate total capacity and needed capacity
    total_capacity = sum(f.capacity for f in linac_facilities)
    needed_capacity = cancer_incidence_per_year * fraction_treatable
    
    # Calculate max_distance_km as 5 times the distance decay parameter
    max_distance_km = 5 * lambda_km
    
    # Allocate patients to LINACs
    unallocated, updated_linacs = allocate_patients(
        treatable_cancer,
        linac_facilities,
        population_raster_path,
        lambda_km,
        max_distance_km
    )
    
    # Calculate allocated cases (treated cases)
    allocated = treatable_cancer - unallocated
    allocated = np.where(allocated < 0, 0, allocated)
    
    # Print number of treated patients per LINAC
    print("Treated patients per LINAC:")
    for linac in updated_linacs:
        patients_treated = linac.capacity - linac.remaining_capacity
        print(f"  {linac.name}: {patients_treated} treated")
    
    # Calculate statistics
    total_treated = np.sum(allocated)
    stats = {
        'total_population': total_population,
        'total_linacs': len(linac_facilities),
        'total_capacity': total_capacity,
        'needed_capacity': needed_capacity,
        'capacity_gap': needed_capacity - total_capacity,
        'capacity_utilization': total_treated / total_capacity if total_capacity > 0 else 0,
        'total_treated': float(total_treated),
        'total_unallocated': float(np.sum(unallocated)),
        'lambda': lambda_km,  # Include lambda in stats
        'facility_stats': {
            f.name: {
                'capacity': f.capacity,
                'utilization': (f.capacity - f.remaining_capacity) / f.capacity,
                'patients_treated': f.capacity - f.remaining_capacity,
                'grid_cells_served': len(f.treated_patients),
                'location': (f.lat, f.lon)  # Include facility location
            }
            for f in updated_linacs
        }
    }
    
    # Save result if output path provided
    if output_path:
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=treatable_cancer.shape[0],
            width=treatable_cancer.shape[1],
            count=1,
            dtype=treatable_cancer.dtype,
            crs=crs,
            transform=transform,
            nodata=-9999
        ) as dst:
            # Save unallocated patients as the accessibility measure
            # (higher values indicate worse access)
            dst.write(unallocated, 1)
        logger.info(f"Saved accessibility raster to {output_path}")
        
        # Also save allocated cases (treated cases)
        treated_output_path = str(output_path).replace('.tif', '_treated.tif')
        with rasterio.open(
            treated_output_path,
            'w',
            driver='GTiff',
            height=treatable_cancer.shape[0],
            width=treatable_cancer.shape[1],
            count=1,
            dtype=treatable_cancer.dtype,
            crs=crs,
            transform=transform,
            nodata=-9999
        ) as dst:
            # Save allocated patients (treated cases)
            dst.write(allocated, 1)
        logger.info(f"Saved treated cases raster to {treated_output_path}")
    
    return unallocated, stats

if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).resolve().parents[2] / "data"
    population_raster = data_dir / "processed" / "nga_pd_2020_10km_UNadj.tif"  # Use 10km coarse resolution
    linac_file = data_dir / "linac" / "Nigeria_DIRAC.xlsx"
    output_raster = data_dir / "processed" / "nga_cancer_accessibility_10km.tif"
    
    accessibility, stats = calculate_accessibility(
        population_raster,
        linac_file,
        output_path=output_raster,
        cancer_incidence_per_year=128000,
        fraction_treatable=0.5,
        patients_per_linac_per_year=600
    )
    
    # Print detailed statistics
    logger.info("\nDetailed Statistics:")
    logger.info(f"Total population: {stats['total_population']:,.0f}")
    logger.info(f"Total LINACs: {stats['total_linacs']}")
    logger.info(f"Total treatment capacity: {stats['total_capacity']:,.0f} patients/year")
    logger.info(f"Needed capacity: {stats['needed_capacity']:,.0f} patients/year")
    logger.info(f"Capacity gap: {stats['capacity_gap']:,.0f} patients/year")
    logger.info(f"Capacity utilization: {stats['capacity_utilization']:.1%}")
    logger.info(f"Total patients treated: {stats['total_treated']:,.0f}")
    logger.info(f"Total unallocated patients: {stats['total_unallocated']:,.0f}")
    logger.info(f"Distance decay parameter (λ): {stats['lambda']} km")
    
    logger.info("\nFacility Statistics:")
    for facility, facility_stats in stats['facility_stats'].items():
        logger.info(f"\n{facility}:")
        logger.info(f"  Location: {facility_stats['location']}")
        logger.info(f"  Capacity: {facility_stats['capacity']:,.0f} patients/year")
        logger.info(f"  Utilization: {facility_stats['utilization']:.1%}")
        logger.info(f"  Patients treated: {facility_stats['patients_treated']:,.0f}")
        logger.info(f"  Grid cells served: {facility_stats['grid_cells_served']:,}") 