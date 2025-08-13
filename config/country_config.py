"""
Country configuration system for cancer accessibility analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Resolve project directories to build absolute paths regardless of CWD
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

@dataclass
class CountryConfig:
    """Configuration for a specific country's analysis."""
    
    # Basic country info
    name: str
    code: str  # ISO country code
    
    # Data paths
    population_raster_path: Path
    linac_data_path: Path
    
    # Analysis parameters
    lambda_km: float = 30.0  # Distance decay parameter
    max_distance_km: Optional[float] = None  # Will be calculated as 5 * lambda if None
    cancer_incidence_per_year: int = 128000
    fraction_treatable: float = 0.5
    patients_per_linac_per_year: int = 600
    
    # Grid resolution
    target_resolution_km: float = 1.0  # 1km grid resolution
    
    # Output paths
    output_dir: Path = DATA_DIR / "processed" / "1km_grid"
    
    # Country-specific boundaries (optional)
    bbox: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy)
    country_bounds: Optional[Dict[str, float]] = None  # {'left', 'bottom', 'right', 'top'}
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.max_distance_km is None:
            self.max_distance_km = 5 * self.lambda_km
            
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def output_files(self) -> Dict[str, Path]:
        """Get output file paths for this country."""
        prefix = self.code.lower()
        return {
            'population_coarse': self.output_dir / f"{prefix}_pd_2020_{int(self.target_resolution_km)}km_UNadj.tif",
            'accessibility': self.output_dir / f"{prefix}_cancer_accessibility.tif",
            'treated_cases': self.output_dir / f"{prefix}_cancer_accessibility_treated.tif",
            'treated_map': self.output_dir / f"{prefix}_treated_cancer_density.png",
            'unallocated_map': self.output_dir / f"{prefix}_unallocated_cases.png",
            'unallocated_no_overlay': self.output_dir / f"{prefix}_unallocated_cases_no_overlay.png",
            'population_density_map': self.output_dir / f"{prefix}_population_density.png",
            'treatable_cancer_density_map': self.output_dir / f"{prefix}_treatable_cancer_density.png"
        }

# Predefined country configurations
NIGERIA_CONFIG = CountryConfig(
    name="Nigeria",
    code="NGA",
    population_raster_path=DATA_DIR / "raw" / "resampled" / "nga_population_1km.tif",
    linac_data_path=DATA_DIR / "linac" / "Nigeria_DIRAC.xlsx",
    lambda_km=30.0,
    cancer_incidence_per_year=128000,
    fraction_treatable=0.5,
    patients_per_linac_per_year=600,
    target_resolution_km=1.0,
    country_bounds={'left': 2.7, 'bottom': 4.3, 'right': 14.7, 'top': 13.9}
)

UK_CONFIG = CountryConfig(
    name="United Kingdom",
    code="GBR",
    population_raster_path=DATA_DIR / "raw" / "resampled" / "gbr_population_1km.tif",
    linac_data_path=DATA_DIR / "linac" / "UK_DIRAC.xlsx",
    lambda_km=30.0,
    cancer_incidence_per_year=375000,  # UK cancer incidence
    fraction_treatable=0.5,
    patients_per_linac_per_year=380,
    target_resolution_km=1.0,
    country_bounds={'left': -8.6, 'bottom': 49.5, 'right': 1.8, 'top': 60.9}
)

US_CONFIG = CountryConfig(
    name="United States",
    code="USA",
    population_raster_path=DATA_DIR / "raw" / "resampled" / "usa_population_10km.tif",
    linac_data_path=DATA_DIR / "linac" / "US_DIRAC_Walmart.xlsx",
    lambda_km=30.0,
    cancer_incidence_per_year=2380000,  # US cancer incidence (approximate)
    fraction_treatable=0.5,
    patients_per_linac_per_year=600,
    target_resolution_km=10.0,
    country_bounds={'left': -125.0, 'bottom': 24.0, 'right': -66.9, 'top': 49.4}
)

UK_10KM_CONFIG = CountryConfig(
    name="United Kingdom (10km)",
    code="GBR10",
    population_raster_path=DATA_DIR / "raw" / "resampled" / "gbr_population_10km.tif",
    linac_data_path=DATA_DIR / "linac" / "UK_DIRAC.xlsx",
    lambda_km=30.0,
    cancer_incidence_per_year=375000,  # UK cancer incidence
    fraction_treatable=0.5,
    patients_per_linac_per_year=380,
    target_resolution_km=10.0,
    country_bounds={'left': -8.6, 'bottom': 49.5, 'right': 1.8, 'top': 60.9}
)

UK_50KM_CONFIG = CountryConfig(
    name="United Kingdom (50km)",
    code="GBR50",
    population_raster_path=DATA_DIR / "raw" / "resampled" / "gbr_population_50km.tif",
    linac_data_path=DATA_DIR / "linac" / "UK_DIRAC.xlsx",
    lambda_km=30.0,
    cancer_incidence_per_year=375000,  # UK cancer incidence
    fraction_treatable=0.5,
    patients_per_linac_per_year=380,
    target_resolution_km=50.0,
    country_bounds={'left': -8.6, 'bottom': 49.5, 'right': 1.8, 'top': 60.9}
)

UK_5KM_CONFIG = CountryConfig(
    name="United Kingdom (5km)",
    code="GBR5",
    population_raster_path=DATA_DIR / "raw" / "resampled" / "gbr_population_5km.tif",
    linac_data_path=DATA_DIR / "linac" / "UK_DIRAC.xlsx",
    lambda_km=30.0,
    cancer_incidence_per_year=375000,  # UK cancer incidence
    fraction_treatable=0.5,
    patients_per_linac_per_year=380,
    target_resolution_km=5.0,
    country_bounds={'left': -8.6, 'bottom': 49.5, 'right': 1.8, 'top': 60.9}
)

US_25KM_CONFIG = CountryConfig(
    name="United States (25km)",
    code="USA25",
    population_raster_path=DATA_DIR / "raw" / "resampled" / "usa_population_25km.tif",
    linac_data_path=DATA_DIR / "linac" / "US_DIRAC_Walmart.xlsx",
    lambda_km=30.0,
    cancer_incidence_per_year=2380000,  # US cancer incidence (approximate)
    fraction_treatable=0.5,
    patients_per_linac_per_year=600,
    target_resolution_km=25.0,
    country_bounds={'left': -125.0, 'bottom': 24.0, 'right': -66.9, 'top': 49.4}
)

US_5KM_CONFIG = CountryConfig(
    name="United States (5km)",
    code="USA5",
    population_raster_path=DATA_DIR / "raw" / "resampled" / "usa_population_5km.tif",
    linac_data_path=DATA_DIR / "linac" / "US_DIRAC_Walmart.xlsx",
    lambda_km=30.0,
    cancer_incidence_per_year=2380000,  # US cancer incidence (approximate)
    fraction_treatable=0.5,
    patients_per_linac_per_year=380,
    target_resolution_km=25.0,
    country_bounds={'left': -125.0, 'bottom': 24.0, 'right': -66.9, 'top': 49.4}
)

def load_country_config(country_code: str) -> CountryConfig:
    """Load configuration for a specific country."""
    configs = {
        'NGA': NIGERIA_CONFIG,
        'GBR': UK_CONFIG,
        'USA': US_CONFIG,
        'USA5': US_5KM_CONFIG,
        'GBR10': UK_10KM_CONFIG,
        'GBR50': UK_50KM_CONFIG,
        'GBR5': UK_5KM_CONFIG,
        'USA25': US_25KM_CONFIG,
    }
    
    if country_code.upper() not in configs:
        raise ValueError(f"Country code '{country_code}' not supported. Available: {list(configs.keys())}")
    
    return configs[country_code.upper()]

def save_country_config(config: CountryConfig, filepath: Path) -> None:
    """Save country configuration to JSON file."""
    config_dict = {
        'name': config.name,
        'code': config.code,
        'population_raster_path': str(config.population_raster_path),
        'linac_data_path': str(config.linac_data_path),
        'lambda_km': config.lambda_km,
        'max_distance_km': config.max_distance_km,
        'cancer_incidence_per_year': config.cancer_incidence_per_year,
        'fraction_treatable': config.fraction_treatable,
        'patients_per_linac_per_year': config.patients_per_linac_per_year,
        'target_resolution_km': config.target_resolution_km,
        'output_dir': str(config.output_dir),
        'bbox': config.bbox
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_country_config_from_file(filepath: Path) -> CountryConfig:
    """Load country configuration from JSON file."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    return CountryConfig(
        name=config_dict['name'],
        code=config_dict['code'],
        population_raster_path=Path(config_dict['population_raster_path']),
        linac_data_path=Path(config_dict['linac_data_path']),
        lambda_km=config_dict['lambda_km'],
        max_distance_km=config_dict.get('max_distance_km'),
        cancer_incidence_per_year=config_dict['cancer_incidence_per_year'],
        fraction_treatable=config_dict['fraction_treatable'],
        patients_per_linac_per_year=config_dict['patients_per_linac_per_year'],
        target_resolution_km=config_dict['target_resolution_km'],
        output_dir=Path(config_dict['output_dir']),
        bbox=config_dict.get('bbox')
    ) 