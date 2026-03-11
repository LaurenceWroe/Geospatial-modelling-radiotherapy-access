"""
Population data loader — Kontur H3 population density GeoPackages.

Each country file is stored in H3_zipped_pop_density_maps/ as:
    {ALPHA2}_H3_population_density_map.gpkg.gz

The GeoPackage is SQLite-based and cannot be streamed; it is decompressed
to a temp file, read, then the temp file is removed.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
import h3 as _h3
import pycountry
import requests
from shapely.geometry import Polygon as _Polygon


ZIPPED_DIR = Path("H3_zipped_pop_density_maps")
BASE_URL = "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets"


def _resolve_alpha2(country_name: str) -> str:
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        raise ValueError(f"Unknown country: {country_name!r}")


def download_population(country_name: str) -> Path:
    """Download Kontur population GeoPackage for *country_name* if not cached.

    Returns the path to the local .gpkg.gz file.
    """
    alpha2 = _resolve_alpha2(country_name)
    gz_path = ZIPPED_DIR / f"{alpha2}_H3_population_density_map.gpkg.gz"
    if gz_path.exists():
        return gz_path

    ZIPPED_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/kontur_population_{alpha2}_20231101.gpkg.gz"
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(gz_path, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    return gz_path


def load_population(country_name: str) -> gpd.GeoDataFrame:
    """Load the H3 population GeoDataFrame for *country_name*.

    Columns guaranteed: ``h3`` (str), ``population`` (float), ``geometry``
    (Shapely polygon, EPSG:4326).

    Raises ``FileNotFoundError`` if no local file exists and download fails.
    """
    alpha2 = _resolve_alpha2(country_name)
    gz_path = ZIPPED_DIR / f"{alpha2}_H3_population_density_map.gpkg.gz"

    # Try unzipped gpkg first (pre-extracted by a previous run)
    gpkg_path = ZIPPED_DIR / f"{alpha2}_H3_population_density_map.gpkg"
    if gpkg_path.exists():
        gdf = gpd.read_file(gpkg_path)
        return _normalise(gdf)

    if not gz_path.exists():
        gz_path = download_population(country_name)

    tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
    try:
        with gzip.open(gz_path, "rb") as src, open(tmp.name, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gdf = gpd.read_file(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    return _normalise(gdf)


def load_population_at_resolution(
    country_name: str, target_resolution: int = 8
) -> gpd.GeoDataFrame:
    """Load population at *target_resolution* (6, 7, or 8).

    Resolution 8 is the native Kontur resolution (~0.74 km²).
    Coarser resolutions (6, 7) aggregate population into parent hexagons.
    """
    gdf = load_population(country_name)
    native_res = _h3.get_resolution(str(gdf["h3"].iloc[0]))

    if target_resolution == native_res:
        return gdf
    if target_resolution > native_res:
        raise ValueError(
            f"Cannot resample to finer resolution ({target_resolution}) "
            f"than native ({native_res})"
        )

    # Aggregate child cells into parent hexagons at target_resolution
    gdf = gdf.copy()
    gdf["parent_h3"] = gdf["h3"].apply(
        lambda h: _h3.cell_to_parent(h, target_resolution)
    )
    df_agg = gdf.groupby("parent_h3")["population"].sum().reset_index()
    df_agg.rename(columns={"parent_h3": "h3"}, inplace=True)
    df_agg["geometry"] = df_agg["h3"].apply(
        lambda h: _Polygon([(lon, lat) for lat, lon in _h3.cell_to_boundary(h)])
    )
    return gpd.GeoDataFrame(df_agg, geometry="geometry", crs="EPSG:4326")


def _normalise(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure consistent CRS (EPSG:4326) and column names."""
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    if "population" not in gdf.columns:
        # Kontur may call it 'pop' in some versions
        for candidate in ("pop", "population_count", "value"):
            if candidate in gdf.columns:
                gdf = gdf.rename(columns={candidate: "population"})
                break
    gdf["population"] = gdf["population"].fillna(0).clip(lower=0)
    return gdf[["h3", "population", "geometry"]].copy()
