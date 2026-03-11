"""
H3-native radiotherapy accessibility calculation.

For each H3 hexagon the probability of a patient accessing treatment is
computed using the same exponential distance-decay model used in the
raster-based pipeline:

    P_total = 1 - ∏_i (1 - exp(-d_i / λ))

where d_i is the geodesic distance (km) from the hexagon centroid to
the i-th LINAC facility and λ is the distance-decay parameter.

Main function
-------------
compute_h3_accessibility(...)
    Given a loaded H3 GeoDataFrame and a list of LINAC locations, adds
    probability and population-weighted-access columns to the GeoDataFrame
    and returns it together with summary statistics.

generate_accessibility_map_h3(...)
    High-level wrapper used by the GUI: loads data, computes access,
    saves a GeoPackage, renders a PNG, and returns a result dict.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import h3
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from pyproj import Geod

from H3.download_h3_test import load_h3_population


# --------------------------------------------------------------------------- #
#  LINAC loading                                                                #
# --------------------------------------------------------------------------- #

def load_linac_locations(linac_excel_path: str) -> List[Tuple[float, float, float]]:
    """
    Parse a DIRAC-format Excel file and return a list of (lat, lon, weight).

    *weight* is the number of linac machines at that facility; facilities
    with zero machines are dropped.

    Supports two coordinate formats in the Excel:
      - A 'Coordinates' column containing "lat, lon" strings.
      - Separate 'Latitude' and 'Longitude' columns (case-insensitive).
    """
    if not os.path.exists(linac_excel_path):
        raise FileNotFoundError(f"LINAC Excel file not found: {linac_excel_path}")

    df = pd.read_excel(linac_excel_path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    coord_col = cols.get("coordinates")
    lat_col = cols.get("latitude")
    lon_col = cols.get("longitude")
    weight_col = (
        cols.get("he photon and electron beam rt")
        or cols.get("linacs")
        or cols.get("count")
    )

    pts: List[Tuple[float, float, float]] = []
    for _, row in df.iterrows():
        try:
            if coord_col:
                val = row[coord_col]
                if not isinstance(val, str):
                    continue
                lat_s, lon_s = [x.strip() for x in val.split(",")]
                lat, lon = float(lat_s), float(lon_s)
            elif lat_col and lon_col:
                lat, lon = float(row[lat_col]), float(row[lon_col])
            else:
                continue

            w = float(row[weight_col]) if (weight_col and pd.notna(row[weight_col])) else 1.0
            if w > 0:
                pts.append((lat, lon, w))
        except Exception:
            continue

    if not pts:
        raise ValueError(f"No valid LINAC locations found in {linac_excel_path}")

    return pts


# --------------------------------------------------------------------------- #
#  Core computation                                                             #
# --------------------------------------------------------------------------- #

def compute_h3_accessibility(
    gdf: gpd.GeoDataFrame,
    linac_locations: List[Tuple[float, float, float]],
    lambda_km: float = 30.0,
    cutoff_km: Optional[float] = None,
    use_weights: bool = True,
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Add accessibility probability columns to an H3 GeoDataFrame.

    For each hexagon, the centroid is used as the representative point.
    The combined probability from all facilities is:

        P_total = 1 - ∏_i (1 - w_i * exp(-d_i / λ))

    where w_i is the weight (number of linacs) for facility i when
    use_weights=True, otherwise w_i=1 for all facilities.

    Parameters
    ----------
    gdf : GeoDataFrame
        H3 population GeoDataFrame with at minimum 'h3' and 'population' columns.
    linac_locations : list of (lat, lon, weight)
        LINAC facility positions and weights.
    lambda_km : float
        Distance-decay parameter in km.
    cutoff_km : float, optional
        Distances beyond this are treated as zero probability.
        Defaults to 5 * lambda_km.
    use_weights : bool
        Whether to scale each facility's probability by its linac count.

    Returns
    -------
    gdf_out : GeoDataFrame
        Input GeoDataFrame with added columns:
            'centroid_lat', 'centroid_lon' : hex centroid coordinates
            'nearest_linac_km'             : distance to nearest facility
            'access_probability'           : combined P_total in [0, 1]
            'pop_with_access'              : population * access_probability
    stats : dict
        Summary statistics for the result.
    """
    if cutoff_km is None:
        cutoff_km = 5.0 * lambda_km

    geod = Geod(ellps="WGS84")
    g = gdf.copy()

    # Extract hex centroids from the H3 index (lat, lon)
    centroids = g["h3"].apply(lambda h: h3.cell_to_latlng(h))
    g["centroid_lat"] = centroids.apply(lambda c: c[0])
    g["centroid_lon"] = centroids.apply(lambda c: c[1])

    lats = g["centroid_lat"].to_numpy(dtype=np.float64)
    lons = g["centroid_lon"].to_numpy(dtype=np.float64)
    n_cells = len(g)

    # Accumulate (1 - p_i) product across all facilities
    product_complement = np.ones(n_cells, dtype=np.float64)
    nearest_km = np.full(n_cells, np.inf, dtype=np.float64)

    for lat_f, lon_f, w in linac_locations:
        # Vectorised geodesic distances
        _, _, dists_m = geod.inv(
            np.full(n_cells, lon_f),
            np.full(n_cells, lat_f),
            lons,
            lats,
        )
        dists_km = dists_m * 1e-3

        np.minimum(nearest_km, dists_km, out=nearest_km)

        p = np.exp(-dists_km / lambda_km)
        p = np.where(dists_km <= cutoff_km, p, 0.0)

        weight = w if use_weights else 1.0
        product_complement *= np.power(np.maximum(1.0 - p, 0.0), weight)

    prob = 1.0 - product_complement
    g["access_probability"] = prob.astype(np.float32)

    nearest_km = np.where(np.isinf(nearest_km), np.nan, nearest_km)
    g["nearest_linac_km"] = nearest_km.astype(np.float32)

    pop = pd.to_numeric(g["population"], errors="coerce").to_numpy(dtype=np.float64)
    pop = np.where(pop > 0, pop, 0.0)
    g["pop_with_access"] = (prob * pop).astype(np.float32)

    # Summary statistics
    total_pop = float(np.nansum(pop))
    pop_with_access = float(np.nansum(prob * pop))
    p_mean = pop_with_access / total_pop if total_pop > 0 else 0.0

    stats = {
        "n_facilities": len(linac_locations),
        "lambda_km": lambda_km,
        "cutoff_km": cutoff_km,
        "total_population": total_pop,
        "pop_with_access": pop_with_access,
        "mean_access_probability": p_mean,
        "n_hexagons": n_cells,
    }

    return g, stats


# --------------------------------------------------------------------------- #
#  Rendering                                                                    #
# --------------------------------------------------------------------------- #

def _render_access_png(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "viridis",
    dpi: int = 150,
) -> bytes:
    """Render an H3 GeoDataFrame column to a PNG and return the bytes."""
    import matplotlib
    matplotlib.use("Agg")

    vals = gdf[value_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.assign(**{value_col: vals}).plot(
        column=value_col,
        ax=ax,
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        missing_kwds={"color": "lightgrey"},
    )
    sm = cm.ScalarMappable(
        norm=Normalize(vmin=vmin, vmax=vmax), cmap=plt.get_cmap(cmap_name)
    )
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    if value_col == "access_probability":
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    data = buf.getvalue()
    plt.close(fig)
    return data


# --------------------------------------------------------------------------- #
#  High-level GUI wrapper                                                       #
# --------------------------------------------------------------------------- #

def generate_accessibility_map_h3(
    country_name: str,
    linac_excel_path: str,
    h3_gpkg_path: Optional[Path] = None,
    h3_input_dir: str = "H3_zipped_pop_density_maps",
    output_dir: str = "h3_accessibility_maps",
    lambda_km: float = 30.0,
    cutoff_km: Optional[float] = None,
    value_to_plot: str = "access_probability",
    use_weights: bool = True,
    overwrite: bool = False,
    return_image: bool = True,
) -> dict:
    """
    End-to-end accessibility map generation for the GUI.

    Loads H3 population data, computes accessibility probabilities for all
    hexagons, saves a GeoPackage, renders a PNG, and returns a result dict.

    Parameters
    ----------
    country_name : str
        Human-readable country name.
    linac_excel_path : str
        Path to DIRAC-format Excel file with LINAC locations.
    h3_gpkg_path : Path, optional
        Direct path to a decompressed .gpkg file; skips h3_input_dir.
    h3_input_dir : str
        Directory containing .gpkg / .gpkg.gz population files.
    output_dir : str
        Directory in which to save outputs.
    lambda_km : float
        Distance-decay parameter λ (km).
    cutoff_km : float, optional
        Maximum distance considered (defaults to 5 × lambda_km).
    value_to_plot : str
        One of 'access_probability' or 'pop_with_access'.
    use_weights : bool
        Scale facility probabilities by their linac count.
    overwrite : bool
        Overwrite existing output files.
    return_image : bool
        Include PNG bytes in the result dict.

    Returns
    -------
    dict with keys:
        gpkg_path (Path)          : path to the saved GeoPackage
        image_bytes (bytes | None): PNG bytes if return_image=True
        stats (dict)              : summary statistics
    """
    import pycountry

    try:
        alpha_3 = pycountry.countries.lookup(country_name).alpha_3.lower()
    except LookupError:
        alpha_3 = country_name[:3].lower()

    lam_tag = int(round(lambda_km))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_gpkg = Path(output_dir) / f"{alpha_3}_{lam_tag}km_access_h3.gpkg"
    output_png = Path(output_dir) / f"{alpha_3}_{lam_tag}km_{value_to_plot}_h3.png"

    if not overwrite and output_gpkg.exists():
        image_bytes = None
        if return_image and output_png.exists():
            with open(output_png, "rb") as fh:
                image_bytes = fh.read()
        return {"gpkg_path": output_gpkg, "image_bytes": image_bytes, "stats": {}}

    # Load H3 population data
    if h3_gpkg_path is not None:
        gdf = gpd.read_file(h3_gpkg_path)
    else:
        gdf, _ = load_h3_population(country_name, input_dir=h3_input_dir)

    # Load LINAC locations
    linac_locations = load_linac_locations(linac_excel_path)

    # Compute accessibility
    gdf_out, stats = compute_h3_accessibility(
        gdf=gdf,
        linac_locations=linac_locations,
        lambda_km=lambda_km,
        cutoff_km=cutoff_km,
        use_weights=use_weights,
    )

    # Save GeoPackage
    gdf_out.to_file(output_gpkg, driver="GPKG")

    # Render PNG
    image_bytes = None
    if return_image:
        if value_to_plot == "access_probability":
            title = (
                f"{country_name} — Probability of Access to Radiotherapy\n"
                f"(λ={lambda_km:.0f} km, N facilities={stats['n_facilities']}, "
                f"mean={stats['mean_access_probability']:.1%})"
            )
            cbar_label = "Probability of access"
            v_min, v_max = 0.0, 1.0
        else:  # pop_with_access
            title = (
                f"{country_name} — Population with Radiotherapy Access\n"
                f"(λ={lambda_km:.0f} km, N facilities={stats['n_facilities']})"
            )
            cbar_label = "Population × access probability"
            v_min = 0.0
            v_max = float(np.nanmax(gdf_out["pop_with_access"].to_numpy()))

        image_bytes = _render_access_png(
            gdf=gdf_out,
            value_col=value_to_plot,
            title=title,
            cbar_label=cbar_label,
            vmin=v_min,
            vmax=v_max,
        )
        with open(output_png, "wb") as fh:
            fh.write(image_bytes)

    return {"gpkg_path": output_gpkg, "image_bytes": image_bytes, "stats": stats}
