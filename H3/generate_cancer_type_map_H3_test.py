import os
import io
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import gzip, shutil, tempfile
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm, ListedColormap
import matplotlib.cm as cm
from io import BytesIO
from H3.download_h3_test import load_h3_population, _load_h3_gdf_from_path, _get_cases_for_cancers, _get_rt_utilisation_maps

# ------------ Utilities ------------
def _norm_key(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _apportion_over_h3(
    gdf: gpd.GeoDataFrame, total: float, population_col: str = "population"
) -> gpd.GeoDataFrame:
    g = gdf.copy()
    pop = pd.to_numeric(g[population_col], errors="coerce").astype(float)
    pop = pop.where(pop > 0, np.nan)
    total_pop = float(np.nansum(pop))
    if not np.isfinite(total_pop) or total_pop <= 0:
        raise ValueError("H3 population sum is non-positive; cannot apportion.")
    g["population_clean"] = pop
    g["apportioned"] = (g["population_clean"] / total_pop) * float(total)
    g.loc[g["population_clean"].isna(), "apportioned"] = np.nan
    return g


def _plot_h3_heatmap_to_png_bytes(
    gdf: gpd.GeoDataFrame,
    value_col: str = "apportioned",
    title: Optional[str] = None,
    vmin: float = 1.0,
    vmax: Optional[float] = None,
    cmap_name: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
) -> bytes:
    import matplotlib
    matplotlib.use("Agg")

    plot_gdf = gdf.copy()
    plot_gdf["plot_val"] = plot_gdf[value_col].where(
        np.isfinite(plot_gdf[value_col]) & (plot_gdf[value_col] > 0)
    )

    cmap = cm.get_cmap(cmap_name, 256)
    new_colors = cmap(np.linspace(0, 1, 256))
    dark_blue = np.array([0, 0, 139 / 255, 1.0])
    new_colors[0] = dark_blue
    custom_cmap = ListedColormap(new_colors)
    custom_cmap.set_under(dark_blue)

    if vmax is None:
        vmax = float(np.nanmax(plot_gdf["plot_val"])) if np.isfinite(
            np.nanmax(plot_gdf["plot_val"])
        ) else vmin

    fig, ax = plt.subplots(figsize=figsize)
    plot_gdf.plot(
        column="plot_val",
        ax=ax,
        cmap=custom_cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        linewidth=0,
        edgecolor=None,
    )
    cbar = plt.cm.ScalarMappable(norm=LogNorm(vmin=vmin, vmax=vmax), cmap=custom_cmap)
    fig.colorbar(cbar, ax=ax, label="Cases per hex")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200)
    buf.seek(0)
    data = buf.getvalue()
    plt.close(fig)
    return data


def _save_h3_geopackage(
    gdf: gpd.GeoDataFrame, out_path: str, layer_name: Optional[str] = None
) -> str:
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(outp, driver="GPKG", layer=layer_name if layer_name else "apportioned")
    return str(outp)


# ---- Helpers ---- 
def _load_h3_gdf_from_path(gpkg_path: Path):
    """Load H3 GeoDataFrame from a GeoPackage."""
    if not Path(gpkg_path).exists():
        raise FileNotFoundError(f"{gpkg_path} not found")
    return gpd.read_file(gpkg_path)

def _get_cases_for_cancers(da: xr.DataArray, cancer_types: list):
    """Return filtered xarray DataArray for selected cancers."""
    if "Cancer" not in da.coords:
        raise ValueError("DataArray missing 'Cancer' coordinate")
    return da.sel(Cancer=cancer_types)

def _get_rt_utilisation_maps(country, da: xr.DataArray, include_actual=True, include_optimal=True):
    """
    Stub: Compute RT utilisation maps for a country.
    Returns dictionary of GeoDataFrames, e.g., {"actual": gdf_actual, "optimal": gdf_optimal}.
    Replace with your logic to calculate RT utilisation per H3 cell.
    """
    # Example: return empty GeoDataFrames for now
    return {
        "actual": gpd.GeoDataFrame(),
        "optimal": gpd.GeoDataFrame()
    }

def _norm_key(name: str) -> str:
    """Normalize cancer type names for matching."""
    return name.strip().lower().replace(" ", "_")

    """
    Load optimal and actual RT utilisation fractions for cancers.
    Returns (optimal_map, actual_map).
    """
    # Optimal fractions (CSV with columns: Cancer, Fraction)
    opt_df = pd.read_csv(optimal_csv_path)
    optimal_map = {
        _norm_key(row["Cancer"]): float(row["Fraction"])
        for _, row in opt_df.iterrows()
    }

    # Actual fractions (if JSON exists per-country)
    actual_map = {}
    actual_file = Path(actual_dir) / f"{country_iso3.lower()}_rt_utilisation.json"
    if actual_file.exists():
        with open(actual_file, "r") as f:
            data = json.load(f)
        actual_map = {_norm_key(k): float(v) for k, v in data.items()}

    return optimal_map, actual_map


# --------- Main polygon-based function ----------
def generate_cancer_type_map_h3_polygons(
    country_iso3: str,
    h3_gpkg_path: Path = None,
    h3_gdf: gpd.GeoDataFrame = None,
    da=None,
    cancer_types=None,
    include_RT_utilisation=True,
    include_optimal_RT_utilisation=True,
    optimal_rt_csv_path=None,
    actual_rt_dir=None,
    include_capacity_weighted=False,
    linac_capacity=250,
    n_linacs=5,
    output_dir="cancer_type_maps_h3",
    return_image=True,
    overwrite=False,
):
    """
    Generate polygon-only cancer type map and optionally a PNG preview.
    Returns dict: {"gpkg_path": Path, "image_bytes": PNG bytes}
    """
    # Load H3 hex polygons
    if h3_gdf is None:
        if h3_gpkg_path is None:
            raise ValueError("Must provide either h3_gdf or h3_gpkg_path")
        h3_gdf = _load_h3_gdf_from_path(h3_gpkg_path)

    # Ensure output folder
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_gpkg = Path(output_dir) / f"{country_iso3}_cancer_map.gpkg"

    # Calculate cases for selected cancers
    if da is None:
        raise ValueError("Xarray DataArray (da) must be provided")
    cases_df = _get_cases_for_cancers(da, cancer_types=cancer_types, country_iso3=country_iso3)

    # Merge with H3 polygons (assumes H3 index column named 'h3')
    map_gdf = h3_gdf.merge(cases_df, how="left", left_on="h3", right_index=True)
    map_gdf.fillna(0, inplace=True)

    # Apply RT utilization if requested
    if include_RT_utilisation or include_optimal_RT_utilisation:
        rt_maps = _get_rt_utilisation_maps(
            map_gdf, cancer_types=cancer_types,
            include_actual=include_RT_utilisation,
            include_optimal=include_optimal_RT_utilisation,
            linac_capacity=linac_capacity,
            n_linacs=n_linacs,
            capacity_weighted=include_capacity_weighted,
        )
        map_gdf = rt_maps  # assume function returns modified GeoDataFrame

    # Save GeoPackage
    if not output_gpkg.exists() or overwrite:
        map_gdf.to_file(output_gpkg, driver="GPKG")

    # Prepare PNG preview
    image_bytes = None
    if return_image:
        # Use sum of all cancer columns if multiple
        value_col = cancer_types[0] if len(cancer_types) == 1 else map_gdf[cancer_types].sum(axis=1)
        values = np.array(value_col)
        values = np.where(values > 0, values, np.nan)

        # Custom colormap
        cmap = cm.get_cmap("Reds", 256)
        new_colors = cmap(np.linspace(0, 1, 256))
        dark_blue = np.array([0, 0, 139 / 255, 1.0])
        new_colors[0] = dark_blue
        custom_cmap = ListedColormap(new_colors)
        custom_cmap.set_under(dark_blue)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        map_gdf.plot(column=value_col, cmap=custom_cmap,
                     norm=LogNorm(vmin=1, vmax=np.nanmax(values)),
                     linewidth=0, ax=ax)

        cbar = plt.colorbar(cm.ScalarMappable(norm=LogNorm(vmin=1, vmax=np.nanmax(values)),
                                              cmap=custom_cmap), ax=ax)
        cbar.set_label("Cancer cases / estimated patients")
        ax.set_title(f"{country_iso3} — Cancer Map")
        ax.axis("off")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        image_bytes = buf.getvalue()

    return {"gpkg_path": output_gpkg, "image_bytes": image_bytes}
