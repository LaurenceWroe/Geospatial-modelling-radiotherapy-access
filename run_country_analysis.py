"""
Batch radiotherapy access analysis — all countries with LINAC data.

Runs compute_accessibility (step function, 200 km cut-off, H3 resolution 4)
for every country present in both the DIRAC database and GLOBOCAN, then
writes results to an Excel file.

Usage
-----
    python run_country_analysis.py [--output results.xlsx] [--capacity 450] [--res 4]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pycountry

# ---------------------------------------------------------------------------
# Make sure project root is on sys.path so local modules resolve
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.linacs import load_linacs_from_dirac_db, DIRAC_CSV
from data.population import load_population_at_resolution
from data.cancer import apportion_cancer_to_h3, get_national_cases
from analysis.accessibility import compute_accessibility

# ---------------------------------------------------------------------------
# Country-name aliases: DIRAC CSV name → pycountry-resolvable name
# ---------------------------------------------------------------------------
_DIRAC_TO_PYCOUNTRY: dict[str, str] = {
    "USA": "United States",
    "Russia": "Russian Federation",
    "South Korea": "Korea, Republic of",
    "North Korea": "Korea, Democratic People's Republic of",
    "Iran": "Iran, Islamic Republic of",
    "Syria": "Syrian Arab Republic",
    "Vietnam": "Viet Nam",
    "Bolivia": "Bolivia, Plurinational State of",
    "Venezuela": "Venezuela, Bolivarian Republic of",
    "Moldova": "Moldova, Republic of",
    "Tanzania": "Tanzania, United Republic of",
    "Taiwan": "Taiwan, Province of China",
    "Macedonia": "North Macedonia",
    "Republic of Ireland": "Ireland",
    "Reunion": "Réunion",
    "Turkey": "Türkiye",
    "Cote D\u2019Ivoire": "Côte d'Ivoire",
    "Democratic Republic of Congo": "Congo, The Democratic Republic of the",
    "Laos": "Lao People's Democratic Republic",
    "Czech Republic": "Czechia",
    "Cape Verde": "Cabo Verde",
    "East Timor": "Timor-Leste",
    "The Gambia": "Gambia",
    "Bosnia - Herzegovina": "Bosnia and Herzegovina",
    "Kingdom of Bahrain": "Bahrain",
    "Brunei": "Brunei Darussalam",
    "Curacao": "Curaçao",
}

# Cancers to include (all excl. NMSC)
_ALL_CANCERS_EXCL_NMSC = "All cancers excl. NMSC"


def _resolve_alpha2(dirac_name: str) -> str | None:
    """Return ISO-2 code for a DIRAC country name, or None if unresolvable."""
    py_name = _DIRAC_TO_PYCOUNTRY.get(dirac_name, dirac_name)
    c = pycountry.countries.get(name=py_name)
    if c is None:
        c = pycountry.countries.get(common_name=py_name)
    return c.alpha_2 if c else None


def _resolve_alpha3(dirac_name: str) -> str | None:
    py_name = _DIRAC_TO_PYCOUNTRY.get(dirac_name, dirac_name)
    c = pycountry.countries.get(name=py_name)
    if c is None:
        c = pycountry.countries.get(common_name=py_name)
    return c.alpha_3 if c else None


def _resolve_pycountry_name(dirac_name: str) -> str | None:
    """Return the official pycountry country name used by load_population_at_resolution."""
    py_name = _DIRAC_TO_PYCOUNTRY.get(dirac_name, dirac_name)
    c = pycountry.countries.get(name=py_name)
    if c is None:
        c = pycountry.countries.get(common_name=py_name)
    return c.name if c else None


def run(output_path: Path, capacity: float, h3_res: int) -> None:
    print(f"Reading DIRAC database: {DIRAC_CSV}")
    dirac_df = pd.read_csv(DIRAC_CSV)
    dirac_df["n_linacs"] = pd.to_numeric(
        dirac_df["He Photon And Electron Beam Rt"], errors="coerce"
    ).fillna(0)
    countries_with_linacs = (
        dirac_df[dirac_df["n_linacs"] > 0]["Country"]
        .unique()
        .tolist()
    )
    print(f"Countries with LINACs in DIRAC: {len(countries_with_linacs)}")

    results = []
    skipped = []

    for i, dirac_name in enumerate(sorted(countries_with_linacs), 1):
        t0 = time.time()
        prefix = f"[{i:3d}/{len(countries_with_linacs)}] {dirac_name}"

        # --- Resolve identifiers -----------------------------------------
        alpha2 = _resolve_alpha2(dirac_name)
        alpha3 = _resolve_alpha3(dirac_name)
        py_name = _resolve_pycountry_name(dirac_name)

        if alpha2 is None or alpha3 is None or py_name is None:
            print(f"{prefix} — SKIP (cannot resolve country code)")
            skipped.append((dirac_name, "unresolvable country name"))
            continue

        # --- Load population ---------------------------------------------
        try:
            gdf = load_population_at_resolution(py_name, target_resolution=h3_res)
        except Exception as e:
            print(f"{prefix} — SKIP (population: {e})")
            skipped.append((dirac_name, f"population: {e}"))
            continue

        # --- Load cancer / RT demand -------------------------------------
        try:
            national = get_national_cases(alpha3, [_ALL_CANCERS_EXCL_NMSC])
            cancer_cases = national.get(_ALL_CANCERS_EXCL_NMSC, 0.0)
            if cancer_cases <= 0:
                print(f"{prefix} — SKIP (no GLOBOCAN data)")
                skipped.append((dirac_name, "no GLOBOCAN data"))
                continue
            gdf = apportion_cancer_to_h3(gdf, alpha3, [_ALL_CANCERS_EXCL_NMSC], use_actual_rt=False)
            demand_col = f"{_ALL_CANCERS_EXCL_NMSC.lower().replace(' ', '_').replace('.', '')}_optimal_rt"
            # Find the demand column (naming convention may vary)
            demand_cols = [c for c in gdf.columns if "optimal_rt" in c]
            if demand_cols:
                import numpy as np
                demand = gdf[demand_cols].sum(axis=1).to_numpy(dtype=float)
            else:
                demand = None
        except Exception as e:
            print(f"{prefix} — SKIP (cancer: {e})")
            skipped.append((dirac_name, f"cancer: {e}"))
            continue

        # --- Load LINACs -------------------------------------------------
        try:
            linac_locs, _ = load_linacs_from_dirac_db(dirac_name)
        except Exception as e:
            print(f"{prefix} — SKIP (LINACs: {e})")
            skipped.append((dirac_name, f"linacs: {e}"))
            continue

        # --- Run accessibility -------------------------------------------
        try:
            _, stats = compute_accessibility(
                gdf,
                linac_locs,
                model="step",
                max_distance_km=200.0,
                capacity_per_machine_per_year=capacity,
                demand=demand,
                h3_resolution=h3_res,
            )
        except Exception as e:
            print(f"{prefix} — SKIP (compute: {e})")
            skipped.append((dirac_name, f"compute: {e}"))
            continue

        # --- Extract metrics ---------------------------------------------
        total_pop = stats["total_population"]
        total_demand = stats["total_rt_demand"]
        n_linacs = stats["total_machines"]
        national_cap = stats["total_national_capacity"]

        geo_access = stats["mean_access_probability"]
        radmaps_ratio = stats["mean_capacity_limited_probability"]
        cap_ratio = min(national_cap / total_demand, 1.0) if total_demand > 0 else 0.0

        elapsed = time.time() - t0
        print(
            f"{prefix} — LINACs={int(n_linacs)}, demand={total_demand/1e3:.1f}k, "
            f"cap={cap_ratio:.2f}, geo={geo_access:.2f}, radmaps={radmaps_ratio:.2f}  ({elapsed:.1f}s)"
        )

        results.append({
            "Country": dirac_name,
            "ISO2": alpha2,
            "ISO3": alpha3,
            "Population": round(total_pop),
            "Cancer Incidence (excl. NMSC)": round(cancer_cases),
            "RT Demand": round(total_demand),
            "n_LINACs": int(n_linacs),
            "National Capacity (pts/yr)": round(national_cap),
            # Ratios (1.0 = demand met)
            "Capacity Ratio": round(cap_ratio, 4),
            "Geography Access": round(geo_access, 4),
            "RadMaps Access": round(radmaps_ratio, 4),
            # Deficits
            "Capacity Deficit": round(1.0 - cap_ratio, 4),
            "Geography Deficit": round(1.0 - geo_access, 4),
            "RadMaps Deficit": round(1.0 - radmaps_ratio, 4),
            # Absolute treated/untreated
            "RT Treated": round(stats["total_rt_treated"]),
            "RT Untreated": round(stats["total_rt_untreated"]),
        })

    # --- Write output --------------------------------------------------------
    df_out = pd.DataFrame(results).sort_values("RadMaps Access", ascending=True)

    df_skipped = pd.DataFrame(skipped, columns=["Country", "Reason"])

    print(f"\nWriting {len(results)} countries to {output_path}")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Results", index=False)
        df_skipped.to_excel(writer, sheet_name="Skipped", index=False)

        # Auto-width columns
        for sheet_name, df in [("Results", df_out), ("Skipped", df_skipped)]:
            ws = writer.sheets[sheet_name]
            for col_idx, col in enumerate(df.columns, 1):
                max_len = max(len(str(col)), df[col].astype(str).str.len().max())
                ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = min(max_len + 2, 40)

    print(f"Done. {len(results)} countries computed, {len(skipped)} skipped.")
    if skipped:
        print("Skipped:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch RT access analysis for all countries.")
    parser.add_argument("--output", default="country_rt_access.xlsx", help="Output Excel file path")
    parser.add_argument("--capacity", type=float, default=450.0, help="Patients per machine per year")
    parser.add_argument("--res", type=int, default=4, help="H3 resolution (default 4)")
    args = parser.parse_args()

    run(Path(args.output), args.capacity, args.res)
