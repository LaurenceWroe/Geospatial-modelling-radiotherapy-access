"""
Refresh / validate the bundled input datasets and record their provenance in
``inputs/manifest.json``.

Sub-commands
------------
manifest            Print the current manifest.
kontur [--force]    Re-download the Kontur population file for every country
                    already cached locally (or for --countries ...). Kontur
                    publishes dated releases; set --version to switch release.
dirac --csv PATH    Validate a fresh DIRAC export, report rows lacking
                    coordinates / machine counts, write it to
                    inputs/dirac/Database_DIRAC_<year>.csv and refresh the
                    manifest. Geocoding is a separate step
                    (scripts/geocode_dirac.py and the scripts it lists).
globocan --csv PATH Rebuild the active GLOBOCAN file (data.cancer.XARRAY_PATH) from a GLOBOCAN
                    'Cancer Today' CSV export. Column names can be overridden
                    with --iso-col / --cancer-col / --cases-col.

GLOBOCAN and DIRAC have no public bulk API, so those two steps start from a
manual export; the script makes the rebuild reproducible and records what was
loaded. Run from the repository root:

    python scripts/refresh_data.py manifest
    python scripts/refresh_data.py kontur --countries GB NG
    python scripts/refresh_data.py dirac --csv ~/Downloads/dirac_export.csv
    python scripts/refresh_data.py globocan --csv ~/Downloads/globocan_2024.csv
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = _ROOT / "inputs" / "manifest.json"

_DIRAC_REQUIRED = [
    "Country", "City", "Operator Name", "He Photon And Electron Beam Rt",
    "Proton Ion Therapy", "XRay Generator", "Brachy Therapy Inc El",
    "Last Update", "Latitude", "Longitude",
]


def _load_manifest() -> dict:
    if _MANIFEST.exists():
        return json.loads(_MANIFEST.read_text())
    return {"datasets": {}}


def _save_manifest(m: dict) -> None:
    m["last_updated"] = _dt.date.today().isoformat()
    _MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest written: {_MANIFEST.relative_to(_ROOT)}")


def cmd_manifest(_args) -> None:
    print(json.dumps(_load_manifest(), indent=2, ensure_ascii=False))


def cmd_kontur(args) -> None:
    import requests
    from data import population as pop

    version = args.version
    zipped = pop.ZIPPED_DIR
    zipped.mkdir(parents=True, exist_ok=True)
    if args.countries:
        alpha2s = [c.upper() for c in args.countries]
    else:
        alpha2s = sorted(p.name.split("_")[0] for p in zipped.glob("*_H3_population_density_map.gpkg.gz"))
    if not alpha2s:
        print("no cached countries found; pass --countries")
        return
    print(f"refreshing {len(alpha2s)} Kontur files (release {version})")
    failed = []
    for a2 in alpha2s:
        dest = zipped / f"{a2}_H3_population_density_map.gpkg.gz"
        if dest.exists() and not args.force:
            print(f"  {a2}: exists (use --force to re-download)")
            continue
        url = f"{pop.BASE_URL}/kontur_population_{a2}_{version}.gpkg.gz"
        try:
            r = requests.get(url, stream=True, timeout=180)
            r.raise_for_status()
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            tmp.replace(dest)
            print(f"  {a2}: ok ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001
            failed.append((a2, str(e)))
            print(f"  {a2}: FAILED {e}")
    # Any change to the population inputs invalidates derived caches.
    for d in (_ROOT / "H3_pop_density_maps", _ROOT / "H3_region_cache"):
        if d.exists() and args.force:
            print(f"  note: derived cache {d.name}/ should be cleared to pick up new data")
    m = _load_manifest()
    m["datasets"].setdefault("population", {})
    m["datasets"]["population"].update({"version": version, "accessed": _dt.date.today().isoformat()})
    _save_manifest(m)
    if failed:
        print(f"{len(failed)} downloads failed")


def cmd_dirac(args) -> None:
    import pandas as pd

    src = Path(args.csv).expanduser()
    df = pd.read_csv(src)
    missing = [c for c in _DIRAC_REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"export is missing required columns: {missing}")

    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lon = pd.to_numeric(df["Longitude"], errors="coerce")
    has_coord = lat.notna() & lon.notna() & lat.between(-90, 90) & lon.between(-180, 180)
    machines = pd.to_numeric(df["He Photon And Electron Beam Rt"], errors="coerce")
    n_rows = len(df)
    n_photon = int((machines.fillna(0) > 0).sum())
    n_machines = int(machines.fillna(0).sum())
    n_nocoord = int((~has_coord).sum())
    print(f"rows: {n_rows}  rows with photon/electron machines: {n_photon}  machines: {n_machines}")
    print(f"rows without usable coordinates: {n_nocoord}  blank machine counts: {int(machines.isna().sum())}")
    years = pd.to_numeric(df["Last Update"], errors="coerce").dropna()
    if len(years):
        print(f"Last Update: min {int(years.min())}, median {years.median():.0f}, max {int(years.max())}")

    todo = df[~has_coord][["Country", "City", "Operator Name"]]
    todo_path = _ROOT / "inputs" / "dirac" / "geocode_todo.csv"
    todo.to_csv(todo_path, index=False)
    print(f"rows needing geocoding written to {todo_path.relative_to(_ROOT)}")

    dest = _ROOT / "inputs" / "dirac" / f"Database_DIRAC_{_dt.date.today():%Y}.csv"
    if dest.exists():
        bak = dest.with_name(f"Database_DIRAC_{_dt.date.today():%Y%m%d}.bak.csv")
        shutil.copy(dest, bak)
        print(f"previous export backed up to {bak.name}")
    shutil.copy(src, dest)
    print(f"NOTE: the app reads the geocoded file (data.linacs.DIRAC_CSV); run scripts/geocode_dirac.py "
          f"--input {dest.relative_to(_ROOT)} and the follow-up scripts to produce it.")

    m = _load_manifest()
    m["datasets"].setdefault("facilities", {})
    m["datasets"]["facilities"].update({
        "version": f"export of {_dt.date.today():%B %Y}",
        "accessed": _dt.date.today().isoformat(),
        "rows": n_rows,
        "rows_with_photon_electron_machines": n_photon,
        "machines_total": n_machines,
        "rows_without_coordinates_at_import": n_nocoord,
    })
    _save_manifest(m)


def cmd_globocan(args) -> None:
    import numpy as np
    import pandas as pd
    import xarray as xr

    src = Path(args.csv).expanduser()
    df = pd.read_csv(src)
    cols = {c.lower(): c for c in df.columns}

    def pick(explicit, *cands):
        if explicit:
            return explicit
        for c in cands:
            if c in cols:
                return cols[c]
        sys.exit(f"could not find a column among {cands}; pass it explicitly")

    iso_col = pick(args.iso_col, "alpha‑3 code", "alpha-3 code", "iso3", "iso code", "iso")
    cancer_col = pick(args.cancer_col, "cancer label", "cancer", "label")
    cases_col = pick(args.cases_col, "total", "number", "new cases", "cases", "value")
    print(f"using columns iso={iso_col!r} cancer={cancer_col!r} cases={cases_col!r}")

    sub = df[[iso_col, cancer_col, cases_col]].dropna()
    sub[cases_col] = pd.to_numeric(sub[cases_col], errors="coerce")
    sub = sub.dropna()
    pivot = sub.pivot_table(index=cancer_col, columns=iso_col, values=cases_col, aggfunc="sum")
    da = xr.DataArray(
        pivot.to_numpy()[:, None, :].astype(np.float64),
        dims=("Cancer", "Metric", "ISO3"),
        coords={"Cancer": pivot.index.astype(str).tolist(),
                "Metric": ["New_Cases_Number"],
                "ISO3": pivot.columns.astype(str).tolist()},
        name="globocan",
    )
    from data.cancer import XARRAY_PATH
    dest = XARRAY_PATH
    if dest.exists():
        bak = dest.with_name(f"{dest.stem}_{_dt.date.today():%Y%m%d}.bak.nc")
        shutil.copy(dest, bak)
        print(f"previous file backed up to {bak.name}")
    da.to_netcdf(dest)
    print(f"wrote {dest.relative_to(_ROOT)}: {da.sizes['Cancer']} cancer types × {da.sizes['ISO3']} countries")
    required = {"All cancers", "All cancers excl. NMSC"}
    if not required.issubset(set(da.coords["Cancer"].values)):
        print(f"WARNING: aggregate rows {required} not found; derived NMSC / Other cancers will be zero")

    m = _load_manifest()
    m["datasets"].setdefault("cancer_incidence", {})
    m["datasets"]["cancer_incidence"].update({
        "version": args.version or m["datasets"]["cancer_incidence"].get("version", ""),
        "accessed": _dt.date.today().isoformat(),
        "coverage": f"{da.sizes['Cancer']} cancer types, {da.sizes['ISO3']} countries",
    })
    _save_manifest(m)
    for d in (_ROOT / "H3_region_cache",):
        if d.exists():
            print(f"note: regional cancer caches in {d.name}/ should be cleared to pick up new incidence")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("manifest").set_defaults(func=cmd_manifest)

    k = sp.add_parser("kontur")
    k.add_argument("--countries", nargs="*", help="ISO alpha-2 codes (default: all cached)")
    k.add_argument("--version", default="20231101", help="Kontur release date tag")
    k.add_argument("--force", action="store_true")
    k.set_defaults(func=cmd_kontur)

    d = sp.add_parser("dirac")
    d.add_argument("--csv", required=True)
    d.set_defaults(func=cmd_dirac)

    g = sp.add_parser("globocan")
    g.add_argument("--csv", required=True)
    g.add_argument("--version", default="")
    g.add_argument("--iso-col"); g.add_argument("--cancer-col"); g.add_argument("--cases-col")
    g.set_defaults(func=cmd_globocan)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
