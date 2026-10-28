"""
Build ``inputs/globocan_xarray_2024.nc`` from the GLOBOCAN 2024 data API.

The Global Cancer Observatory population fact sheets
(https://gco.iarc.who.int/media/globocan/factsheets/populations/<M49>-<name>-fact-sheet.pdf)
are rendered from a JSON API; this script reads that API directly rather than
parsing the PDFs. Output has the same layout as ``globocan_xarray.nc``
(dims Cancer × Metric × ISO3, variable ``cancer_incidence``, new cases, both
sexes, all ages) and the same cancer names, so ``data/cancer.py`` can load
it unchanged. The 186 GLOBOCAN countries are included plus the 11 regional
aggregates the 2022 file carried (WLD, AFR, EUR, ...).

Two requests per population: one with NMSC folded into "other" (all sites)
and one with NMSC as its own row, giving "All cancers excl. NMSC" as
All cancers − NMSC. Raw responses are cached so re-runs don't hit the API.

    python scripts/build_globocan_2024.py [--cache DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

_ROOT = Path(__file__).resolve().parent.parent
_API = "https://gco-api.iarc.fr/api/globocan/v3/2024"
_FACTSHEET = _API + "/factsheet/population/{pop}/?group_CRC=1&include_nmsc=1&include_nmsc_other={other}"
_COUNTRY_GROUPINGS = {"Africa", "Asia", "Europe", "Latin America and the Caribbean",
                      "Oceania", "Northern America"}

# GLOBOCAN population id for the regional aggregates kept from the 2022 file
_REGIONS = {"WLD": 900, "AFR": 903, "EUR": 908, "OCE": 909, "EAS": 906,
            "CAR": 915, "CAM": 916, "SEA": 920, "SCA": 921, "WAS": 922, "SAM": 931}

# GLOBOCAN cancer code -> name used in globocan_xarray.nc
_CANCERS = {
    1: "Lip, oral cavity", 2: "Salivary glands", 3: "Oropharynx", 4: "Nasopharynx",
    5: "Hypopharynx", 6: "Oesophagus", 7: "Stomach", 11: "Liver", 12: "Gallbladder",
    13: "Pancreas", 14: "Larynx", 15: "Lung", 16: "Melanoma", 18: "Mesothelioma",
    19: "Kaposi sarcoma", 20: "Breast", 21: "Vulva", 22: "Vagina", 23: "Cervix uteri",
    24: "Corpus uteri", 25: "Ovary", 26: "Penis", 27: "Prostate", 28: "Testis",
    29: "Kidney", 30: "Bladder", 31: "Brain CNS", 32: "Thyroid", 33: "Hodgkin lymphoma",
    34: "NHL", 35: "Multiple myeloma", 36: "Leukaemia", 39: "All cancers", 41: "Colorectum",
}
_ALL, _NMSC = 39, 17
_EXCL_NMSC = "All cancers excl. NMSC"


def _get(url: str, retries: int = 6):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(30 * (attempt + 1))  # rate-limited


def _fetch(pop: int, other: int, cache: Path) -> list:
    f = cache / f"{pop}_{other}.json"
    if not f.exists():
        data = _get(_FACTSHEET.format(pop=pop, other=other))
        if not data.get("dataset"):
            raise RuntimeError(f"population {pop}: empty response ({data.get('error')})")
        f.write_text(json.dumps(data))
        time.sleep(3)
    return json.loads(f.read_text())["dataset"]


def _incidence(rows: list) -> dict:
    """New cases, both sexes, keyed by cancer code."""
    return {r["cancer_code"]: float(r["total"]) for r in rows if r["type"] == 0 and r["sex"] == 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "globocan_2024_api")
    ap.add_argument("--out", type=Path, default=_ROOT / "inputs" / "globocan_xarray_2024.nc")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    meta = _get(_API + "/meta/populations/all/")
    pops = {p["country_iso3"]: p["country"] for p in meta if p["grouping"] in _COUNTRY_GROUPINGS}
    assert len(pops) == 186, f"expected 186 GLOBOCAN 2024 countries, got {len(pops)}"
    pops.update(_REGIONS)

    cancers = sorted(set(_CANCERS.values()) | {_EXCL_NMSC})
    isos = sorted(pops)
    arr = np.full((len(cancers), 1, len(isos)), np.nan, dtype=np.float32)
    for j, iso in enumerate(isos):
        inc = _incidence(_fetch(pops[iso], 1, args.cache))
        nmsc = _incidence(_fetch(pops[iso], 0, args.cache))[_NMSC]
        missing = set(_CANCERS) - set(inc)
        if missing:
            print(f"  {iso}: no data for {sorted(_CANCERS[c] for c in missing)}")
        for code, name in _CANCERS.items():
            if code in inc:
                arr[cancers.index(name), 0, j] = inc[code]
        arr[cancers.index(_EXCL_NMSC), 0, j] = inc[_ALL] - nmsc

    da = xr.DataArray(
        arr, name="cancer_incidence",
        dims=["Cancer", "Metric", "ISO3"],
        coords={"Cancer": np.array(cancers, dtype=object),
                "Metric": np.array(["New_Cases_Number"], dtype=object),
                "ISO3": np.array(isos, dtype=object)},
        attrs={"source": "GLOBOCAN 2024 (IARC Global Cancer Observatory, Cancer Today v1.0)",
               "api": _API, "accessed": time.strftime("%Y-%m-%d")},
    )
    da.to_dataset().to_netcdf(args.out)
    print(f"Wrote {args.out}: {len(cancers)} cancers × {len(isos)} populations "
          f"({len(isos) - len(_REGIONS)} countries + {len(_REGIONS)} regions)")


if __name__ == "__main__":
    main()
