"""
Regression test: the paper's Table 3 (six countries, H3 res 5, 2 h driving
step threshold, 450 patients/linac/yr, Delaney 2005 optimal RTU) must be
reproduced by the default engine configuration.

Reference values live in figures/table3_metrics.csv and were generated from
the engine as published (commit 5e180f8) with the July 2025 DIRAC export; the
A_RM column was regenerated on 26 Oct 2026 after the deliberate change to the
nearest-first ring rule (rings by rounded travel time), which moved Australia,
India and the USA by 0.1-1.9 k patients. The test selects that export explicitly
(the tool's default is now the October 2026 export),
and GLOBOCAN 2022 (the default is now the 2024 file). Any change to the default
nearest-first allocation, the step decay, demand apportionment, or the
cached travel-time handling that moves these numbers is a behaviour change
and must be deliberate.

The test needs the local Kontur population files and the cached TravelTime
matrices, so it is skipped when either is missing (e.g. on CI).

Run only this test:  pytest tests/test_regression_table3.py -q
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

h3 = pytest.importorskip("h3")
gpd = pytest.importorskip("geopandas")
pycountry = pytest.importorskip("pycountry")

_ROOT = Path(__file__).resolve().parent.parent
_REF = _ROOT / "figures" / "table3_metrics.csv"
_TT_CACHE = _ROOT / "travel_time_cache"
_POP_DIR = _ROOT / "H3_zipped_pop_density_maps"

RES = 5
HRS = 2
CAP = 450.0
_AGG = {"all cancers", "all cancers excl. nmsc", "all cancers excl nmsc"}


def _reference_rows():
    import csv
    with open(_REF) as fh:
        return list(csv.DictReader(fh))


def _tt_matrix(iso3, locs):
    ll = [(lat, lon) for lat, lon, _ in locs]
    lh = hashlib.md5(json.dumps(sorted(ll)).encode()).hexdigest()[:8]
    name = f"{iso3}_res{RES}_{HRS}h_{lh}_driving.npz"
    # native res-5 fetches were moved to native/ when scripts/fetch_tt_res6.py
    # replaced them with res-6 aggregates; the reference was built on the former
    p = _TT_CACHE / "native" / name
    if not p.exists():
        p = _TT_CACHE / name
    return np.load(p)["matrix"] if p.exists() else None


@pytest.mark.slow
@pytest.mark.skipif(not _REF.exists(), reason="reference table missing")
@pytest.mark.skipif(not _TT_CACHE.exists() or not _POP_DIR.exists(),
                    reason="local population / travel-time caches not available")
@pytest.mark.parametrize("row", _reference_rows() if _REF.exists() else [], ids=lambda r: r["short"])
def test_table3_reproduced(row, monkeypatch):
    from data.population import load_population_at_resolution
    from data.linacs import load_linacs_from_dirac_db, DIRAC_CSV_2025
    import data.cancer
    from data.cancer import apportion_cancer_to_h3, get_cancer_types, DERIVED_CANCER_TYPES
    monkeypatch.setattr(data.cancer, "XARRAY_PATH", data.cancer.GLOBOCAN_FILES["2022"])
    from analysis.accessibility import compute_accessibility

    obj = pycountry.countries.lookup(row["country"])
    alpha2 = obj.alpha_2
    if not (_POP_DIR / f"{alpha2}_H3_population_density_map.gpkg.gz").exists():
        pytest.skip(f"no cached population for {row['country']}")

    gdf = load_population_at_resolution(obj.name, target_resolution=RES)
    cgdf = apportion_cancer_to_h3(gdf, obj.alpha_3, get_cancer_types() + DERIVED_CANCER_TYPES,
                                  use_actual_rt=False, rtu_set="delaney_2005")
    cols = [c for c in cgdf.columns if c.endswith("_optimal_rt")
            and c[:-len("_optimal_rt")].strip().lower() not in _AGG]
    demand = cgdf[cols].sum(axis=1).clip(lower=0).to_numpy(np.float64)
    locs, _ = load_linacs_from_dirac_db(obj.name, csv_path=DIRAC_CSV_2025)
    tt = _tt_matrix(obj.alpha_3, locs)
    if tt is None:
        pytest.skip(f"no cached 2 h driving matrix for {row['country']}")

    g, s = compute_accessibility(
        gdf, locs, model="step", max_distance_km=HRS * 60,
        capacity_per_machine_per_year=CAP, demand=demand,
        h3_resolution=RES, travel_time_matrix=tt,
    )
    D = s["total_rt_demand"] / 1e3
    A_C = min(s["total_rt_demand"], s["total_machines"] * CAP) / 1e3
    A_G = float((g["access_probability"] * g["rt_demand"]).sum()) / 1e3
    A_RM = s["total_rt_treated"] / 1e3

    # Reference is printed to 0.1 k; allow rounding slack.
    assert D == pytest.approx(float(row["D"]), abs=0.06)
    assert A_C == pytest.approx(float(row["A_C"]), abs=0.06)
    assert A_G == pytest.approx(float(row["A_G"]), abs=0.06)
    assert A_RM == pytest.approx(float(row["A_RM"]), abs=0.06)
