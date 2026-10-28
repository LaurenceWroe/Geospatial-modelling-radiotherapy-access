"""
Headless end-to-end smoke test of the Streamlit app with Streamlit's AppTest.

Drives the sidebar through the non-default paths (country mode, Gaussian
decay, proportional and optimal allocation, fractions-based capacity, the
Workload display metric) and both Benchmark buttons, then asserts there were
no exceptions and prints the headline stats.

Memory: loads one country (United Kingdom) at resolution 5 — about 2 GB peak.
Run it on the server, not alongside other heavy jobs:

    scripts/remote_run.sh python scripts/smoke_test_app.py
"""
from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

from streamlit.testing.v1 import AppTest  # noqa: E402


def by_label(elems, label):
    m = [e for e in elems if e.label == label]
    if not m:
        raise AssertionError(f"no widget labelled {label!r}; have {[e.label for e in elems][:30]}")
    return m[0]


def check(at, tag):
    print(f"[{tag}] exceptions={len(at.exception)}")
    for e in at.exception:
        print("   EXC:", str(e.message)[:500])
        st = getattr(e, "stack_trace", None)
        if st:
            print("   ", "\n    ".join(str(x) for x in st[-3:]))
    if at.exception:
        raise SystemExit(1)


def main() -> None:
    t0 = time.time()
    at = AppTest.from_file(os.path.join(_ROOT, "app.py"), default_timeout=1800)
    at.run()
    check(at, "default World render")

    by_label(at.radio, "Analyse by").set_value("Country"); at.run()
    cb = by_label(at.selectbox, "Country"); cb.set_value("United Kingdom"); at.run()
    by_label(at.radio, "Access Decay model").set_value("gaussian"); at.run()
    by_label(at.selectbox, "Capacity allocation rule").set_value("proportional"); at.run()
    by_label(at.radio, "Capacity basis").set_value("Fractions per LINAC (case-mix adjusted)"); at.run()
    by_label(at.selectbox, "Display metric").set_value("Workload Deficit (fractions)"); at.run()
    check(at, "configured")

    by_label(at.button, "Calculate RT Access").click(); at.run()
    check(at, "calculated (gaussian / proportional / fractions capacity / workload metric)")
    s = at.session_state["_map_result"]["stats"]
    keys = ("model", "allocation", "rtu_set", "capacity_per_machine_per_year", "total_rt_demand",
            "total_rt_treated", "unused_capacity", "demand_weighted_access",
            "total_fractions_demand", "total_fractions_treated", "mean_fractions_per_course",
            "fractions_per_linac", "linac_equivalents_deficit")
    print({k: (round(s[k], 3) if isinstance(s.get(k), float) else s.get(k)) for k in keys})
    g = at.session_state["_map_result"]["gdf_out"]
    assert "rt_fractions_untreated" in g.columns, "fraction columns missing from result"
    for var in ("2sfca", "e2sfca", "3sfca", "gravity"):
        for col in (f"fca_{var}", f"fca_{var}_raw", f"fca_{var}_deficit"):
            assert col in g.columns, f"{col} missing from result: {s.get('fca_error')}"
    print("FCA columns present; capped means:",
          {v: round(float(g[f"fca_{v}"].mean()), 3) for v in ("2sfca", "e2sfca", "3sfca", "gravity")})
    print("implied deficit sums:", {v: round(float(g[f"fca_{v}_deficit"].sum())) for v in ("2sfca", "e2sfca", "3sfca", "gravity")},
          "RadMaps deficit:", round(s["total_rt_demand"] - s["total_rt_treated"]))
    by_label(at.selectbox, "Display metric").set_value("2SFCA ratio"); at.run()
    check(at, "map with 2SFCA ratio metric")
    by_label(at.selectbox, "Display metric").set_value("3SFCA implied deficit"); at.run()
    check(at, "map with 3SFCA implied deficit metric")

    by_label(at.button, "Run allocation comparison").click(); at.run()
    check(at, "benchmark: allocation rules")
    for r in at.session_state["_bm_alloc"][0]:
        print("   ", r)
    by_label(at.button, "Run floating-catchment benchmarks").click(); at.run()
    check(at, "benchmark: FCA indices")
    for r in at.session_state["_bm_fca"][0]:
        print("   ", {k: r[k] for k in ("Index", "Implied treated, min(1, ratio) (% of demand)", "Implied deficit (patients/yr)")})

    by_label(at.selectbox, "Capacity allocation rule").set_value("optimal"); at.run()
    by_label(at.button, "Calculate RT Access").click(); at.run()
    check(at, "optimal allocation")
    s = at.session_state["_map_result"]["stats"]
    print("optimal:", {k: round(s[k], 1) for k in ("total_rt_demand", "total_rt_treated", "unused_capacity")})
    print(f"OK in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
