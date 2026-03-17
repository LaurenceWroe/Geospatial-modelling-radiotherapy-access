# RT Access

An interactive web tool for modelling and visualising sub-national access to radiotherapy (RT) at country and regional scale.

**Live app → [[rt-access.streamlit.app](https://rt-access.streamlit.app)]** 

---

## Overview

Approximately half of all cancer cases require radiotherapy, yet access to RT remains critically low in many parts of the world. National-level statistics mask large within-country inequalities driven by two compounding factors:

- **Geographic access** — RT requires attendance over several weeks; patients far from a facility are substantially less likely to complete treatment.
- **Machine capacity** — the finite number of linear accelerators (linacs) limits total annual throughput.

This tool combines both constraints simultaneously, producing per-hexagon estimates of:

- Population density and cancer burden (GLOBOCAN 2022)
- RT demand (optimal utilisation rates or user-defined fraction)
- Geographic access probability (exponential, Weibull, or step-function distance-decay models)
- Capacity-limited (modelled) access via a ring-based proportional allocation algorithm

Maps are built on [H3 hexagonal grids](https://h3geo.org/) (Uber H3) at resolutions from ~400 m (country-level) down to ~87,000 km² (regional), enabling consistent multi-scale analysis.

---

## Data sources

| Data | Source |
|---|---|
| Population | [Kontur Population Dataset](https://www.kontur.io/portfolio/population-dataset/) — H3 resolution 8 (~400 m) |
| Cancer incidence | [GLOBOCAN 2022](https://gco.iarc.fr/today/) (IARC) — 175 countries |
| Optimal RT utilisation | Delaney *et al.* 2005 — site-specific RT fractions |
| LINAC locations & counts | [DIRAC database](https://dirac.iaea.org/) (IAEA) |

---

## Getting started

### Requirements

Python 3.9+ with dependencies listed in `requirements.txt`. Key packages: `streamlit`, `geopandas`, `h3`, `pydeck`, `xarray`, `pyogrio`.

```bash
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

On first run for a new country, population data is downloaded from the Kontur S3 bucket (~1–60 seconds depending on country size). Subsequent loads are cached.

---

## Project structure

```
app.py                        # Streamlit application
analysis/
  accessibility.py            # H3 accessibility computation (geographic + capacity-limited)
data/
  cancer.py                   # GLOBOCAN loader and H3 apportionment
  linacs.py                   # DIRAC LINAC database loader
  population.py               # Kontur population loader and resampler
  regions.py                  # Region definitions (Africa, Europe, etc.)
cancer_data/
  globocan_xarray.nc          # GLOBOCAN 2022 data (Cancer × Metric × ISO3)
  optimal_rt_utilisations.csv # Site-specific RT utilisation fractions
linac_data/
  Database_DIRAC_fixed.csv    # DIRAC LINAC database (geocoding-corrected)
assets/
  flowchart.png               # Pipeline flowchart (shown in Method tab)
  toy_example/                # Step-by-step worked example figures
H3_region_cache/              # Pre-generated regional H3 parquets (res 1–3)
H3_pop_density_maps/          # Per-country Kontur GPKG files (downloaded on demand)
```

---

## Probability models

Three distance-decay models are available:

| Model | Formula | Key parameter |
|---|---|---|
| Exponential | P(d) = exp(−d/λ) | λ = decay length (km) |
| Weibull | P(d) = exp(−(d/λ)ᵏ) | λ = scale (km), k = shape |
| Step function | P(d) = 1 if d ≤ d_max, else 0 | d_max = cutoff (km) |

The Weibull model (k ≥ 1) generalises exponential decay with a flat near-facility plateau followed by a steeper drop-off, matching observed patterns in RT attendance with distance.

Capacity allocation uses a **ring-based proportional algorithm**: for each facility, hexagons are grouped into concentric rings; each ring is served in full before moving outward, and when a ring exhausts remaining capacity the budget is split proportionally by demand weight across all hexagons in that ring.

---

## Citation

If you use this tool in your work, please cite:

> Wroe L, Ho A, Brown A, Martin S. *RT Access: a geospatial tool for modelling sub-national radiotherapy access.* (2025). https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access

---

## License

Released under the [MIT License](LICENSE).

---

## Contact

Laurence Wroe — laurencewroe@gmail.com
