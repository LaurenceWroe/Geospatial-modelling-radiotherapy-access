# RadMaps

**Geospatial modelling of radiotherapy access using H3 hexagonal grids**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io)

RadMaps is a Streamlit application that models and visualises global access to radiotherapy (linear accelerators / linacs). It combines population density, cancer incidence, and LINAC facility data to estimate the probability that a person at any given location can access radiotherapy treatment.

---

## Screenshot

*Screenshot placeholder — add an image of the app here.*

---

## Overview

Approximately half of all cancer cases require radiotherapy, yet access to RT remains critically low in many parts of the world. National statistics mask large within-country inequalities driven by two compounding factors:

- **Geographic access** — RT requires attendance over several weeks; patients far from a facility are substantially less likely to complete treatment.
- **Machine capacity** — the finite number of linacs limits total annual throughput.

RadMaps combines both constraints simultaneously, producing per-hexagon estimates of:

- Population density and cancer burden (GLOBOCAN 2024)
- RT demand (optimal utilisation rates or user-defined fraction)
- RT demand from per-cancer-type optimal utilisation rates (CCORE 2012 model by default; Delaney 2005 as published in the preprint also selectable) and the matching workload in treatment fractions (Wong et al. 2016), with case-mix-adjusted capacity per linac
- Geographic access probability using step, Weibull, exponential, Gaussian (E2SFCA), power-law (gravity) or uniform decay models
- Capacity-limited access via a global-ring nearest-first allocation, with proportional (balanced-FCA) and optimal (linear-programme) rationing rules and 2SFCA / E2SFCA / gravity indices available as benchmarks

Maps are built on [H3 hexagonal grids](https://h3geo.org/) at resolutions from ~400 m (country level) down to ~87,000 km² (regional), enabling consistent multi-scale analysis. Travel-time routing via the TravelTime API (driving / public transport) is supported as an alternative to straight-line distance.

---

## Quick Start

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/your-org/radmaps.git
cd radmaps

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Population data for the selected region is downloaded automatically on first use from Kontur's public S3 bucket and cached locally.

---

## App Tabs

| Tab | Description |
|---|---|
| Introduction | Overview and motivation |
| Map Modelling | Interactive map: population, cancer incidence, RT demand, and access probability |
| Geography-Only | Access limited by geographic distance only |
| Capacity-Only | Access limited by LINAC capacity only |
| Data | Inspect and download underlying data |
| Method | Technical description of the modelling approach |
| Assumptions | Key assumptions and their justifications |
| Toy Example | Worked example to build intuition |
| Probability Models | Compare distance-decay functions interactively |

---

## Data Sources

| Data | Source | Notes |
|---|---|---|
| Population | [Kontur Population Dataset](https://www.kontur.io/portfolio/population-dataset/) | H3 resolution 8 (~400 m); downloaded automatically |
| Cancer incidence | [GLOBOCAN 2024](https://gco.iarc.who.int/today/) (IARC; Sung et al. 2026) | Bundled; 186 countries |
| Optimal RT utilisation | Delaney *et al.* 2005 | Site-specific RT fractions; bundled |
| LINAC locations & counts | [IAEA DIRAC](https://dirac.iaea.org/) | Bundled |

---

## Configuration

### TravelTime API (optional)

By default, RadMaps uses straight-line distance. To enable travel-time routing (driving or public transport), sign up for a free account at [traveltime.com](https://traveltime.com) and add your credentials to `.streamlit/secrets.toml`:

```toml
[traveltime]
app_id = "your_app_id"
api_key = "your_api_key"
```

A template is provided at `.streamlit/secrets.toml.example`.

---

## Project Structure

```
app.py                        # Main Streamlit application
requirements.txt              # Python dependencies
analysis/
  accessibility.py            # H3 accessibility computation (geographic + capacity-limited)
data/
  cancer.py                   # GLOBOCAN loader and H3 apportionment
  linacs.py                   # DIRAC LINAC database loader
  population.py               # Kontur population loader and resampler
  travel_time.py              # TravelTime API integration
  regions.py                  # Region definitions (Africa, Europe, etc.)
cancer_data/
  globocan_xarray_2024.nc     # GLOBOCAN 2024 data (Cancer × Metric × ISO3), the default
  globocan_xarray.nc          # GLOBOCAN 2022 (preprint; RADMAPS_GLOBOCAN=2022)
  optimal_rt_utilisations.csv # Site-specific RT utilisation fractions
linac_data/
  Database_DIRAC_fixed.csv    # DIRAC LINAC database (geocoding-corrected)
assets/
  flowchart.png               # Pipeline flowchart (shown in Method tab)
  toy_example/                # Step-by-step worked example figures
H3_region_cache/              # Regional H3 parquets (res 1–3; generated on demand, gitignored)
H3_pop_density_maps/          # Per-country Kontur GPKG files (downloaded on demand, gitignored)
.streamlit/
  secrets.toml.example        # TravelTime credentials template
```

---

## Probability Models

Six decay models are available (`analysis/decay.py`); *x* is straight-line distance (km) or travel time (min):

| Model | Formula | Key parameter |
|---|---|---|
| Step function | P(x) = 1 if x ≤ x_max, else 0 | x_max = catchment (classic 2SFCA) |
| Weibull | P(x) = exp(−(x/λ)ᵏ) | λ = scale, k = shape (k = 1 is exponential) |
| Exponential | P(x) = exp(−x/λ) | λ = decay length |
| Gaussian | P(x) = (e^{−½(x/x_max)²} − e^{−½}) / (1 − e^{−½}) within x_max | x_max = catchment (E2SFCA kernel, Luo & Qi 2009) |
| Power law | P(x) = min(1, (x/λ)^{−β}) | λ = plateau, β = exponent (gravity model) |
| Uniform | P(x) = 1 for all x | — |

The Probability Models tab offers published parameterisations as presets (Luo & Qi 2009 30-min Gaussian bands; 120-min catchments used by Wawrzuta et al. 2025 and the Philippine E2SFCA study).

### Capacity allocation

The default **nearest-first** rule allocates capacity in distance rings across *all facilities at once*: every hexagon–facility pair is assigned to a ring one hexagon wide, rings are processed nearest to farthest, and within a ring each facility serves the demand it reaches (weighted by demand × P) until its capacity is exhausted, splitting proportionally when a ring would exhaust it. The result is independent of the order facilities appear in the input file (verified by `tests/test_accessibility.py`).

Two alternative rules can be selected for comparison:

- **Proportional** — a capacity-capped balanced floating catchment (Paez et al. 2019): each hexagon's accessible demand is split across the facilities it can reach, and each facility serves its share at min(1, capacity / demand received). Shortage is shared equally across a catchment rather than rationed nearest-first.
- **Optimal** — a linear programme (SciPy HiGHS) maximising total patients treated, as an upper bound that shows how much capacity nearest-first allocation strands where catchments overlap.

Conventional **2SFCA, E2SFCA, 3SFCA and gravity** supply-to-demand ratios (`analysis/fca.py`) are reported in the Benchmark tab alongside the RadMaps metrics, and can be mapped per hexagon as a capped ratio or as an *implied deficit* (demand × (1 − min(1, ratio))). 3SFCA (Wan et al. 2012) splits each hexagon's demand across reachable facilities by selection weight, the uncapped counterpart of the proportional rule; none of the indices conserves capacity, so their national implied deficits are not comparable with the RadMaps deficit. The paper's Table 3 is pinned as a regression test (`tests/test_regression_table3.py`) so the default configuration keeps reproducing the published numbers.

---

## Citation

If you use this tool in your work, please cite:

> Wroe L. *RadMaps: a geospatial tool for modelling sub-national radiotherapy access.* (2025). https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access

---

## License

Released under the [MIT License](LICENSE).

---

## Contact

Laurence Wroe — laurencewroe@gmail.com
