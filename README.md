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

- Population density and cancer burden (GLOBOCAN 2022)
- RT demand (optimal utilisation rates or user-defined fraction)
- Geographic access probability using exponential, Weibull, step, or uniform distance-decay models
- Capacity-limited access via a ring-based proportional allocation algorithm

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
| Cancer incidence | [GLOBOCAN 2022](https://gco.iarc.who.int/today/) (IARC) | Bundled; 175 countries |
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
  globocan_xarray.nc          # GLOBOCAN 2022 data (Cancer × Metric × ISO3)
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

Four distance-decay models are available:

| Model | Formula | Key parameter |
|---|---|---|
| Exponential | P(d) = exp(−d/λ) | λ = decay length (km) |
| Weibull | P(d) = exp(−(d/λ)ᵏ) | λ = scale (km), k = shape |
| Step function | P(d) = 1 if d ≤ d_max, else 0 | d_max = cutoff (km) |
| Uniform | P(d) = 1 for all d | — |

The Weibull model (k ≥ 1) generalises exponential decay with a flat near-facility plateau followed by a steeper drop-off, matching observed patterns in RT attendance with distance.

Capacity allocation uses a **ring-based proportional algorithm**: for each facility, hexagons are grouped into concentric rings; each ring is served in full before moving outward, and when a ring exhausts remaining capacity the budget is split proportionally by demand weight across all hexagons in that ring.

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
