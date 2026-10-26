"""Static explanatory tab bodies, extracted from app.py to keep it lean.

These tabs are pure content (no sidebar/session state), so they live here as
plain functions called from within their `with tab_x:` blocks in app.py.
"""
import os as _os

import pandas as _pd
import streamlit as st


def render_introduction():
    st.header("Introduction")

    st.subheader("About RadMaps")
    st.markdown(
        """
        RadMaps is open source and released under the
        [MIT License](https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access/blob/main/LICENSE).
        Source code is available on [GitHub](https://github.com/LaurenceWroe/Geospatial-modelling-radiotherapy-access).
        """
    )
    st.markdown(
        """
        Approximately half of all cancer cases require radiotherapy, yet worldwide access
        to RT remains unacceptably low. 

        Access to RT is constrained by two principal factors:

        - **Machine capacity** — the finite number of linear accelerators (linacs) within a
          country limits the total number of patients that can be treated each year.
        - **Geographic access** — RT treatment typically requires visiting a hospital every day over a period
          of a few weeks. Patients that live far from a facility may experience reduced treatment outcomes 
          ([Silverwood et al. 2024](https://doi.org/10.1016/j.adro.2024.101652)).

        Previous work has addressed each of these factors independently (e.g. capacity 
        ([Abdel-Wahab *et al.* 2025](https://doi.org/10.1016/S1470-2045(24)00678-8) and geography 
        ([Wawrzuta *et al.* 2025](https://doi.org/10.1016/j.radonc.2025.111061))). 
        
        RadMaps illuminate both constraints, and provide a measure of access based on both of them. 
        It provides fast visualisation and analysis of access to radiotherapy (RT) at
        the sub-national scale, within countries and regions. 
        

        
        """
    )

    st.subheader("What each tab does")
    st.markdown(
        """
        | Tab | Contents |
        |---|---|
        | **🗺️ Access Maps** | Interactive H3 hexagon maps — population density, cancer burden, RT demand, geographic access probability, capacity-limited access, and workload (treatment fractions) demand and deficit. Select a country or region in the sidebar and click **Calculate RT Access**. |
        | **📊 Data** | Country-level data tables — cancer incidence by site (GLOBOCAN 2024), LINAC locations (DIRAC), and RT utilisation rates. |
        | **⚡ Capacity-Only** | Headline LINAC gap estimates (optimal RTU, proportional, and population benchmarks) without geographic constraints. |
        | **🌍 Geography-Only** | Distance/travel-time distributions to nearest LINAC after running Calculate RT Access. |
        | **🔧 Machine Planning** | Suggest optimal placements for new LINACs, or add custom facilities and re-evaluate access. |
        | **📖 Method** | Full pipeline description with flowchart, data sources, and step-by-step methodology. |
        | **⚠️ Assumptions** | Tabulated model assumptions and limitations, ranked by likely impact, with suggested improvements. |
        | **🧪 Toy Example** | Step-by-step worked example showing how each pipeline stage transforms inputs into access outputs. |
        | **📐 Probability Models** | Explanation and visualisation of the six decay models (step, Weibull, exponential, Gaussian, power-law, uniform) used to compute geographic access probability, with published parameterisations as presets. |
        | **📉 Sensitivity** | Equity (Lorenz curve / access Gini) and a ±parameter sweep showing how robust the headline figures are. |
        | **🔁 Benchmark** | Compare the RadMaps nearest-first allocation with proportional (balanced-FCA) and optimal rationing, and with conventional 2SFCA / E2SFCA / 3SFCA / gravity accessibility indices (also mappable as ratios or implied deficits). |
        | **🔬 Convergence** | How access estimates depend on H3 resolution (res 3–8), for both distance and driving time. |
        """
    )

    st.subheader("How regional analysis works")
    st.markdown(
        """
        When you select a **region** (e.g. Africa, Europe, or the whole World) rather than a
        single country, RadMaps computes access **per country** and then aggregates — it does
        not treat the region as one borderless surface:

        - Each country's cancer incidence (GLOBOCAN) is apportioned across **its own** population
          hexagons, and that demand can only be served by **that country's own LINACs**.
          Cross-border travel is not modelled, reflecting that patients are almost always treated
          within their own health system.
        - Every country is solved independently at the chosen resolution, then all countries are
          merged into one regional map. Shared border hexagons are combined so no population is
          double-counted.
        - Regional headline figures (A_G, A_C, A_RM) are the population- or demand-weighted totals
          across all included countries.
        - Countries lacking GLOBOCAN or LINAC data (see below) are omitted from the regional totals,
          so regional figures are lower bounds where coverage is incomplete.
        """
    )

    st.subheader("Data coverage & gaps")
    st.markdown(
        """
        RadMaps draws on three global datasets, each with its own coverage limits. Results for
        the affected countries should be read with these gaps in mind.

        **Cancer incidence — GLOBOCAN 2024.** National incidence is available for 186 countries and territories (Sung et al. 2026).
        **31 smaller territories and dependencies are not covered** and cannot be selected as a
        country (and are omitted from regional/world totals). The most notable are **Hong Kong,
        Taiwan, Macao, Mongolia, North Korea, Puerto Rico**, and the French overseas departments
        (**Guadeloupe, Martinique, Réunion**); the remainder are European microstates (Andorra,
        Liechtenstein, Monaco, San Marino, Vatican) and small Caribbean and Pacific island nations.

        **Driving / public-transport time — TravelTime API.** The routing provider has **no
        road-network coverage for China, Russia, North Korea, South Korea, and Macao.** For these
        countries, travel-time analysis is unavailable and only **straight-line distance** can be
        used. China (~1.4 billion people) is by far the largest such gap — for global travel-time
        summaries these countries are either excluded or approximated with a distance-based proxy.

        **Population — Kontur (2023).** Available for essentially all populated territories at
        ~400 m (H3 resolution 8); this is the one dataset without material gaps.
        """
    )

    st.subheader("Quick start guide")
    st.markdown(
        """
        1. **Select a country or region** from the dropdown in the left sidebar. Countries
           are limited to those with GLOBOCAN cancer incidence data. Regions (Africa, Europe,
           etc.) are also available at lower resolutions.

        2. **Choose a map type** — start with *Population Data* (then select *Population Density*)
           to see the underlying data, then *Cancer Incidence* or *Radiotherapy Demand* under
           the same dropdown to see cancer burden, then switch to *Radiotherapy Access* to see
           the combined model output.

        3. **Set the H3 resolution** — the map is built on an
           [H3 hexagonal grid](https://h3geo.org/). Resolution 8 (~400 m hexagons) gives the
           most detail for single countries; lower resolutions (3–5) are faster and better
           suited to regions.

        4. **Click Generate Map** — the first load for a new country downloads the Kontur
           population file (~1–60 seconds depending on country size); subsequent loads are
           instant.

        5. **Explore the access model** — under *Radiotherapy Access*, adjust the decay
           model (step, Weibull, exponential, Gaussian, power-law, or uniform), its
           parameters, the capacity per machine, and the allocation rule to see how
           results change.

        6. **Check the Data tab** for country-level cancer and LINAC statistics, and the
           **⚡ Capacity-Only** tab for a headline capacity gap estimate independent of
           geographic constraints.
        """
    )


def render_method():
    st.header("Method")

    # ------------------------------------------------------------------
    # Method / flowchart
    # ------------------------------------------------------------------
    st.subheader("Model Overview")

    import os as _os
    _flowchart_path = _os.path.join(_os.path.dirname(__file__), "assets", "flowchart.png")
    if _os.path.exists(_flowchart_path):
        st.image(_flowchart_path, use_container_width=True)

    st.markdown(
        """
        The model pipeline proceeds as follows:

        1. **Population density** — sub-national population is sourced from the
           [Kontur Population Dataset](https://www.kontur.io/portfolio/population-dataset/)
           (aggregated H3 hexagonal grid, ~400 m resolution at level 8). Each hexagon
           represents an area unit for all subsequent calculations.

        2. **Cancer incidence** — national cancer incidence figures are taken from
           [GLOBOCAN 2024](https://gco.iarc.fr/today/) (IARC). These are apportioned to
           individual hexagons in proportion to their population, under the assumption of
           spatially uniform cancer incidence rates (see Assumptions below).

        3. **Radiotherapy demand** — the number of patients requiring RT in each hexagon is
           estimated either by applying cancer-type-specific optimal RT utilisation rates
           to each cancer type, or by applying a user-specified proportional rate to all
           cancers excluding non-melanoma skin cancer (NMSC, RT utilisation ≈ 0%). The
           default optimal rates are the updated CCORE 2012 model (Barton *et al.* 2014,
           per-site values as tabulated by Wong *et al.* 2016; 48.4 % of all cancers). The
           Delaney *et al.* (2005) rates used for the RadMaps preprint (52.3 %) remain
           selectable. Both sets cover first-course external-beam indications of **both
           curative and palliative** intent: about 77 % of first courses are radical and
           23 % palliative, and Jacob *et al.* (2010) put the palliative share at about
           14 % of all new patients. Every rate can be overridden in the Data tab.

        4. **Linac locations and capacity** — facility locations and machine counts are
           sourced from the
           [DIRAC database](https://dirac.iaea.org/) (IAEA). By default each linac is
           assumed to treat a fixed number of patients per year (450). Optionally the
           capacity can instead be expressed as **fractions per linac per year** and
           converted to patients using the evidence-based fractions per first course of
           Wong *et al.* (2016; 19.4 overall, 2.0 for kidney to 29.1 for brain) weighted
           by the country's own cancer-type mix, with a hypofractionation factor to
           shorten every course. Fractions *per course* are used, not fractions *per
           patient*, because the demand already carries the utilisation rate.

        5. **Geographic access probability** — for each hexagon, the probability that a
           patient reaches *any* facility is computed as:

           $$P_{\\text{geo}} = 1 - \\prod_{i} \\left(1 - p(x_i)\\right)$$

           where $x_i$ is the straight-line distance, driving time or public-transport
           time to facility $i$ and $p(x_i)$ is the decay model (step, Weibull,
           exponential, Gaussian, power-law or uniform). Each facility counts once
           regardless of its machine count. This metric is independent of capacity.

        6. **Capacity-limited (modelled) access** — the default *nearest-first* rule
           allocates capacity in distance rings across **all facilities at once**: every
           hexagon–facility pair is assigned to a ring one hexagon wide, rings are processed
           from nearest to farthest, and within a ring each facility serves the demand it
           can reach, weighted by $D_h \\, p(x_{hi})$, until its remaining capacity is
           exhausted. If a ring would exhaust a facility's capacity, that capacity is split
           proportionally across the hexagons in the ring. Because all facilities advance
           ring by ring together, the result does not depend on the order facilities appear
           in the input file (same-ring ties are broken by a fixed coordinate sort). No
           hexagon receives more than its outstanding demand, and no facility exceeds its
           capacity. The ratio of treated to total demand per hexagon is the **Modelled
           Access Ratio**.

        7. **Workload in treatment fractions** — every demand and allocation figure is also
           expressed as treatment fractions, $F_h = \\sum_i D_{h,i}\\, f_i$, using the
           evidence-based fractions per first course $f_i$ of Wong *et al.* (2016) for each
           cancer type (editable in the Data tab for hypofractionation scenarios). Delivered
           fractions follow the patient allocation at each hexagon's mean course length, and
           the fraction deficit divided by the fractions one linac delivers per year gives
           the **LINAC-equivalents short**: the machines the unmet workload represents at
           benchmark throughput, if they could be placed within reach. With national
           incidence rates the cancer mix is uniform within a country, so fraction and
           patient access ratios coincide; the two diverge once sub-national incidence or
           site-specific fractionation is introduced.

           Two alternative rationing rules are available for comparison in the sidebar and
           the Benchmark tab: *proportional*, where each hexagon's accessible demand is split
           across the facilities it can reach and every facility serves its share at the
           ratio $\\min(1, \\text{capacity}/\\text{demand received})$ (a capacity-capped
           balanced floating catchment); and *optimal*, a linear programme that maximises
           the total number of patients treated and shows how much capacity nearest-first
           allocation leaves stranded where catchments overlap.
        """
    )


def render_assumptions():
    st.header("Assumptions and Limitations")
    st.markdown(
        "The model contains a number of simplifying assumptions. "
        "These are divided below by their likely impact on results."
    )

    import pandas as _pd

    _more_significant = _pd.DataFrame([
        {
            "Assumption": "Uniform cancer incidence",
            "Limitation": "Cancer incidence assumed proportional to population density; demographic and geographic variation not captured.",
            "To Improve": "Incorporate sub-national cancer incidence data at H3 resolution where available.",
        },
        {
            "Assumption": "No patient stratification",
            "Limitation": "All cancer patients treated as equivalent. Access barriers differ by age, mobility, socioeconomic status, cancer stage, and RT modality required.",
            "To Improve": "Stratify demand by cancer type, stage, and demographic; incorporate access modifiers where data permit.",
        },
        {
            "Assumption": "Probability model for geographic access",
            "Limitation": "The distance–RT uptake relationship is poorly characterised. No single model is universally accepted (Perez et al. 2016; Lin et al. 2015; Yap et al. 2023).",
            "To Improve": "Incorporate empirically validated, country-specific probability models.",
        },
        {
            "Assumption": "Distance-ranked allocation",
            "Limitation": "Capacity is allocated to the nearest patients first (globally, across all facilities, so the result is independent of facility ordering). Real referral patterns depend on clinical pathways, waiting times, and patient choice: roughly a fifth to a third of patients in England bypass their nearest centre (Aggarwal et al. 2017), more in the US and France. Nearest-first rationing can also leave capacity unused where catchments overlap.",
            "To Improve": "Compare with the proportional and optimal rules in the Benchmark tab; incorporate referral pathway data where available.",
        },
        {
            "Assumption": "National boundaries as hard limits (regional analysis)",
            "Limitation": "Regional and world maps are computed strictly per-country: every country's demand is served only by facilities inside that country. A patient living just across a border from a large centre cannot reach it in the model, so small countries and border regions can appear far more deprived than they are in reality. This is the single largest structural limitation of the regional view.",
            "To Improve": "Pool facilities within the distance/travel-time cutoff of a national border into both neighbouring countries' calculations, or model the whole region as one continuous surface.",
        },
    ])

    _less_significant = _pd.DataFrame([
        {
            "Assumption": "Incident (new) cancer cases only",
            "Limitation": "Demand based on new cases per year. Prevalent cases requiring re-treatment or delayed RT are excluded, so demand may be underestimated.",
            "To Improve": "Apply a correction factor based on the proportion of prevalent cases requiring RT.",
        },
        {
            "Assumption": "Linacs only",
            "Limitation": "Brachytherapy, orthovoltage, proton therapy, and other modalities excluded.",
            "To Improve": "Linacs dominate external-beam RT; fractional corrections for other modalities could be added.",
        },
        {
            "Assumption": "Equal weighting of facilities",
            "Limitation": "All facilities within range weighted by distance only. Referral networks may make distant specialist centres effectively inaccessible.",
            "To Improve": "Incorporate referral pathway data to weight facility accessibility.",
        },
        {
            "Assumption": "Static snapshot",
            "Limitation": "GLOBOCAN incidence and DIRAC machine counts are point-in-time. Population growth, ageing, and planned facilities are not modelled.",
            "To Improve": "Allow temporal projection using demographic growth rates and infrastructure pipelines.",
        },
        {
            "Assumption": "Private and public facilities treated equally",
            "Limitation": "DIRAC includes private facilities, but access to private machines is not universal. Effective access may be lower than modelled.",
            "To Improve": "Allow the user to flag or exclude private facilities based on healthcare system context.",
        },
        {
            "Assumption": "Data quality",
            "Limitation": "DIRAC machine locations may be out of date or contain coordinate errors. GLOBOCAN data unavailable for some countries (e.g. Mongolia).",
            "To Improve": "Validate and correct data sources as errors are identified.",
        },
        {
            "Assumption": "Modifiable Areal Unit Problem (MAUP)",
            "Limitation": "All spatial estimates depend on the chosen H3 resolution. Aggregating data into larger hexagons smooths local variation and can change apparent patterns — a known issue in areal statistics.",
            "To Improve": "Examine results at multiple resolutions; report sensitivity to resolution choice.",
        },
    ])

    st.markdown("##### More Significant Assumptions")
    st.dataframe(_more_significant, use_container_width=True, hide_index=True)

    st.markdown("##### Less Significant Assumptions")
    st.dataframe(_less_significant, use_container_width=True, hide_index=True)


def render_toy_example():
    st.header("Toy Example")
    st.markdown(
        """
        The following figures walk through each stage of the model pipeline using a
        simplified toy scenario. This illustrates how the inputs are transformed into
        the final access outputs step by step.
        """
    )

    _toy_dir = _os.path.join(_os.path.dirname(__file__), "assets", "toy_example")

    _toy_figures = [
        (
            "PopulationDensity.png",
            "Step 1 — Population Density",
            "The spatial distribution of population across the region, sourced from the "
            "Kontur H3 dataset. Each hexagon represents the number of population living within "
            "that cell. This forms the base layer for all subsequent calculations.",
        ),
        (
            "AnnualNewCancerDensity.png",
            "Step 2 — Cancer Incidence",
            "National cancer incidence figures (GLOBOCAN) are apportioned to each hexagon "
            "in proportion to its population. This gives an estimate of the number of new "
            "cancer cases arising in each cell each year.",
        ),
        (
            "CancerCasesRequiringRT.png",
            "Step 3 — Cancer Cases Requiring Radiotherapy",
            "Each cancer type is multiplied by its type-specific optimal radiotherapy "
            "utilisation rate (Delaney et al. 2005 in this illustration; the CCORE 2012 "
            "update is the tool default) and the results summed per hexagon. "
            "This gives the estimated number of patients in each cell who require RT annually.",
        ),
        (
            "GeographicProbability.png",
            "Step 4 — Geographic Access Probability",
            "For each hexagon, the probability that a patient can reach at least one facility "
            "is computed using the selected distance-decay model, combining contributions from "
            "all linacs. This is independent of machine capacity.",
        ),
        (
            "LinacCapacity.png",
            "Step 5 — Linac Capacity Allocation",
            "Machine capacity is distributed nearest-first in distance rings processed "
            "across all facilities together: the nearest ring of every facility is served "
            "before any facility moves to its next ring, so a patient close to one centre "
            "is never displaced by a patient far from another. Within a ring, a facility "
            "that cannot meet all the demand shares its remaining capacity proportionally. "
            "The proportion of demand met in each hexagon gives the Modelled Access Ratio.",
        ),
        (
            "Untreated.png",
            "Step 6 — Modelled Inaccessible Patients",
            "The difference between RT demand and allocated capacity in each hexagon gives "
            "the estimated number of patients who cannot access treatment. This highlights "
            "which areas are most underserved, whether due to distance or capacity shortfall.",
        ),
    ]

    for fname, heading, caption in _toy_figures:
        fpath = _os.path.join(_toy_dir, fname)
        st.subheader(heading)
        if _os.path.exists(fpath):
            st.image(fpath, width="stretch")
        else:
            st.warning(f"Image not found: {fname}")
        st.caption(caption)
        st.divider()
