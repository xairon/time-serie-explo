---
title: 'Junon: A hybrid platform for piezometric forecasting combining transfer function-noise models and deep learning'
tags:
  - Python
  - TypeScript
  - hydrogeology
  - groundwater
  - time series forecasting
  - transfer function-noise
  - deep learning
  - MLflow
authors:
  - name: Nicolas Ringuet
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Université de Tours, France
   index: 1
date: 28 May 2026
bibliography: paper.bib
---

# Summary

Junon is a full-stack web platform for groundwater level forecasting that brings together two complementary modelling paradigms behind a single user interface and a shared experiment registry. On one side, it integrates `Pastas` [@Collenteur2019] transfer function-noise (TFN) models, which link piezometric levels to climatic forcings (precipitation, evapotranspiration) through calibrated impulse responses. On the other side, it exposes the deep-learning forecasters of `Darts` [@Herzen2022], including TFT, N-BEATS, and Transformer-based architectures, with hyperparameter optimisation via `Optuna` [@Akiba2019] and explainability through SHAP and Captum.

The platform targets operational hydrogeologists working with the BRGM (French Geological Survey) data warehouse, which integrates piezometric chronicles from Hub'Eau, hydrogeological metadata from BDLISA, and reanalysis climate data from ERA5. Junon couples this data layer to a model registry (MLflow) where Pastas calibrations and Darts training runs are tagged with a common schema, enabling cross-paradigm comparison on shared calibration/validation splits. A scenario engine then lets users interrogate any calibrated Pastas model with parametric perturbations — synthetic pumping (drinking water, irrigation, industrial), climate trends, stress scaling — and visualises the impact (Δh) against the model's natural baseline.

# Statement of need

Operational groundwater monitoring agencies routinely face two distinct modelling questions. First, they need physically interpretable models that quantify how rainfall variability propagates to aquifer levels, the historical role of TFN approaches such as `Pastas`. Second, they need accurate short-to-medium term forecasts of piezometric levels, where data-driven deep-learning models often outperform parametric baselines [@Wunsch2021]. In practice, these two needs are addressed with disjoint tooling, separate validation protocols, and incompatible data plumbing, making the comparison or chaining of the two paradigms an ad-hoc effort.

Junon addresses this gap with three contributions. First, a **unified data layer** ingests heterogeneous sources (Hub'Eau API, BDLISA ontology, ERA5 reanalysis) and exposes them with a common temporal index and station identifier scheme. Second, an **MLflow-backed registry** tags every model — Pastas or Darts — with the same metadata (station, calibration window, aquifer family) so that side-by-side validation is one query away. Third, a **physically-bounded scenario engine** lets users perturb the calibrated Pastas model with new stresses (pumping, climate change) under constraints derived from the BDLISA aquifer family and from the calibrated step response itself, surfacing realistic drawdown estimates and flagging unrealistic configurations.

Junon is designed to lower the entry barrier for hydrogeologists who want to use deep-learning baselines without abandoning the interpretability of TFN models, and to give modellers a sandbox for exploring "what-if" scenarios that respect the calibrated physics of each station.

# Architecture

Junon is structured in three layers (Figure 1):

1. A **framework-free Python core** (`dashboard/utils/`) that wraps `Pastas` and `Darts` with project-specific helpers: a Pastas builder that auto-detects trends, a Darts model factory, a scenario engine that mutates calibrated models via the superposition principle, and an explainability suite (SHAP, attention, gradients).
2. A **FastAPI REST + SSE layer** (`api/`) that exposes the core through versioned endpoints, streams training progress in real-time, and validates all SQL identifiers against an allowlist.
3. A **React 19 single-page application** (`frontend/`) that presents the workflows (Observatory → Pastas Lab → AI Lab) with Plotly interactive charts, TanStack Query for state, and a dark-themed Tailwind UI in French (the operational language of BRGM hydrogeologists).

All long-running tasks — Pastas auto-fit, Darts training, counterfactual generation — are streamed via Server-Sent Events from FastAPI to React.

# Scenario engine

The scenario engine in `dashboard/utils/pastas/scenario.py` accepts a calibrated Pastas model and a list of modifications (pumping series, stress scaling, linear trend) and returns the simulated head response. New pumping stress models are configured with **aquifer-family-specific time constants** (`PUMPING_RFUNC_DEFAULTS`) that reflect horizontal pressure propagation, which is slower than the vertical recharge percolation calibrated by the original model. The engine derives soft and hard drawdown thresholds from the calibrated step response, and translates them into adaptive flow-rate bounds (`Q_soft`, `Q_hard`) that the UI surfaces to the user in real time as they configure a scenario.

# Acknowledgements

Junon builds on the open source work of the `Pastas` [@Collenteur2019] and `Darts` [@Herzen2022] development teams. We thank the BRGM data services (Hub'Eau, BDLISA, ADES) for making French piezometric chronicles openly available.

# References
