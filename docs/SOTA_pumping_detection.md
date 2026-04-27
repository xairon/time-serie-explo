# Exhaustive Literature Review: State of the Art for Unsupervised Detection of Groundwater Pumping/Abstraction from Piezometric Time Series

---

## A. PHYSICS-BASED / HYDROGEOLOGICAL APPROACHES

### A1. Transfer Function Noise (TFN) Residual Analysis

**Von Asmuth, J.R., Bierkens, M.F.P., & Maas, K. (2002). Transfer function-noise modeling in continuous time using predefined impulse response functions. *Water Resources Research*, 38(12), 23-1--23-12.**
- **Method**: Introduces PIRFICT (Predefined IR Function in Continuous Time) models that decompose observed groundwater head time series into contributions from different hydrological stresses (rainfall, evaporation, pumping) using convolution with physically-based impulse response functions. The residuals after fitting represent unexplained variance -- potentially including unmodeled pumping.
- **Data requirements**: Irregular or regular groundwater head time series, precipitation, evapotranspiration; optionally pumping rates.
- **Strengths**: Works with irregular time steps; physically interpretable parameters; the residual after fitting known stresses can reveal hidden pumping influence.
- **Limitations**: Requires assumption of linearity; does not detect pumping per se, but residuals can indicate missing stresses.
- **Applied to pumping detection**: Indirectly -- residual analysis after fitting known stresses reveals unexplained drawdown.
- **Open source**: Pastas (Python), HydroSight (MATLAB).

**Collenteur, R.A., Bakker, M., Caljé, R., Klop, S.A., & Schaars, F. (2019). Pastas: Open Source Software for the Analysis of Groundwater Time Series. *Groundwater*, 57(6), 877-885.**
- **Method**: Python implementation of TFN modeling using predefined response functions (Gamma, Hantush, etc.). Models the effect of precipitation, evaporation, and pumping wells on observed groundwater heads. Decomposition plots show individual stress contributions. The ps.Hantush response function is used with `up=False` for pumping stresses.
- **Data requirements**: Groundwater head observations (regular or irregular), stress time series (precipitation, ET, pumping rates where known).
- **Strengths**: Open-source; flexible; physically-based response functions; active community; well-documented; handles irregular data.
- **Limitations**: Requires known stress inputs for full decomposition; pumping detection relies on residual analysis when pumping data is missing.
- **Applied to pumping detection**: Yes -- explicitly supports pumping well modeling and stress decomposition. Residuals indicate unexplained stresses.
- **Open source**: [Pastas](https://github.com/pastas/pastas) (Python, pip install pastas).
- Sources: [Pastas PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6899905/), [Pastas adding wells doc](https://pastas.readthedocs.io/stable/examples/adding_wells.html)

**Bakker, M. (2019). Solving Groundwater Flow Problems with Time Series Analysis: You May Not Even Need Another Model. *Groundwater*.**
- **Method**: Demonstrates that TFN time series analysis can estimate drawdown from multiple well fields without requiring a physical groundwater flow model, using physically-based response functions.
- **Data requirements**: Observation well data, pumping well locations and approximate rates.
- **Strengths**: Lighter computational requirements than numerical models; applicable at regional scale.
- **Limitations**: Assumes superposition (linearity); requires at least approximate pumping locations.
- **Applied to pumping detection**: Yes, estimates drawdown attributable to pumping.
- Sources: [Bakker 2019](https://ngwa.onlinelibrary.wiley.com/doi/10.1111/gwat.12927)

**Peterson, T.J. & Western, A.W. (2014). Nonlinear time-series modeling of unconfined groundwater head. *Water Resources Research*, 50(10).**
- **Method**: Extends TFN models to nonlinear systems (unconfined aquifers) using HydroSight. Includes joint estimation of gross recharge, groundwater usage, and hydraulic properties.
- **Data requirements**: Groundwater head time series, climate data.
- **Strengths**: Accounts for nonlinear storage changes; can jointly estimate pumping usage.
- **Limitations**: More complex parameterization; requires careful calibration.
- **Applied to pumping detection**: Yes -- explicitly estimates groundwater usage alongside other parameters.
- **Open source**: [HydroSight](https://github.com/peterson-tim-j/HydroSight) (MATLAB).
- Sources: [Peterson 2014 WRR](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013WR014800), [HydroSight GitHub](https://github.com/peterson-tim-j/HydroSight)

**Zaadnoordijk, W.J., Bus, S.A.R., Lourens, A., & Berendrecht, W.L. (2019). Automated Time Series Modeling for Piezometers in the National Database of the Netherlands. *Groundwater*, 57(6), 834-843.**
- **Method**: Automated TFN modeling applied to the entire Dutch national piezometric database. Separation of precipitation/evaporation from other influences (including pumping) at thousands of piezometers. The residual component captures unexplained variance including unmodeled pumping.
- **Data requirements**: Large-scale piezometric database, precipitation and evapotranspiration data.
- **Strengths**: Scalable, automated; national-scale deployment; online visualization at grondwatertools.nl.
- **Limitations**: Does not explicitly attribute residuals to pumping vs. other unmodeled stresses.
- **Applied to pumping detection**: Indirectly -- the "other influences" component captures pumping.
- **Open source**: Uses Pastas framework.
- Sources: [Zaadnoordijk 2019](https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/gwat.12819)

### A2. Analytical Solutions (Theis, Hantush) for Pumping Signature Recognition

**Lin, Y.-C., Yeh, T.-C.J., et al. (2024). Analysis of Groundwater Time Series With Limited Pumping Information in Unconfined Aquifer: Response Function Based on Lagging Theory. *Water Resources Research*, 60, e2023WR036747.**
- **Method**: Novel mathematical model that captures drawdown response due to pumping using lagging theory and the Boussinesq equation. Critically, it estimates hydrogeological parameters **without dependence on specific pumping information** (pumping rate time series, locations of wells).
- **Data requirements**: Groundwater head time series only (no pumping data required).
- **Strengths**: Does not require knowledge of pumping well locations or rates; works on unconfined aquifers; captures capillary fringe effects.
- **Limitations**: Assumes uniform pumping approach; restricted to unconfined aquifers.
- **Applied to pumping detection**: Yes, directly -- estimates pumping influence from head data alone.
- Sources: [Lin 2024 WRR](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR036747)

### A3. BRGM Tools

**GARDENIA (BRGM). Lumped hydrological modelling of a catchment basin.**
- **Method**: Rainfall-discharge-groundwater lumped model. Uses meteorological data (precipitation, PET) to calculate flow rates and groundwater levels. Time series of pumping flow rates can be included in calculations. Residuals between modeled and observed groundwater levels when pumping is not included in inputs could indicate hidden abstraction.
- **Data requirements**: Precipitation, PET, optionally pumping rates, observed groundwater levels for calibration.
- **Strengths**: Well-validated; used operationally in France; can include/exclude pumping as model input.
- **Limitations**: Lumped model (no spatial distribution); requires calibration.
- **Applied to pumping detection**: Indirectly -- model-data misfit when pumping is omitted reveals its influence.
- **Availability**: Proprietary BRGM software, free upon request.
- Sources: [GARDENIA BRGM](https://www.brgm.fr/en/software/gardenia-lumped-hydrological-modelling-catchment-basin)

**TIGRE (BRGM). Théorie des Images dans une Géométrie REctangulaire.**
- **Method**: Calculates piezometric level evolution over time from a set of pumping/injection wells in a homogeneous aquifer using image well theory. Can be used to simulate the expected impact of known pumping and compare with observations.
- **Data requirements**: Pumping rates and well locations; aquifer transmissivity and storage coefficient.
- **Strengths**: Simple, fast computation; accounts for boundary conditions.
- **Limitations**: Assumes homogeneous aquifer; requires known pumping inputs; limited to rectangular boundaries.
- **Applied to pumping detection**: Indirectly -- comparison of TIGRE predictions with observations reveals discrepancies from unknown pumping.
- **Availability**: Free download from BRGM (Windows), [TIGRE BRGM](https://www.brgm.fr/en/software/tigre-easy-calculation-influence-well-field-homogeneous-aquifer)

**OUAIP (BRGM). Outil d'Aide à l'Interprétation des Pompages d'essai.**
- **Method**: Pumping test interpretation tool offering visual and automatic curve-matching using analytical solutions (Theis, Hantush, etc.) for various aquifer types. Version 3.1 includes drawdown derivative diagnosis with flow rate deconvolution.
- **Data requirements**: Pumping test drawdown data; pumping rates.
- **Strengths**: Comprehensive set of analytical solutions; free; multilingual (FR/EN/ES); handles well effects and boundary effects.
- **Limitations**: Designed for pumping tests, not continuous monitoring; requires dedicated pumping test data.
- **Applied to pumping detection**: Indirectly -- identifies aquifer parameters that can then be used to detect pumping signatures in monitoring data.
- **Availability**: Free at [ouaip.brgm.fr](http://ouaip.brgm.fr/)

**MétéEAU Nappes (BRGM). Groundwater monitoring and forecasting platform.**
- **Method**: Operational real-time monitoring and 6-month forecasting of French groundwater levels using global models (GARDENIA, EROS, Tempo). Forecasts are compared with prefectural drought thresholds. Discrepancies between model forecasts and observations could indicate unanticipated abstraction.
- **Data requirements**: Real-time piezometric data from the French national monitoring network; meteorological data.
- **Strengths**: Operational; national scale; real-time data; freely accessible online.
- **Limitations**: Not designed specifically for pumping detection; models are lumped.
- **Applied to pumping detection**: Indirectly -- systematic model-observation discrepancies in specific areas could flag undeclared abstraction.
- **Availability**: [meteeaunappes.brgm.fr](https://meteeaunappes.brgm.fr/en)
- Sources: [MétéEAU Nappes](https://www.brgm.fr/en/solutions/meteeau-nappes-tool-real-time-monitoring-forecasting-groundwater-levels)

### A4. Recession Curve Analysis

**MRCPtool: Master Recession Curve Parameterization Tool. *Computers & Geosciences*, 2019.**
- **Method**: Builds master recession curves (MRC) from observed groundwater hydrographs. Anomalous deviations from the expected MRC (steeper recession, unexpected drawdown events) can indicate pumping impacts. Multiple approaches to MRC construction are compared.
- **Data requirements**: Groundwater head time series (ideally high-frequency).
- **Strengths**: Physically intuitive; parameter-free baseline; deviations from MRC are interpretable.
- **Limitations**: Recession segments vary with recharge features, spatial distribution, and aquifer properties; requires sufficient dry periods for MRC construction.
- **Applied to pumping detection**: Transferable -- anomalous recession patterns deviating from natural MRC could indicate pumping.
- Sources: [MRCPtool](https://www.sciencedirect.com/science/article/abs/pii/S0098300419301025), [USGS MRC](https://rdrr.io/github/USGS-R/DVstats/f/inst/doc/MasterRecessionCurve.pdf)

### A5. Water Balance Methods

**Salmoral, G., et al. (2025). Estimation of Groundwater Abstractions from Irrigation Wells in Mediterranean Agriculture: An Ensemble Approach. *Sustainability*, 17(12), 5618.**
- **Method**: Combines NDVI time series, crop water requirement modeling, and spatial analysis of irrigation systems within a GIS environment. Soil water balance models estimate irrigation requirements, with groundwater abstraction inferred as the residual unknown in the water budget.
- **Data requirements**: Remote sensing (NDVI), meteorological data, soil properties, land use maps.
- **Strengths**: Spatially distributed; integrates multiple data sources; works without direct pumping measurements.
- **Limitations**: Significant uncertainty (20-100% for groundwater extractions); cannot distinguish individual wells.
- **Applied to pumping detection**: Yes, directly -- estimates total abstraction volumes from indirect evidence.

**Gonzalez-Dugo, M.P., et al. (2009). Methodology for Quantifying Groundwater Abstractions for Agriculture via Remote Sensing and GIS. *Water Resources Management*, 24, 795-814.**
- **Method**: Teledetection-based water balance combining satellite data with GIS to estimate net groundwater use in irrigated areas.
- **Data requirements**: Satellite imagery, meteorological stations, soil data, crop maps.
- **Strengths**: Regional scale; independent of metering; repeatable.
- **Limitations**: Indirect estimation with high uncertainty.
- **Applied to pumping detection**: Yes, estimates unmeasured abstraction.
- Sources: [Gonzalez-Dugo 2009](https://link.springer.com/article/10.1007/s11269-009-9473-7)

---

## B. SIGNAL PROCESSING APPROACHES

### B1. Hilbert-Huang Transform (HHT) + EMD

**Hsieh, C.-S., Yeh, T.-C.J., et al. (2024). Estimating spatiotemporal pumping amounts using multiple signal decomposition methods. *Journal of Hydrology*, 638, 131461.**
- **Method**: Integrates HHT (Hilbert-Huang Transform) with EOF (Empirical Orthogonal Function) analysis on first-differenced head data to extract high-frequency variations closely related to pumping. EOF identifies spatial patterns associated with pumping locations; HHT removes noise from the extracted pumping signals. Validated against MODFLOW synthetic data.
- **Data requirements**: Hourly groundwater head data from a dense monitoring well network.
- **Strengths**: Produces high space-time resolution estimates of pumping distribution; validated against synthetic data; does not require pumping data as input.
- **Limitations**: Requires dense monitoring network; sensitive to noise; validated primarily on synthetic data.
- **Applied to pumping detection**: Yes, directly -- this is the primary purpose.
- Sources: [Hsieh et al. 2024 JoH](https://www.sciencedirect.com/science/article/abs/pii/S0022169424002506)

**Hsieh, C.-S., Yeh, T.-C.J., et al. (2023). A novel framework for spatiotemporal groundwater pumping process estimation based on data-driven approaches. *Journal of Hydrology*, 625, 130097.**
- **Method**: Predecessor framework using EOF + HHT. Pumping-related signals identified from distinct localized spatial EOF patterns, then temporally refined with HHT.
- **Data requirements**: Same as above -- dense, high-frequency monitoring data.
- **Strengths**: Pioneering data-driven approach; physically interpretable decomposition.
- **Limitations**: Assumes pumping produces localized spatial patterns distinct from natural recharge.
- **Applied to pumping detection**: Yes, directly.
- Sources: [Hsieh et al. 2023 JoH](https://www.sciencedirect.com/science/article/abs/pii/S0022169423006510)

### B2. Wavelet Decomposition

**Doña Nacional Park Wavelet Analysis. Hristov, V., et al. (2023). Wavelet Analysis on Groundwater, Surface-Water Levels and Water Temperature in Doñana National Park. *Water*, 15(4), 796.**
- **Method**: Wavelet transform (CWT/DWT) applied to groundwater, surface water, and temperature to identify periodic components at multiple scales. Seasonal agricultural pumping can be isolated as a distinct component at 64-128 day scales.
- **Data requirements**: Regular time series of groundwater levels; optionally surface water and meteorological data.
- **Strengths**: Multi-resolution analysis; reveals time-varying frequency content; can isolate pumping-related periodicities.
- **Limitations**: Interpretation requires domain expertise; edge effects at series boundaries.
- **Applied to pumping detection**: Yes -- seasonal pumping patterns identified at characteristic scales.
- Sources: [Hristov 2023 Water](https://www.mdpi.com/2073-4441/15/4/796)

**Groundwater Level Prediction by Wavelet Deep Learning with Smart Pumping Data. *Water Resources Management*, 39(6), 2025.**
- **Method**: Hybrid wavelet-deep learning model combining wavelet analysis with LSTM/RNN, using IoT smart pumping meter data to decompose groundwater level signals.
- **Data requirements**: Groundwater levels + IoT pumping meter data.
- **Strengths**: Combines signal decomposition with deep learning; uses real-time pump metering data.
- **Limitations**: Requires smart metering infrastructure.
- **Applied to pumping detection**: Indirectly -- validates that wavelet decomposition captures pumping-specific components.
- Sources: [WRM 2025](https://link.springer.com/article/10.1007/s11269-024-04088-0)

### B3. Fourier / Spectral Analysis

**Spectral analysis of water level fluctuations in aquifers. *Stochastic Environmental Research and Risk Assessment*, 2003.**
- **Method**: Fast Fourier Transform (FFT) applied to groundwater head time series to identify periodic components. Pumping signatures appear as spectral peaks at pumping frequencies (daily, weekly cycles for intermittent pumping). Fourier analysis of daily and sub-daily head fluctuations reveals pumping-induced periodicities.
- **Data requirements**: High-frequency (ideally hourly or sub-daily) groundwater head time series.
- **Strengths**: Simple, well-understood method; computationally efficient; clear interpretation of periodic signals.
- **Limitations**: Assumes stationarity; cannot capture time-varying pumping; poor resolution for short records.
- **Applied to pumping detection**: Yes -- periodic pumping creates characteristic spectral signatures.
- Sources: [Spectral analysis aquifers](https://link.springer.com/article/10.1007/s00477-002-0106-4)

**Spectral Analysis of Groundwater Level Time Series for Robust Estimation of Aquifer Response Times. 2024.**
- **Method**: Welch's method for power spectral estimation to estimate aquifer response times and identify dominant periodicities.
- **Data requirements**: Regular groundwater head time series.
- **Strengths**: Robust spectral estimation; noise reduction through averaging.
- **Applied to pumping detection**: Transferable -- aquifer response time anomalies could indicate pumping.
- Sources: [ResearchGate](https://www.researchgate.net/publication/397802991)

### B4. Singular Spectrum Analysis (SSA)

**Singular Spectrum Analysis for Time Series. Golyandina, N. & Zhigljavsky, A. (2013). Springer.**
- **Method**: Non-parametric, data-driven decomposition of time series via trajectory matrix embedding, SVD, and diagonal averaging. Extracts trend, oscillatory components, and noise without assuming a parametric model. Pairs of SSA eigenmodes with nearly equal eigenvalues and quadrature principal components represent oscillations.
- **Data requirements**: Any regular time series (groundwater head).
- **Strengths**: Model-free; extracts polynomial and exponential trends; identifies oscillatory components adaptively; handles non-stationary data.
- **Limitations**: Choice of window length (L) affects decomposition; grouping of components requires expertise or heuristics.
- **Applied to pumping detection**: Transferable -- separation of pumping-induced trend from natural variation; detection of artificial periodic components.
- **Open source**: `pyssa` (Python), `Rssa` (R).
- Sources: [Springer book](https://link.springer.com/book/10.1007/978-3-642-34913-3)

**Reconstruction of groundwater levels using SSA and MSSA. *Hydrological Sciences Journal*, 2019.**
- **Method**: SSA and multichannel SSA (MSSA) applied to reconstruct groundwater level time series and impute missing values. MSSA exploits spatial correlations across multiple wells.
- **Data requirements**: Multiple groundwater head time series (potentially with gaps).
- **Strengths**: Handles missing data; exploits spatial correlations in multi-well networks.
- **Limitations**: Requires sufficient temporal record length.
- **Applied to pumping detection**: Transferable -- anomalous components in SSA decomposition could indicate pumping.
- Sources: [MSSA groundwater](https://www.tandfonline.com/doi/full/10.1080/02626667.2019.1669793)

### B5. Independent Component Analysis (ICA)

**Hsieh, C.-S., et al. (2015). Independent component analysis for characterization and quantification of regional groundwater pumping. *Journal of Hydrology*, 527, 505-516.**
- **Method**: Uses ICA to separately extract characteristics of different pumping types from hourly groundwater head observations at monitoring wells. ICA decomposes the mixed signal observed at each well into statistically independent source components, each associated with a different type of pumping. These are then fitted with a calibrated groundwater simulation model to quantify pumping rates.
- **Data requirements**: Hourly groundwater head observations from multiple monitoring wells.
- **Strengths**: Separates mixed pumping signals without a priori knowledge of pumping types; combined with physics-based model for quantification; does not require pumping data.
- **Limitations**: Assumes statistical independence of sources; requires sufficient monitoring well density; needs calibrated numerical model for quantification step.
- **Applied to pumping detection**: Yes, directly -- primary purpose is to characterize and quantify regional pumping from head observations.
- Sources: [Hsieh 2015 JoH](https://www.sciencedirect.com/science/article/abs/pii/S0022169415003637)

**Time-frequency analysis of groundwater depth variation based on ICA-WTC composite method. *Journal of Hydrology*, 2022.**
- **Method**: Combines ICA with Wavelet Transform Coherence (WTC) for time-frequency analysis of groundwater depth variations.
- **Data requirements**: Multi-well groundwater head time series.
- **Strengths**: Integrates source separation (ICA) with time-frequency analysis (WTC).
- **Applied to pumping detection**: Transferable -- separates natural and anthropogenic components.
- Sources: [ICA-WTC 2022](https://www.sciencedirect.com/science/article/abs/pii/S0022169422014846)

---

## C. STATISTICAL / CHANGE POINT DETECTION

### C1. PELT, BOCPD, CUSUM

**Killick, R., Fearnhead, P., & Eckley, I.A. (2012). Optimal Detection of Changepoints with a Linear Computational Cost. *JASA*, 107(500), 1590-1598.**
- **Method**: PELT (Pruned Exact Linear Time) algorithm for multiple changepoint detection. Uses dynamic programming with pruning for exact global optimal detection. Introduces a penalty term to control overfitting.
- **Data requirements**: Any univariate or multivariate time series.
- **Strengths**: Computationally efficient (linear time); exact solution; well-suited for detecting abrupt changes in mean or variance.
- **Limitations**: Assumes changes in statistical properties; may not distinguish pumping from other causes of level change.
- **Applied to pumping detection**: Transferable -- abrupt drawdown events from pumping onset/cessation appear as changepoints in mean or trend.
- **Open source**: `ruptures` (Python), `changepoint` (R).
- Sources: [ruptures PELT](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/)

**Adams, R.P. & MacKay, D.J.C. (2007). Bayesian Online Changepoint Detection. arXiv:0710.3742.**
- **Method**: BOCPD recursively computes posterior probability of changepoint at each new observation. Operates online (real-time).
- **Data requirements**: Sequential observations.
- **Strengths**: Online/streaming capable; probabilistic output (uncertainty quantification); does not require specifying number of changepoints.
- **Limitations**: Computational complexity can be high for long series; requires specification of hazard function and observation model.
- **Applied to pumping detection**: Transferable -- real-time detection of abrupt level changes from pumping.

### C2. BEAST

**Zhao, K., et al. (2019). Detecting change-point, trend, and seasonality in satellite time series data to track abrupt changes and nonlinear dynamics: A Bayesian ensemble algorithm. *Remote Sensing of Environment*, 232, 111181.**
- **Method**: BEAST (Bayesian Estimator of Abrupt Change, Seasonality, and Trend) uses Bayesian model averaging over tens of thousands of possible decomposition models. Decomposes time series into trend, seasonal, and noise components while detecting changepoints in both trend and seasonality. Provides posterior probabilities for each changepoint.
- **Data requirements**: Regular time series (groundwater heads, satellite data).
- **Strengths**: Quantifies uncertainty; handles non-stationary series; detects changes in both trend and seasonality; robust to diverse patterns.
- **Limitations**: Computationally intensive; primarily designed for single time series.
- **Applied to pumping detection**: Yes -- recent groundwater application (2024 study on UK groundwater drought detecting trend and seasonal abrupt changes along groundwater and precipitation index time series using BEAST).
- **Open source**: `Rbeast` (R, Python, MATLAB) -- [GitHub](https://github.com/zhaokg/Rbeast), [PyPI](https://pypi.org/project/Rbeast/)
- Sources: [BEAST ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0034425719301853), [UK groundwater drought 2024](https://www.sciencedirect.com/science/article/abs/pii/S0022169424008254)

### C3. Hidden Markov Models (Regime Switching)

**Bracken, C., et al. (2014). A hidden Markov model combined with climate indices for multidecadal streamflow simulation. *Water Resources Research*, 50(10).**
- **Method**: HMM treats the system as switching between hidden states (e.g., pumping-on, pumping-off, natural regime) via a Markov chain with transition probabilities. Each state has a characteristic distribution of observations.
- **Data requirements**: Groundwater head time series; optionally climate indices.
- **Strengths**: Naturally models on/off regime switching; probabilistic state assignment; captures persistence.
- **Limitations**: Requires specification of number of states; interpretation of states as "pumping" vs "natural" requires validation.
- **Applied to pumping detection**: Transferable -- pumping on/off states create distinct regimes in groundwater heads. The Chengdu study (below) applied Markov-type models to groundwater anomaly detection.

**Groundwater Fluctuation Prediction using HMM. *Water Resources Management*, 2011.**
- **Method**: Sequence-based Markovian stochastic model representing daily groundwater fluctuation magnitudes in ten states and pattern changes in three states.
- **Data requirements**: Daily groundwater level data.
- **Strengths**: Maximum likelihood ratio above 90%; RMSE below 0.15 m.
- **Applied to pumping detection**: Transferable -- anomalous state transitions could indicate pumping.
- Sources: [WRM 2011](https://link.springer.com/article/10.1007/s11269-011-9808-z)

### C4. Mann-Kendall Trend Tests

**Fang, S., et al. (2019). Groundwater Level Analysis Using Regional Kendall Test for Trend with Spatial Autocorrelation. *Groundwater*, 57(1).**
- **Method**: Non-parametric trend test comparing relative magnitudes of sample data. Regional Kendall variant accounts for spatial autocorrelation across monitoring wells.
- **Data requirements**: Long-term groundwater level records.
- **Strengths**: Non-parametric; robust to outliers; widely used and understood; handles seasonal data.
- **Limitations**: Detects monotonic trends only; cannot isolate pumping from other causes of decline; affected by autocorrelation.
- **Applied to pumping detection**: Yes -- widely used for detecting long-term pumping-induced decline (e.g., Dhaka, Bangladesh: -1.5 to -2.11 m/year; Malwa India study 2025).
- Sources: [Fang 2019 Groundwater](https://ngwa.onlinelibrary.wiley.com/doi/10.1111/gwat.12800), [Dhaka study](https://link.springer.com/article/10.1007/s12665-021-09633-3), [Malwa 2025](https://link.springer.com/article/10.1007/s12665-025-12732-0)

### C5. Structural Break Tests (Chow, Bai-Perron)

**Bai, J. & Perron, P. (2003). Computation and Analysis of Multiple Structural Change Models. *Journal of Applied Econometrics*, 18(1), 1-22.**
- **Method**: Detects and dates multiple structural breaks in linear regression models. The Bai-Perron framework tests for an unknown number of breaks at unknown dates using sup-F statistics and dynamic programming.
- **Data requirements**: Time series with potential covariates.
- **Strengths**: Rigorous statistical framework; dates breaks with confidence intervals; handles multiple breaks.
- **Limitations**: Requires large samples for reliable detection; primarily designed for linear regression models; no specific groundwater implementation found.
- **Applied to pumping detection**: Transferable -- structural breaks in the relationship between rainfall and groundwater levels could indicate onset of pumping.
- **Open source**: `strucchange` (R), `xtbreak` (Stata).

---

## D. MACHINE LEARNING ANOMALY DETECTION

### D1. LSTM Autoencoders

**Rezaiezadeh Roukerd, F. & Rajabi, M.M. (2024). Anomaly detection in groundwater monitoring data using LSTM-Autoencoder neural networks. *Environmental Monitoring and Assessment*, 196, 848.**
- **Method**: LSTM autoencoder trained on "normal" groundwater monitoring data. Reconstruction error on new observations serves as anomaly score -- high error indicates data that deviates from learned normal patterns. Considers temporal and contextual aspects.
- **Data requirements**: Groundwater head time series; a training period of "clean" (non-anomalous) data.
- **Strengths**: Captures temporal dependencies; unsupervised; can detect complex, non-obvious anomalies.
- **Limitations**: Requires a representative "normal" training period; black-box; sensitive to hyperparameters; not specific to pumping (detects any anomaly).
- **Applied to pumping detection**: Transferable -- pumping-induced drawdown deviating from normal patterns would produce high reconstruction error.
- Sources: [Rezaiezadeh 2024 EMA](https://link.springer.com/article/10.1007/s10661-024-12848-z)

### D2. Isolation Forest, One-Class SVM, LOF

**Liu, Z., et al. (2023). Machine learning-based anomaly detection of groundwater microdynamics: case study of Chengdu, China. *Scientific Reports*, 13, 11684.**
- **Method**: Compares sl-Pauta, Isolation Forest (iForest), One-Class SVM (OCSVM), and KNN for detecting anomalies in groundwater level data influenced by atmospheric pressure, precipitation, and tidal effects. OCSVM achieved best performance (precision 88.89%, recall 91.43%).
- **Data requirements**: Groundwater level time series; atmospheric pressure; precipitation.
- **Strengths**: Comparative evaluation of multiple methods; real-world application; accounts for atmospheric and tidal confounders.
- **Limitations**: Requires feature engineering; methods detect generic anomalies, not specifically pumping.
- **Applied to pumping detection**: Transferable -- pumping-induced microdynamic anomalies could be detected as deviations from expected atmospheric/tidal responses.
- Sources: [Liu 2023 Scientific Reports](https://www.nature.com/articles/s41598-023-38447-5)

**Cascade of One-Class Classifiers for Water Level Anomaly Detection. *Electronics*, 9(6), 1012, 2020.**
- **Method**: Dual-staged cascade OCSVM: first stage detects point anomalies from single observations; second stage uses n-gram feature vectors to discover collective anomalies (sequences).
- **Data requirements**: Water level time series.
- **Strengths**: Detects both point and collective anomalies; two-stage design reduces false positives.
- **Applied to pumping detection**: Transferable -- collective anomalies (sustained drawdown periods) characteristic of pumping.
- Sources: [Cascade OCC 2020](https://www.mdpi.com/2079-9292/9/6/1012)

### D3. Variational Autoencoders (VAE)

**FCVAE: Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective. *ACM Web Conference 2024*.**
- **Method**: Frequency-enhanced Conditional VAE (FCVAE) that integrates both global and local frequency features. Addresses limitations of standard VAEs in capturing long-periodic heterogeneous patterns and short-periodic trends simultaneously.
- **Data requirements**: Multivariate time series.
- **Strengths**: Better reconstruction of normal data than standard VAE; captures multi-scale temporal patterns; unsupervised.
- **Limitations**: Complex architecture; no groundwater-specific application found.
- **Applied to pumping detection**: Transferable -- multi-scale pumping signatures (daily cycles, seasonal patterns) could be captured.
- Sources: [FCVAE ACM 2024](https://dl.acm.org/doi/10.1145/3589334.3645710)

**MST-VAE: Multi-Scale Temporal Variational Autoencoder. *Applied Sciences*, 12(19), 10078, 2022.**
- **Method**: Combines multi-scale temporal convolutional kernels in 1D CNN with VAE.
- **Data requirements**: Multivariate time series.
- **Strengths**: Captures various temporal patterns; stochastic modeling.
- **Applied to pumping detection**: Transferable.
- Sources: [MST-VAE 2022](https://www.mdpi.com/2076-3417/12/19/10078)

### D4. Transformer-Based Anomaly Detection

**Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy. *ICLR 2022 (Spotlight)*.**
- **Method**: Proposes Anomaly-Attention mechanism using "association discrepancy" criterion. Normal points form broad associations across the series, while anomalies concentrate associations on adjacent points (adjacent-concentration bias). A minimax strategy amplifies this distinguishability.
- **Data requirements**: Unsupervised -- only time series data (no labels).
- **Strengths**: State-of-the-art on 6 benchmarks; principled criterion for anomaly detection; unsupervised; handles SWaT water treatment dataset.
- **Limitations**: High computational cost for long series; no groundwater-specific application.
- **Applied to pumping detection**: Transferable -- pumping events would create localized anomalies with concentrated temporal associations.
- **Open source**: [GitHub](https://github.com/thuml/Anomaly-Transformer)
- Sources: [Anomaly Transformer arXiv](https://arxiv.org/abs/2110.02642)

**Tuli, S., Casale, G., & Jennings, N.R. (2022). TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data. *VLDB 2022*, 15(6), 1201-1214.**
- **Method**: Two-phase adversarial training with attention-based sequence encoders. Focus score-based self-conditioning enables robust multi-modal feature extraction. Adversarial training amplifies reconstruction errors.
- **Data requirements**: Multivariate time series.
- **Strengths**: +17% F1 improvement over baselines; 99% faster training; handles multivariate data.
- **Limitations**: Requires careful hyperparameter tuning; no groundwater-specific validation.
- **Applied to pumping detection**: Transferable.
- **Open source**: [GitHub](https://github.com/imperial-qore/TranAD)
- Sources: [TranAD VLDB](https://dl.acm.org/doi/abs/10.14778/3514061.3514067)

### D5. Graph Neural Networks for Spatial Anomaly Detection

**Spatial-temporal graph neural networks for groundwater data. *Scientific Reports*, 14, 24284, 2024.**
- **Method**: ST-GNN processes multivariate time series with a graph structure delineating interconnections between monitoring wells. Captures spatial interconnectivity and temporal dynamics of groundwater systems. Modified Multivariate Time Graph Neural Network handles missing data.
- **Data requirements**: Multi-well groundwater head time series; spatial coordinates of wells.
- **Strengths**: Captures spatial correlations between wells; handles missing data; anomaly detection through disrupted spatial dependencies.
- **Limitations**: Requires well network with sufficient spatial density; graph construction choices affect results.
- **Applied to pumping detection**: Transferable -- pumping at one well disrupts expected spatial correlations with neighbors, detectable as anomalies.
- Sources: [ST-GNN groundwater 2024](https://www.nature.com/articles/s41598-024-75385-2)

**Graph Neural Network-Based Anomaly Detection for River Network Systems. *F1000Research*, 12, 991, 2023.**
- **Method**: GNN-based approach for anomaly detection in spatially connected water networks. Graph Deviation Network (GDN) detects even small-deviation anomalies overlooked by distance-based and density-based methods.
- **Data requirements**: Multivariate sensor network data with spatial connectivity.
- **Strengths**: Detects small deviations; learns interdependencies; spatial awareness.
- **Applied to pumping detection**: Transferable from river networks to well networks.
- **Open source**: [gnnad GitHub](https://github.com/KatieBuc/gnnad)
- Sources: [GNN river networks 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC11162521/)

---

## E. XAI-BASED DETECTION (NOVEL ANGLE)

### E1. SHAP / TimeSHAP for Feature Attribution Shifts

**Clark, S.R., et al. (2025). Explainable AI for Interpreting Spatiotemporal Groundwater Predictions. *Water Resources Research*, 61, e2025WR041303.**
- **Method**: Feed-forward neural network predicting groundwater levels in the Murray-Darling Basin, probed with SHAP to identify the most influential inputs. Analysis at high spatiotemporal resolution reveals contribution of each predictor to individual monthly predictions. Shifts in feature attributions across space/time could reveal hidden stresses like pumping.
- **Data requirements**: Groundwater head, climate, remote sensing, and land use predictors.
- **Strengths**: Provides physical interpretation of ML predictions; identifies anomalous attribution patterns; spatially explicit.
- **Limitations**: Post-hoc explanation; SHAP computational cost for large datasets; attribution shifts could have multiple causes.
- **Applied to pumping detection**: Indirectly -- anomalous feature attributions (e.g., model relying on unexpected features or showing unexplained residual attribution) could flag areas with hidden pumping.
- Sources: [Clark 2025 WRR](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025WR041303?af=R)

**ShaTS: A Shapley-based Explainability Method for Time Series AI Models Applied to Anomaly Detection. *Future Generation Computer Systems*, 2025.**
- **Method**: Model-agnostic SHAP variant designed specifically for time series. Uses a priori feature grouping preserving temporal dependencies (sensor/actuator grouping, process grouping). Produces coherent and actionable sensor-level attributions for anomaly detection.
- **Data requirements**: Multivariate time series + trained anomaly detection model.
- **Strengths**: Preserves temporal structure in explanations; actionable groupings; model-agnostic.
- **Limitations**: Designed for industrial IoT, not groundwater specifically.
- **Applied to pumping detection**: Transferable -- could explain why an anomaly detector flags certain groundwater patterns as anomalous, attributing to specific features (e.g., lack of rainfall correlation).
- Sources: [ShaTS 2025](https://arxiv.org/html/2506.01450), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167739X25004728)

**Antwarg, L., et al. (2021). Explaining anomalies detected by autoencoders using SHAP. *Expert Systems with Applications*, 186, 115736.**
- **Method**: Applies SHAP to explain anomalies detected by autoencoders. Provides feature-level explanations for why specific data points are flagged as anomalous.
- **Data requirements**: Trained autoencoder + anomalous samples.
- **Strengths**: Bridges black-box anomaly detection with interpretable explanations; identifies which features drive anomaly scores.
- **Applied to pumping detection**: Transferable -- explains why specific groundwater level patterns are flagged as anomalous.
- Sources: [Antwarg 2021 ESWA](https://www.sciencedirect.com/science/article/abs/pii/S0957417421011155)

### E2. Concept Drift Detection via XAI

**DriftGuard: A Hierarchical Framework for Concept Drift Detection and Remediation. arXiv:2601.08928, 2025.**
- **Method**: Five-module framework: baseline establishment, ensemble detection, SHAP-based diagnosis, hierarchical impact assessment, and cost-aware adaptive retraining. Uses SHAP feature importance drift to detect concept drift before it manifests in prediction errors. 97.8% detection recall at 4.2-day mean latency on M5 dataset.
- **Data requirements**: Trained forecasting model + streaming data.
- **Strengths**: Proactive detection via feature attribution changes; SHAP-based root cause diagnosis; hierarchical assessment.
- **Limitations**: Designed for supply chain forecasting; computational overhead of continuous SHAP computation.
- **Applied to pumping detection**: Highly transferable -- if a groundwater prediction model shows drifting SHAP attributions (e.g., decreasing importance of rainfall, increasing unexplained component), this could signal the onset of undeclared pumping changing the system dynamics.
- Sources: [DriftGuard arXiv](https://arxiv.org/abs/2601.08928)

**XDrift: Explainable Concept Drift Detection in Data Streams. 2024.**
- **Method**: Combines four detection algorithms (KS-Test, performance-based, DDM, Page-Hinkley) with XAI (SHAP, LIME, counterfactuals, tree comparison).
- **Data requirements**: Streaming data with model predictions.
- **Strengths**: Multi-method consensus; multiple XAI approaches.
- **Applied to pumping detection**: Transferable.
- Sources: [XDrift ResearchGate](https://www.researchgate.net/publication/399671839)

**Explainability and Interpretability in Concept and Data Drift: A Systematic Literature Review. *Algorithms*, 18(7), 443, 2025.**
- **Method**: Comprehensive survey of XAI techniques applied to drift detection.
- **Strengths**: Overview of the field; identifies research gaps.
- **Applied to pumping detection**: Background resource.
- Sources: [SLR Algorithms 2025](https://www.mdpi.com/1999-4893/18/7/443)

### E3. Integrated Gradients for Temporal Attribution

**Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML 2017*.**
- **Method**: Integrated Gradients computes attributions by integrating gradients of model output w.r.t. inputs along the path from a baseline to the input. Axiomatic (satisfies sensitivity and implementation invariance).
- **Data requirements**: Differentiable model + input data.
- **Strengths**: Theoretically grounded; no model modification needed; applicable to any differentiable model.
- **Limitations**: Requires careful baseline selection; computational cost of integration.
- **Applied to pumping detection**: Transferable -- temporal IG attributions on groundwater prediction models could reveal time steps with anomalous influence, indicating unmodeled pumping events.
- **Open source**: [Captum](https://captum.ai/) (PyTorch), TensorFlow Integrated Gradients.
- Sources: [Captum IG](https://captum.ai/docs/extension/integrated_gradients)

**AXIS: Explainable Time Series Anomaly Detection with Large Language Models. arXiv:2509.24378, 2025.**
- **Method**: Uses LLMs combined with attribution methods (including IG) for explainable anomaly detection in time series. Identifies critical time steps that significantly impact predictions.
- **Data requirements**: Time series + LLM model.
- **Strengths**: Natural language explanations of anomalies; integrates multiple XAI methods.
- **Applied to pumping detection**: Transferable.
- Sources: [AXIS arXiv](https://arxiv.org/html/2509.24378v1)

### E4. Attention Mechanism Analysis

**Temporal Pattern Attention for Multivariate Time Series Forecasting. *Machine Learning*, 108, 2019.**
- **Method**: Attention weights on rows select relevant variables; since the context vector is a weighted sum of row vectors containing information across multiple time steps, it captures temporal patterns. Attention weights are interpretable -- they reveal which variables and time steps the model considers important.
- **Data requirements**: Multivariate time series.
- **Strengths**: Interpretable via attention weights; captures variable selection and temporal patterns simultaneously.
- **Limitations**: Attention weights may not always faithfully represent model reasoning.
- **Applied to pumping detection**: Transferable -- if a groundwater forecasting model trained without pumping data shows anomalous attention patterns (e.g., attending to local rather than regional features), this could indicate unmodeled local stresses.
- Sources: [TPA 2019](https://link.springer.com/article/10.1007/s10994-019-05815-0)

### E5. XAI Combined with Groundwater Analysis

**How much X is in XAI: Responsible use of "Explainable" AI in hydrology and water resources. *Journal of Hydrology: Regional Studies*, 2024.**
- **Method**: Critical review of XAI applications in hydrology, identifying pitfalls and best practices. Warns against over-interpreting SHAP values and other XAI outputs without physical validation.
- **Strengths**: Essential methodological guidance; identifies common mistakes.
- **Applied to pumping detection**: Background resource for responsible XAI-based detection.
- Sources: [XAI Hydrology 2024](https://www.sciencedirect.com/science/article/pii/S2589915524000154)

**Explainable Artificial Intelligence in Hydrology: A Review. *Water Resources Management*, 2025.**
- **Method**: Comprehensive review of XAI methods (SHAP, LIME, Grad-CAM, ICE) applied to hydrological problems including groundwater prediction.
- **Strengths**: Covers LSTM, GRU, CNN with SHAP/LIME; identifies challenges (scalability, domain unevenness).
- **Applied to pumping detection**: Background resource.
- Sources: [XAI Hydrology Review 2025](https://link.springer.com/article/10.1007/s11269-025-04435-9)

---

## F. CAUSAL INFERENCE APPROACHES

### F1. PCMCI / LPCMCI (Tigramite)

**Runge, J., et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances*, 5(11), eaau4996.**
- **Method**: PCMCI combines PC1 condition selection with Momentary Conditional Independence (MCI) test. Two-step: (1) identify superset of parents for each variable, (2) test conditional independence. Much higher detection power than Granger causality for both small and large variable sets.
- **Data requirements**: Multivariate time series (can handle nonlinear dependencies with appropriate conditional independence tests).
- **Strengths**: Handles nonlinear dependencies; scalable; higher detection power than Granger causality; avoids conditioning on irrelevant variables.
- **Limitations**: Assumes causal sufficiency (no hidden confounders) for PCMCI; causal Markov condition must hold.
- **Applied to pumping detection**: Transferable -- if pumping is a latent variable, PCMCI on rainfall/ET/GW time series would show causal links that cannot be explained by observed variables.
- **Open source**: [Tigramite](https://github.com/jakobrunge/tigramite) (Python).
- Sources: [Runge 2019 SciAdv](https://www.science.org/doi/10.1126/sciadv.aau4996)

**LPCMCI (Latent-PCMCI). Runge, J. (2020). Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. *UAI 2020*.**
- **Method**: Extension of PCMCI that explicitly allows for unobserved (latent) time series variables. Detects that certain causal links cannot be explained by observed variables alone, indicating hidden confounders.
- **Data requirements**: Multivariate time series (some variables may be unobserved).
- **Strengths**: Explicitly handles latent confounders; outputs PAG (partial ancestral graph) indicating ambiguous causal relationships.
- **Limitations**: More conservative (fewer claims) due to latent variable allowance; computationally more expensive.
- **Applied to pumping detection**: Highly relevant -- LPCMCI could detect that groundwater levels are influenced by an unobserved variable (undeclared pumping) that is not captured in the observed dataset.
- **Open source**: [Tigramite LPCMCI tutorial](https://github.com/jakobrunge/tigramite/blob/master/tutorials/causal_discovery/tigramite_tutorial_latent-pcmci.ipynb)
- Sources: [LPCMCI tutorial](https://github.com/jakobrunge/tigramite/blob/master/tutorials/causal_discovery/tigramite_tutorial_latent-pcmci.ipynb)

### F2. Granger Causality

**Standard Granger causality tests have been widely applied in hydrology but are limited by:**
- Linear assumption
- Requires stationarity
- All relevant variables must be observed (no latent confounders)
- Low power compared to PCMCI in high-dimensional settings

For groundwater-pumping analysis, Granger causality can test whether lagged pumping data (where available) improves prediction of groundwater levels beyond what is possible from other predictors alone.

### F3. Convergent Cross Mapping (CCM)

**Bonotto, G., Peterson, T.J., Fowler, K., & Western, A.W. (2022). Identifying Causal Interactions Between Groundwater and Streamflow Using Convergent Cross-Mapping. *Water Resources Research*, 58, e2021WR030231.**
- **Method**: CCM detects causality in nonlinear dynamical systems by testing whether the state space of one variable can reconstruct the state space of another (Sugihara et al., 2012). Applied to identify causal interactions between streamflow and groundwater in Victoria, Australia. Detects weaker interactions during Millennium Drought.
- **Data requirements**: Two or more time series (daily recommended); sufficient length for convergence.
- **Strengths**: Works with nonlinear systems (unlike Granger); can detect weak/moderate coupling; direction-specific.
- **Limitations**: Cannot easily distinguish strong unidirectional coupling from bidirectional; requires long time series; results must be interpreted carefully.
- **Applied to pumping detection**: Partially -- directly applied to GW-SW interactions, but could detect causal influence of an unobserved variable (pumping) on groundwater if it breaks expected causal patterns.
- Sources: [Bonotto 2022 WRR](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR030231)

**Detecting hydrological connectivity using causal inference from time series: synthetic and real karstic case studies. *HESS*, 26, 2181-2199, 2022.**
- **Method**: Tests CCM and other causal inference methods on karstic systems.
- **Strengths**: Validates CCM on complex hydrogeological systems.
- Sources: [HESS 2022 karst](https://hess.copernicus.org/articles/26/2181/2022/)

### F4. Causal Discovery with Hidden Confounders

**Causal discovery for time series with latent confounders. Reiser, C. (2022). arXiv:2209.03427.**
- **Method**: Survey and comparison of causal discovery methods that handle latent confounders in time series, including LPCMCI, FCI, and related approaches.
- **Strengths**: Comprehensive comparison; identifies best methods for different scenarios.
- **Applied to pumping detection**: Directly relevant -- identifies methods that can detect the influence of unobserved pumping.
- Sources: [Reiser 2022 arXiv](https://arxiv.org/pdf/2209.03427)

---

## G. REMOTE SENSING / INDIRECT DETECTION

### G1. InSAR for Land Subsidence

**Solari, L., et al. (2022). Review of satellite radar interferometry for subsidence analysis. *Earth-Science Reviews*, 235, 104239.**
- **Method**: InSAR (Interferometric Synthetic Aperture Radar) measures surface deformation at mm precision. Multi-temporal InSAR (MT-InSAR) including PS-InSAR and SBAS characterize land subsidence from groundwater over-abstraction. Subsidence patterns correlate with pumping well locations (cone of depression shape).
- **Data requirements**: SAR satellite imagery (Sentinel-1, ALOS, TerraSAR-X); repeat passes.
- **Strengths**: High spatial resolution; mm-scale precision; non-invasive; covers large areas; independent of ground infrastructure.
- **Limitations**: Requires compressible aquifer layers; signal decorrelation in vegetated areas; measures surface effect, not pumping directly; latency.
- **Applied to pumping detection**: Yes, directly -- InSAR subsidence patterns reveal locations of excessive groundwater extraction. Studies in Iran, Mexico (Aguascalientes >10 cm/year), Beijing, Prato (Italy).
- **Open source**: ISCE, SNAP, StaMPS, MintPy for InSAR processing.
- Sources: [Review InSAR subsidence](https://www.sciencedirect.com/science/article/pii/S0012825222003233), [Iran 2022](https://www.nature.com/articles/s41598-022-17438-y), [Prato 2024](https://www.nature.com/articles/s41598-024-67725-z)

**Advancing remote sensing and machine learning-driven frameworks for groundwater withdrawal estimation in Arizona: Linking land subsidence to groundwater withdrawals. *Hydrological Processes*, 37, e14757, 2023.**
- **Method**: Combines InSAR-derived subsidence with machine learning to estimate groundwater withdrawals.
- **Data requirements**: InSAR data, pumping records for training.
- **Strengths**: Integrates remote sensing with ML for quantitative withdrawal estimation.
- Sources: [Arizona InSAR-ML 2023](https://onlinelibrary.wiley.com/doi/10.1002/hyp.14757)

### G2. GRACE Satellite

**Famiglietti, J.S., et al. (2019). Identifying Climate-Induced Groundwater Depletion in GRACE Observations. *Scientific Reports*, 9, 4947.**
- **Method**: GRACE satellite gravimetry measures total water storage changes at ~300 km resolution. Groundwater storage is estimated by subtracting soil moisture and surface water from total water storage. Separates climatic from human-induced depletion.
- **Data requirements**: GRACE/GRACE-FO data; auxiliary data for water balance partitioning.
- **Strengths**: Global coverage; measures total groundwater storage change; independent of ground infrastructure.
- **Limitations**: Low spatial resolution (~300 km); cannot detect individual well-scale pumping; requires auxiliary data for partitioning.
- **Applied to pumping detection**: Yes -- detects regional over-abstraction (India, California Central Valley, North China Plain, Middle East). Human-induced GW depletion identified with 96% relative contribution in some regions.
- Sources: [Famiglietti 2019 SciRep](https://www.nature.com/articles/s41598-019-40155-y), [GRACE groundwater JPL](https://grace.jpl.nasa.gov/applications/groundwater/)

**Monitoring Groundwater Storage Changes Using GRACE: A Review. *Remote Sensing*, 10(6), 829, 2018.**
- **Method**: Review of GRACE applications to groundwater monitoring.
- Sources: [GRACE Review 2018](https://www.mdpi.com/2072-4292/10/6/829)

### G3. Thermal/NDVI for Detecting Irrigated Areas

**Foster, T., et al. (2020). Satellite-Based Monitoring of Irrigation Water Use: Assessing Measurement Errors. *Water Resources Research*, 56(11), e2020WR028378.**
- **Method**: Thermal infrared satellite imagery (MODIS, Landsat) estimates crop evapotranspiration, converted to consumptive irrigation water use. Errors of 20-100% for groundwater extraction estimates. NDVI >0.4 indicates actively irrigated land.
- **Data requirements**: Satellite imagery (Landsat, Sentinel-2, MODIS); meteorological data.
- **Strengths**: Detects irrigation activity without ground access; identifies areas irrigated from groundwater without permits.
- **Limitations**: High uncertainty (20-100%); cannot directly measure pumping volumes; confounded by rainfed agriculture.
- **Applied to pumping detection**: Yes -- identifies irrigated areas likely relying on undeclared groundwater sources.
- Sources: [Foster 2020 WRR](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020WR028378)

**OpenET consortium (2023-2024). Toward field-scale groundwater pumping using remote sensing. *Agricultural Water Management*.**
- **Method**: OpenET provides 30m-resolution satellite-based ET at daily/monthly/annual scales. Validated against metered groundwater pumping: 7% difference in Diamond Valley, 17% in Harney Basin. Field-scale groundwater pumping depth MAE ~11-14%.
- **Data requirements**: Landsat imagery; ground validation data.
- **Strengths**: Operational platform; field-scale (30m); publicly available; validated against metered data.
- **Limitations**: Still 11-17% error; requires partitioning ET between irrigation and natural sources.
- **Applied to pumping detection**: Yes, directly -- operational tool for monitoring groundwater irrigation use.
- **Availability**: [OpenET](https://etdata.org/) (free online platform).
- Sources: [DRI OpenET](https://www.dri.edu/groundwater-use-can-be-accurately-monitoredwith-satellites-using-openet/), [OpenET NASA](https://www.nasa.gov/image-article/openet-satellite-based-water-data-resource/)

### G4. Satellite + GIS Combined for Illegal Well Detection

**Detection of illegal wells using advanced GIS analysis through Landsat 8 and Sentinel-2 image fusion in Bastam, Iran. *Scientific Reports*, 2025.**
- **Method**: Combines Landsat 8 and Sentinel-2 imagery fusion with advanced GIS spatial analysis (kernel density estimation, Euclidean distance) to map high-risk areas for illegal wells.
- **Data requirements**: Multi-sensor satellite imagery; known well locations for validation.
- **Strengths**: Identifies spatial patterns of illegal extraction; multi-sensor fusion improves detection.
- **Applied to pumping detection**: Yes, directly -- detects illegal wells from satellite indicators.
- Sources: [Iran illegal wells 2025](https://www.nature.com/articles/s41598-025-91188-5)

---

## H. OPERATIONAL / REGULATORY TOOLS

### H1. European Framework (IMPEL TIGDA/WODA Projects)

**TIGDA: Tackling Illegal Groundwater Drilling and Abstractions. IMPEL, EU.**
- **Method**: Guidance document for EU member states providing targeted inspection techniques, practical checklists, and recommendations for detecting and managing illegal drilling and abstraction. Combines command-and-control mechanisms with earth observation methods.
- **Tools**: Random control and penalties; SAR interferometry for subsidence monitoring; evapotranspiration monitoring from satellites.
- **Applied to pumping detection**: Yes, directly -- operational framework for detection and enforcement.
- Sources: [TIGDA IMPEL](https://www.impel.eu/en/topic/water-and-land/water-pollution/projects/tackling-illegal-groundwater-drilling-and-abstractions-tigda)

**WODA: Water Over-Abstraction and Illegal Abstraction Detection and Assessment. IMPEL, 2015.**
- **Method**: EU project developing methods for detecting over-abstraction and illegal abstraction.
- Sources: [WODA IMPEL](https://impel.eu/en/topic/water-and-land/water-pollution/projects/water-over-abstraction-and-illegal-abstraction-detection-and-assessment-woda)

### H2. French Regulatory Framework

**SDAGE (Schéma Directeur d'Aménagement et de Gestion des Eaux)**: 6-year water management planning documents defining major guidelines for preserving aquatic environments and managing water quantity/quality. The water police must refer to SDAGE provisions for any authorization.

**SAGE (Schéma d'Aménagement et de Gestion des Eaux)**: Local implementation of SDAGE. Can prohibit new withdrawals in over-exploited zones.

**Police de l'eau**: All drilling and pumping is subject to declaration. Unauthorized withdrawals are monitored by prefectoral services.

**Zone de Répartition des Eaux (ZRE)**: Special protection zones where water resources are insufficient to meet demand.
- Sources: [SDAGE overview](https://www.eau-loire-bretagne.fr/sites/sdage-sage/home/le-sdage-2022-2027/quest-ce-que-le-sdage.html), [Police de l'eau Gard](https://www.gard.gouv.fr/Actions-de-l-Etat/Environnement/Eaux-et-milieux-aquatiques/Reglementation/Police-de-l-eau2/Prelevements-d-eau)

### H3. California SGMA

**Sustainable Groundwater Management Act (SGMA), 2014.**
- **Framework**: Requires Groundwater Sustainability Agencies (GSAs) for high/medium priority basins. GSPs must include measurable objectives and monitoring networks. Six undesirable results defined: level declines, storage reduction, seawater intrusion, water quality degradation, subsidence, surface water depletion.
- **Monitoring tools**: Continuous well-level monitoring; InSAR for subsidence; SGMApy for sustainability metrics.
- **Metering**: State Water Resources Control Board adopted resolution for measuring and reporting groundwater pumping (2016-2017).
- **Open source**: [SGMApy](https://www.usgs.gov/centers/california-water-science-center/science/sgmapy-open-source-platform-computing) (USGS, Python).
- Sources: [SGMA USGS](https://ca.water.usgs.gov/sustainable-groundwater-management/), [SGMApy](https://www.usgs.gov/centers/california-water-science-center/science/sgmapy-open-source-platform-computing)

### H4. Australian Water Plans

**Groundwater Regulation, Compliance and Enforcement in NSW, Australia. Springer, 2020.**
- **Framework**: Natural Resources Access Regulator (NRAR) for enforcement. National Metering Standards under National Water Initiative (NWI) and Murray-Darling Basin Compliance Compact. Victoria: 91% groundwater metered; ~1,800 observation bores with some telemetry. NSW, QLD, VIC, SA all installing telemetry instruments in at-risk groundwater areas.
- **Key tools**: Telemetry (4G, LTE-M, NB-IoT, LoRa); smart metering; satellite monitoring.
- **Lessons**: Compliance enforcement has oscillated between under-resourced low priority and national reform primacy.
- Sources: [NSW Groundwater Regulation](https://link.springer.com/chapter/10.1007/978-3-030-32766-8_22), [Victorian compliance](https://www.water.vic.gov.au/our-programs/murray-darling-basin/compliance), [Non-urban metering](https://www.dcceew.gov.au/water/policy/policy/nwi/nonurban-water-metering-framework)

### H5. IoT and Real-Time Monitoring Systems

**Low-Cost, Open Source Wireless Sensor Network for Real-Time, Scalable Groundwater Monitoring. *Water*, 12(4), 1066, 2020.**
- **Method**: Open-source wireless sensor network for real-time groundwater monitoring. Uses low-cost sensors with wireless communication (LoRa, Zigbee, Bluetooth BLE, NB-IoT) transmitting to cloud platforms.
- **Data requirements**: Deployed sensors in monitoring wells.
- **Strengths**: Real-time data streaming; low cost; scalable; eliminates manual data collection.
- **Limitations**: Requires sensor deployment and maintenance; connectivity in remote areas.
- **Applied to pumping detection**: Enables high-frequency data collection that makes algorithmic pumping detection feasible.
- Sources: [Low-cost WSN 2020](https://www.mdpi.com/2073-4441/12/4/1066), [Royal Eijkelkamp telemetry](https://www.royaleijkelkamp.com/solutions/for-applications/data/telemetry/)

---

## REVIEW PAPERS AND SURVEYS

**Duran-Llacer, I., et al. (2025). A systematic review of machine learning in groundwater monitoring. *Environmental Modelling & Software*, 206, 106549.**
- **Method**: Reviews 20 years of ML applications in groundwater monitoring. Covers clustering, time series forecasting, PCA, RNNs, evolutionary algorithms.
- **Strengths**: Comprehensive; identifies trends and gaps.
- Sources: [Systematic review ML GW 2025](https://www.sciencedirect.com/science/article/pii/S1364815225002336)

**Valiente, M., et al. (2022). Review of Groundwater Withdrawal Estimation Methods. *Water*, 14(17), 2762.**
- **Method**: Reviews 34 journal articles (1970-2021) on GW withdrawal estimation methods. Categorizes into direct (metering) and indirect (water balance, remote sensing, modeling) approaches. Provides systematic guide.
- **Strengths**: Comprehensive categorization; identifies advantages/disadvantages of each approach.
- Sources: [GW Withdrawal Review 2022](https://www.mdpi.com/2073-4441/14/17/2762)

**Adams, K.A., et al. (2022). Remote Sensing of Groundwater: Current Capabilities and Future Directions. *Water Resources Research*, 58, e2022WR032219.**
- **Method**: Reviews remote sensing technologies for groundwater including GRACE, InSAR, thermal imagery, and integrated approaches.
- Sources: [RS GW Review 2022](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022WR032219)

**Retike, I., et al. (2022). Rescue of groundwater level time series: How to visually identify and treat errors. *Journal of Hydrology*, 605, 127294.**
- **Method**: Systematic approach to identifying and correcting errors in groundwater level time series. Groups errors into measurement/recording errors, technical problems, and local anthropogenic impacts (including pumping). Applied to Latvian national database (1.68M observations).
- **Strengths**: Practical guidance; large-scale application; identifies pumping artifacts as a specific error category.
- **Open source**: R-Shiny web interface for visual identification.
- Sources: [Retike 2022 JoH](https://www.sciencedirect.com/science/article/pii/S0022169421013445)

**Peterson, T.J., Western, A.W., & Cheng, X. (2018). The good, the bad and the outliers: automated detection of errors and outliers from groundwater hydrographs. *Hydrogeology Journal*, 26, 371-380.**
- **Method**: Automated approach requiring only the observed hydrograph. Identifies errors and outliers without requiring knowledge of hydrogeology. Distinguishes "good" outliers (biophysical insights) from "bad" errors (monitoring failures).
- **Strengths**: Automated; requires minimal data; available in HydroSight.
- **Limitations**: Under-estimates outliers where variance is non-stationary; over-estimates where trend increases rapidly.
- **Open source**: HydroSight (MATLAB).
- Sources: [Peterson 2018 HJ](https://link.springer.com/article/10.1007/s10040-017-1660-7)

---

## I. AVAILABLE DATA SOURCES

### I1. France — BNPE × ADES Cross-Referencing

**BNPE (Banque Nationale des Prélèvements en Eau)**: Annual withdrawal volumes per facility with geolocation.
- **API**: `https://hubeau.eaufrance.fr/api/v1/prelevements/`
- **Endpoints**: `/chroniques` (annual volumes), `/ouvrages` (facilities with lat/lon), `/points_prelevement`
- **Limitation**: Annual resolution only — no daily/monthly pumping time series.

**ADES**: National piezometric database via Hub'Eau Piézométrie API.
- **API**: `https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/`
- **Key**: Each station has `codes_bdlisa` linking to BDLISA aquifer entities.

**Cross-referencing strategy**:
1. Query BNPE `/ouvrages` for withdrawal facilities in a department → lat/lon + `code_ouvrage`
2. Query ADES `/stations` for piezometers in the same area → lat/lon + `codes_bdlisa`
3. Spatial join: find BNPE ouvrages within X km of ADES piezometers
4. Aquifer filter via BDLISA to ensure both tap the same aquifer entity
5. Pull chroniques from both: BNPE annual volumes + ADES piezometric levels

**R package `hubeau`** (INRAE) wraps both APIs: `https://inrae.github.io/hubeau/`

### I2. International Datasets

| Source | Data Type | Resolution | Access |
|--------|-----------|------------|--------|
| **DINOloket (NL)** | Levels + pumping tests | Varies | REST/SOAP |
| **California SGMA** | Continuous levels | 15-min | Download |
| **Australia NGIS** | Levels + extraction | Annual | Download |
| **USGS (USA)** | Levels + some pumping | Daily-Annual | REST |

### I3. Key Limitation
**No publicly available high-frequency (daily) pumping time series paired with nearby piezometric levels.** BNPE annual data establishes which piezometers are near pumping wells, but temporal disaggregation is needed for ML modeling. Pumping data used in only **1.7%** of groundwater ML studies (Sahoo et al., WRR 2017).

### I4. Best French Case Study: Nappe de Beauce
~3,600 agricultural operations, volumetric management since 1999, significant extraction reductions since 2003. Best-documented pumping impact on French piezometry.
- INRAE study: `https://hal.inrae.fr/hal-02595627`

---

## J. XAI COMPARISON PROTOCOLS (From Adjacent Literature)

### J1. Metrics for Comparing Feature Attributions Between Models

| Metric | Usage | Source |
|--------|-------|--------|
| **Feature Agreement (FA)** | % features in same top-K | Koenen & Wright, xAI 2024 |
| **Rank Agreement (Spearman/Kendall)** | Correlation of feature rankings | Koenen & Wright, xAI 2024 |
| **NDCG** | Weighted ranking divergence | Amazon SageMaker (production) |
| **KL / Jensen-Shannon divergence** | Distribution-level SHAP comparison | Google Cloud (production) |
| **SHAP interaction values (2nd order)** | Synergies/redundancies detection | Vater et al., ESANN 2025 |
| **Autocorrelation of residuals (ACF/PACF)** | Missing variable signature | arXiv 2402.01000, 2024 |

### J2. Concept Drift Detection via XAI

**Duckworth et al. (Scientific Reports 2021)**: Trained classifier on pre-COVID data, monitored SHAP values during pandemic → detected drift *before* accuracy degraded. Protocol: track relative variation of feature SHAP values vs. global importance baseline.

**Hinder et al. (Neurocomputing 2023, 2024)**: Formal characterization of concept drift via feature attribution changes. Defines feature-wise notion of drift enabling semantic interpretation.

### J3. Omitted Variable Detection

**arXiv 2402.01000 (2024)**: Residual autocorrelation in deep learning forecasts is directly caused by missing covariates. Lag-1 autocorrelation = signature of missing variables.

**Chernozhukov et al. (Review of Econ & Stats 2024, Best Paper)**: OVB framework for ML — sensitivity analysis to bound impact of omitted variable.

---

## SYNTHESIS AND RESEARCH GAPS

### Methods That DIRECTLY Address Pumping Detection from Piezometric Data:
1. **ICA for pumping characterization** (Hsieh 2015) -- most directly applicable
2. **HHT + EOF for spatiotemporal pumping estimation** (Hsieh 2023, 2024) -- state of the art for signal decomposition
3. **TFN residual analysis** (Pastas, HydroSight) -- well-established, operational
4. **Lagging theory response function without pumping data** (Lin 2024) -- novel, no pumping data needed
5. **InSAR subsidence mapping** -- operational for regional detection
6. **OpenET satellite ET** -- operational for irrigation monitoring

### Methods Highly Transferable to Pumping Detection:
1. **LPCMCI latent variable detection** (Tigramite) -- could detect unobserved pumping as latent confounder
2. **BEAST changepoint + trend detection** -- detects abrupt changes in groundwater regime
3. **LSTM autoencoders / Anomaly Transformer** -- unsupervised anomaly detection
4. **SHAP-based concept drift / DriftGuard** -- detects shifts in model behavior caused by new stresses
5. **GNN spatial anomaly detection** -- exploits disrupted spatial correlations

### Key Research Gap -- Your Novel Angle (XAI-Based Detection):
No published work has specifically combined XAI attribution analysis (SHAP, TimeSHAP, Integrated Gradients) with groundwater forecasting models to detect undeclared pumping. The closest works are:
- Clark 2025 (SHAP for groundwater interpretation, not anomaly detection)
- DriftGuard 2025 (concept drift via SHAP, but in supply chain)
- ShaTS 2025 (time series SHAP for anomaly detection, but in industrial IoT)

**The combination of training a groundwater prediction model on "clean" data, then using XAI attribution shifts to detect when new, unmodeled stresses (pumping) appear, represents a genuine gap in the literature.** This would merge the physical interpretability of TFN approaches with the flexibility of deep learning and the diagnostic power of XAI.

Sources:
- [Pastas (Collenteur 2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6899905/)
- [HydroSight](https://github.com/peterson-tim-j/HydroSight)
- [Von Asmuth 2002](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2001WR001136)
- [Lin 2024 WRR](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR036747)
- [Hsieh 2024 JoH](https://www.sciencedirect.com/science/article/abs/pii/S0022169424002506)
- [Hsieh 2023 JoH](https://www.sciencedirect.com/science/article/abs/pii/S0022169423006510)
- [Hsieh 2015 JoH](https://www.sciencedirect.com/science/article/abs/pii/S0022169415003637)
- [ICA-WTC 2022](https://www.sciencedirect.com/science/article/abs/pii/S0022169422014846)
- [GARDENIA BRGM](https://www.brgm.fr/en/software/gardenia-lumped-hydrological-modelling-catchment-basin)
- [TIGRE BRGM](https://www.brgm.fr/en/software/tigre-easy-calculation-influence-well-field-homogeneous-aquifer)
- [OUAIP BRGM](http://ouaip.brgm.fr/)
- [MétéEAU Nappes](https://meteeaunappes.brgm.fr/en)
- [Zaadnoordijk 2019](https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/gwat.12819)
- [Peterson 2018](https://link.springer.com/article/10.1007/s10040-017-1660-7)
- [Retike 2022](https://www.sciencedirect.com/science/article/pii/S0022169421013445)
- [Bakker 2019](https://ngwa.onlinelibrary.wiley.com/doi/10.1111/gwat.12927)
- [Liu 2023 (Chengdu)](https://www.nature.com/articles/s41598-023-38447-5)
- [Rezaiezadeh 2024](https://link.springer.com/article/10.1007/s10661-024-12848-z)
- [Anomaly Transformer (ICLR 2022)](https://arxiv.org/abs/2110.02642)
- [TranAD (VLDB 2022)](https://dl.acm.org/doi/abs/10.14778/3514061.3514067)
- [ST-GNN groundwater 2024](https://www.nature.com/articles/s41598-024-75385-2)
- [GNN river anomaly 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC11162521/)
- [Clark 2025 WRR](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025WR041303?af=R)
- [ShaTS 2025](https://www.sciencedirect.com/science/article/pii/S0167739X25004728)
- [DriftGuard 2025](https://arxiv.org/abs/2601.08928)
- [Antwarg 2021](https://www.sciencedirect.com/science/article/abs/pii/S0957417421011155)
- [XAI Hydrology 2024](https://www.sciencedirect.com/science/article/pii/S2589915524000154)
- [XAI Hydrology Review 2025](https://link.springer.com/article/10.1007/s11269-025-04435-9)
- [BEAST (Zhao 2019)](https://www.sciencedirect.com/science/article/abs/pii/S0034425719301853)
- [UK GW drought BEAST 2024](https://www.sciencedirect.com/science/article/abs/pii/S0022169424008254)
- [Runge 2019 PCMCI](https://www.science.org/doi/10.1126/sciadv.aau4996)
- [Tigramite](https://github.com/jakobrunge/tigramite)
- [LPCMCI tutorial](https://github.com/jakobrunge/tigramite/blob/master/tutorials/causal_discovery/tigramite_tutorial_latent-pcmci.ipynb)
- [Bonotto 2022 CCM](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR030231)
- [HESS 2022 karst CCM](https://hess.copernicus.org/articles/26/2181/2022/)
- [InSAR subsidence review 2022](https://www.sciencedirect.com/science/article/pii/S0012825222003233)
- [InSAR Iran 2022](https://www.nature.com/articles/s41598-022-17438-y)
- [InSAR Prato 2024](https://www.nature.com/articles/s41598-024-67725-z)
- [InSAR-ML Arizona 2023](https://onlinelibrary.wiley.com/doi/10.1002/hyp.14757)
- [GRACE review 2018](https://www.mdpi.com/2072-4292/10/6/829)
- [Famiglietti 2019 GRACE](https://www.nature.com/articles/s41598-019-40155-y)
- [Foster 2020 satellite irrigation](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020WR028378)
- [OpenET](https://etdata.org/)
- [Iran illegal wells 2025](https://www.nature.com/articles/s41598-025-91188-5)
- [TIGDA IMPEL](https://www.impel.eu/en/topic/water-and-land/water-pollution/projects/tackling-illegal-groundwater-drilling-and-abstractions-tigda)
- [WODA IMPEL](https://impel.eu/en/topic/water-and-land/water-pollution/projects/water-over-abstraction-and-illegal-abstraction-detection-and-assessment-woda)
- [SGMA USGS](https://ca.water.usgs.gov/sustainable-groundwater-management/)
- [SGMApy](https://www.usgs.gov/centers/california-water-science-center/science/sgmapy-open-source-platform-computing)
- [NSW Groundwater Regulation](https://link.springer.com/chapter/10.1007/978-3-030-32766-8_22)
- [Australian metering](https://www.dcceew.gov.au/water/policy/policy/nwi/nonurban-water-metering-framework)
- [GW Withdrawal Review 2022](https://www.mdpi.com/2073-4441/14/17/2762)
- [RS GW Review 2022](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022WR032219)
- [ML GW Systematic Review 2025](https://www.sciencedirect.com/science/article/pii/S1364815225002336)
- [Mann-Kendall Fang 2019](https://ngwa.onlinelibrary.wiley.com/doi/10.1111/gwat.12800)
- [Captum](https://captum.ai/)
- [FCVAE 2024](https://dl.acm.org/doi/10.1145/3589334.3645710)
- [MRCPtool](https://www.sciencedirect.com/science/article/abs/pii/S0098300419301025)
- [Spectral analysis aquifers](https://link.springer.com/article/10.1007/s00477-002-0106-4)
- [SSA Golyandina 2013](https://link.springer.com/book/10.1007/978-3-642-34913-3)
- [MSSA groundwater 2019](https://www.tandfonline.com/doi/full/10.1080/02626667.2019.1669793)
- [Low-cost WSN 2020](https://www.mdpi.com/2073-4441/12/4/1066)
- [Salmoral 2025 Water Balance](https://www.mdpi.com/2071-1050/17/12/5618)