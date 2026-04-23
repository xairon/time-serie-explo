# Advanced Analytics & Domain Tools — Design Spec

**Date**: 2026-04-23
**Scope**: 8 new analytical features for the Pastas pipeline
**Status**: Draft

## Problem

The Pastas pipeline provides core model fitting and basic diagnostics, but lacks several analytical tools that hydrogeologists expect: confidence intervals on predictions, recession analysis, baseflow separation, spectral validation, signal decomposition, cross-correlation analysis, multi-station comparison, and input data quality screening.

## Goals

1. Add 8 analytical features organized into 3 coherent groups
2. Follow the existing architecture: pure Python utils → FastAPI endpoints → React components
3. All analyses are lazy (computed on demand via React Query, not at fit time)
4. Integrate into the existing FitResultsPanel as new collapsible sections

## Non-Goals

- Modifying the core Pastas fitting pipeline
- Real-time computation during fitting
- Replacing existing diagnostics (QQ plot, PACF, etc.)

---

## Architecture

### Common Pattern

Each feature follows the same structure:

```
dashboard/utils/pastas/<module>.py     → pure Python computation
api/routers/pastas.py                  → GET endpoint (lazy, cached)
frontend/src/hooks/usePastas.ts        → React Query hook
frontend/src/components/pastas/<X>.tsx  → visualization component
frontend/src/components/pastas/FitResultsPanel.tsx → integration
```

All endpoints load the model via existing `load_model(run_id)` (LRU-cached) and extract `tmin`/`tmax`/`code_bss` from MLflow params/tags.

---

## Sub-project A: Pastas Advanced Analytics

### A1. Confidence Intervals

**Backend**: `dashboard/utils/pastas/confidence_intervals.py`

```python
def compute_confidence_bands(
    model, tmin: str, tmax: str, n_bootstrap: int = 200
) -> dict:
```

Method: Bootstrap residuals — resample residuals with replacement 200×, add each sample to the simulation, compute percentile 5/25/75/95 at each timestep.

Returns:
```python
{
    "index": ["2015-01-01", ...],
    "p5": [10.1, ...], "p25": [10.3, ...],
    "p75": [10.8, ...], "p95": [11.0, ...],
    "simulation": [10.5, ...],
}
```

**Endpoint**: `GET /api/v1/pastas/models/{run_id}/confidence-bands`

**Frontend**: No new component — modify the existing Observed vs Simulated chart in FitResultsPanel to overlay shaded bands (P5-P95 light, P25-P75 darker) when data is available. Fetched lazily alongside existing data.

---

### A2. Recession Analysis

**Backend**: `dashboard/utils/pastas/recession.py`

```python
def compute_recession_analysis(
    model, tmin: str, tmax: str,
    min_duration_days: int = 30,
    min_drop_m: float = 0.05,
) -> dict:
```

Steps:
1. Extract observed series, compute daily derivative
2. Identify recession segments: consecutive days with derivative < 0, lasting ≥ min_duration_days, total drop ≥ min_drop_m
3. For each segment: fit h(t) = h0 × exp(-t/T) via least squares → extract T (time constant in days)
4. Compute Master Recession Curve: normalize all segments to start at h=1, overlay, fit aggregate exponential
5. Summary stats: T_median, T_mean, T_std, n_segments

Returns:
```python
{
    "segments": [
        {"start": "2016-03-01", "end": "2016-05-15", "h0": 12.3, "h_end": 11.8,
         "T_days": 45.2, "r_squared": 0.94, "duration_days": 75},
        ...
    ],
    "mrc": {"normalized_time": [0, 1, 2, ...], "normalized_level": [1.0, 0.98, ...]},
    "T_median_days": 42.0,
    "T_mean_days": 48.5,
    "T_std_days": 15.2,
    "n_segments": 8,
}
```

**Endpoint**: `GET /api/v1/pastas/models/{run_id}/recession`

**Frontend**: `RecessionPanel.tsx`
- Left: MRC plot (normalized time vs normalized level, all segments in gray, aggregate in cyan)
- Right: histogram of T values with median line
- Below: table of segments (start, end, T, R², duration)

---

### A3. Baseflow Index (BFI)

**Backend**: `dashboard/utils/pastas/baseflow.py`

```python
def compute_baseflow(
    model, tmin: str, tmax: str, alpha: float = 0.925
) -> dict:
```

Method: Lyne & Hollick recursive digital filter (one-parameter):
```
quickflow[t] = alpha * quickflow[t-1] + (1+alpha)/2 * (obs[t] - obs[t-1])
baseflow[t] = obs[t] - max(0, quickflow[t])
```
Three forward passes for stability.

Returns:
```python
{
    "bfi": 0.72,  # baseflow / total ratio
    "index": ["2015-01-01", ...],
    "observed": [10.5, ...],
    "baseflow": [10.2, ...],
    "quickflow": [0.3, ...],
}
```

**Endpoint**: `GET /api/v1/pastas/models/{run_id}/baseflow`

**Frontend**: `BaseflowPanel.tsx`
- Stacked area chart: baseflow (blue) + quickflow (cyan) = observed (gray line overlay)
- BFI value displayed as a KPI badge: "BFI: 0.72 — dominated by baseflow"

---

### A4. Spectral Analysis

**Backend**: `dashboard/utils/pastas/spectral.py`

```python
def compute_spectral_analysis(
    model, tmin: str, tmax: str
) -> dict:
```

Method: `scipy.signal.welch` on observed and simulated series (daily, nperseg=365).

Returns:
```python
{
    "frequencies": [0.001, 0.002, ...],  # cycles/day
    "periods_days": [1000, 500, ...],
    "psd_observed": [1.2e-3, ...],
    "psd_simulated": [1.1e-3, ...],
    "coherence": [0.95, 0.92, ...],  # optional: magnitude-squared coherence
}
```

**Endpoint**: `GET /api/v1/pastas/models/{run_id}/spectral`

**Frontend**: `SpectralPanel.tsx`
- Log-log plot: PSD observed (gray) vs PSD simulated (cyan)
- Vertical annotation lines at key periods: 365d (annual), 182d (semi-annual), 30d (monthly)
- Color bands: "interannual" (>1yr), "seasonal" (6m-1yr), "event" (<6m)

---

## Sub-project B: Signal Analysis & Multi-station

### B5. STL Decomposition

**Backend**: `dashboard/utils/pastas/signal_decomposition.py`

```python
def compute_stl_decomposition(
    model, tmin: str, tmax: str, period: int = 12
) -> dict:
```

Method: `statsmodels.tsa.seasonal.STL` on monthly resampled observed series.

Returns:
```python
{
    "index": ["2015-01-01", ...],
    "observed": [10.5, ...],
    "trend": [10.4, ...],
    "seasonal": [0.1, ...],
    "residual": [0.0, ...],
    "trend_strength": 0.85,  # 1 - var(residual) / var(trend + residual)
    "seasonal_strength": 0.72,
}
```

**Endpoint**: `GET /api/v1/pastas/models/{run_id}/decomposition`

**Frontend**: `DecompositionPanel.tsx`
- 4 stacked subplots (shared x-axis): Observed, Trend, Seasonal, Residual
- KPI badges: trend strength, seasonal strength

---

### B6. Cross-correlogram

**Backend**: `dashboard/utils/pastas/cross_correlation.py`

```python
def compute_cross_correlation(
    model, code_bss: str, tmin: str, tmax: str, engine
) -> dict:
```

Method:
1. Resample observed piezo and precipitation to monthly
2. Compute normalized cross-correlation (scipy.signal.correlate, mode='full')
3. Identify lag of maximum correlation
4. Compare to Pastas T95 if step_response is available

Returns:
```python
{
    "lags_months": [-24, -23, ..., 0, ..., 23, 24],
    "correlation": [0.01, 0.03, ..., 0.82, ..., 0.05, 0.02],
    "max_lag_months": 2,
    "max_correlation": 0.82,
    "t95_months": 3.5,  # from Pastas step response, null if unavailable
}
```

**Endpoint**: `GET /api/v1/pastas/models/{run_id}/cross-correlation`

**Frontend**: `CrossCorrelationPanel.tsx`
- Bar chart of correlation vs lag (negative lags = precip leads piezo)
- Vertical line at lag of max correlation (annotated)
- If T95 available: second vertical line for comparison
- Text: "Peak response at {N} months — Pastas T95: {M} months"

---

### B7. Multi-station Residual Comparison

**Backend**: `dashboard/utils/pastas/multi_station_residuals.py`

```python
def compute_regional_residuals(
    code_bss: str, residuals_monthly: pd.Series, engine
) -> dict:
```

Method:
1. Get BDLISA siblings (reuse existing sibling lookup)
2. For each sibling: fetch monthly niveau_moyen, compute z-score per calendar month
3. For the current station: use model residuals normalized as z-scores
4. Return aligned time series for overlay

Returns:
```python
{
    "index": ["2015-01-01", ...],
    "model_residual_zscore": [0.5, -1.2, ...],
    "siblings": [
        {"code_bss": "XXX/P1", "zscore": [0.3, -1.0, ...]},
        ...
    ],
    "correlation_with_siblings": 0.65,  # mean correlation model residuals vs sibling z-scores
}
```

**Endpoint**: `GET /api/v1/pastas/models/{run_id}/regional-residuals`

**Frontend**: `RegionalResidualsPanel.tsx`
- Line chart: model residual z-score (thick cyan) + sibling z-scores (thin gray lines)
- When model and siblings co-move → regional signal (annotation)
- When model diverges from siblings → local issue (annotation)
- KPI: correlation coefficient

---

## Sub-project C: AI Input Quality

### C8. Anomaly Detection

**Backend**: `dashboard/utils/pastas/input_quality.py`

```python
def detect_input_anomalies(
    code_bss: str, engine,
    contamination: float = 0.05,
) -> dict:
```

Method:
1. Fetch monthly climate + piezo data
2. Build feature matrix: [precip, temp, ETP, niveau, Δniveau, precip_rolling_3m, niveau_rolling_3m]
3. Fit Isolation Forest (contamination=5%, n_estimators=200)
4. Return anomaly scores and flagged months

Returns:
```python
{
    "months": ["2015-01-01", ...],
    "scores": [-0.1, -0.3, ...],  # negative = more anomalous
    "flagged": [
        {"month": "2018-07-01", "score": -0.52, "reason": "precip=0mm with +0.3m level rise"},
        ...
    ],
    "n_flagged": 5,
    "n_total": 120,
}
```

**Endpoint**: `POST /api/v1/pastas/diagnose` — enrich existing response with new `anomalous_months` field

**Frontend**: Modify existing `PreFitDiagnosticPanel` to add an "Input anomalies" indicator showing flagged months with scores. Clicking expands a list of suspicious months with short explanations.

---

## Files to Create

| File | Responsibility |
|------|----------------|
| `dashboard/utils/pastas/confidence_intervals.py` | Bootstrap confidence bands |
| `dashboard/utils/pastas/recession.py` | Recession analysis + MRC |
| `dashboard/utils/pastas/baseflow.py` | Lyne & Hollick BFI |
| `dashboard/utils/pastas/spectral.py` | PSD comparison |
| `dashboard/utils/pastas/signal_decomposition.py` | STL decomposition |
| `dashboard/utils/pastas/cross_correlation.py` | Cross-correlogram |
| `dashboard/utils/pastas/multi_station_residuals.py` | Regional residual comparison |
| `dashboard/utils/pastas/input_quality.py` | Isolation Forest anomaly detection |
| `frontend/src/components/pastas/RecessionPanel.tsx` | Recession UI |
| `frontend/src/components/pastas/BaseflowPanel.tsx` | Baseflow UI |
| `frontend/src/components/pastas/SpectralPanel.tsx` | Spectral UI |
| `frontend/src/components/pastas/DecompositionPanel.tsx` | STL UI |
| `frontend/src/components/pastas/CrossCorrelationPanel.tsx` | Cross-correlation UI |
| `frontend/src/components/pastas/RegionalResidualsPanel.tsx` | Multi-station UI |
| `tests/pastas/test_confidence_intervals.py` | Tests |
| `tests/pastas/test_recession.py` | Tests |
| `tests/pastas/test_baseflow.py` | Tests |
| `tests/pastas/test_spectral.py` | Tests |
| `tests/pastas/test_signal_decomposition.py` | Tests |
| `tests/pastas/test_cross_correlation.py` | Tests |
| `tests/pastas/test_multi_station_residuals.py` | Tests |
| `tests/pastas/test_input_quality.py` | Tests |

## Files to Modify

| File | Change |
|------|--------|
| `api/routers/pastas.py` | 7 new GET endpoints + enrich POST /diagnose |
| `frontend/src/hooks/usePastas.ts` | 7 new React Query hooks |
| `frontend/src/lib/api.ts` | 7 new API client methods |
| `frontend/src/components/pastas/FitResultsPanel.tsx` | Import + render 6 new Sections + confidence bands on time series chart |
| `frontend/src/components/pastas/PreFitDiagnosticPanel.tsx` | Add anomaly detection indicator |

## Performance Considerations

- All endpoints use lazy computation via React Query (user clicks a section → data loads)
- Model loading is LRU-cached (maxsize=32)
- Bootstrap (200 iterations) takes ~2-5s — acceptable for lazy load
- STL/spectral/cross-correlation are sub-second on monthly data
- Isolation Forest fit is <1s on typical station data (60-600 months)
- Regional residuals reuse the sibling lookup pattern from outlier diagnostics
