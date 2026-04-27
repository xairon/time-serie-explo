# Pumping Detection — Design Spec

**Date**: 2026-03-12
**Status**: Draft
**Scope**: Phase 1 — Single-station unsupervised pumping detection with 3-layer hybrid pipeline

## 1. Problem Statement

Illegal or undeclared groundwater pumping is a major issue in France (and globally). No publicly available tool combines physics-based modeling, deep learning explainability, and contrastive embeddings to detect hidden pumping from standard piezometric monitoring data.

**Goal**: Build a new page in Junon that takes an existing dataset (station BRGM + covariables) and runs a 3-layer unsupervised detection pipeline to identify temporal windows where an unmodeled stress (likely pumping) is affecting the piezometric signal.

**Novel contribution**: No published work combines XAI attribution drift with TFN residual analysis and contrastive embeddings for pumping detection. This is a genuine research gap (see `docs/SOTA_pumping_detection.md` for full literature review).

## 2. Architecture Overview

```
Existing dataset (from Data page)
  station piézo + precip + temp + ETP
            │
            ▼
    ┌───────────────────┐
    │  Analysis Pipeline │  ← SSE stream to frontend
    │  (sequential+parallel) │
    └───────┬───────────┘
            │
            ▼
        Layer 1 (PHYSICS)     ← runs first (Layer 2 depends on it)
            │
    ┌───────┴───────┐
    ▼               ▼
 Layer 2          Layer 3       ← run in parallel
 ML+XAI          EMBEDDINGS
 (needs clean     (independent)
  periods from
  Layer 1)
    │               │
    └───────┬───────┘
            ▼
      Fusion & Scoring
            │
            ▼
    Detection Results
```

### Execution Order
1. **Layer 1 first** — Pastas calibration + residuals + change points. Produces clean period mask.
2. **Layer 2 and Layer 3 in parallel** — Layer 2 uses clean periods from Layer 1. Layer 3 is fully independent.
3. **Fusion** — After all layers complete (or after available layers complete if some failed).

### Layer Availability & Graceful Degradation
Each layer is **optional**. The fusion engine adapts to the number of available layers:
- **3 layers available**: 3/3 = HIGH, 2/3 = MEDIUM, 1/3 = LOW.
- **2 layers available** (e.g., SoftCLT not ready): 2/2 = HIGH, 1/2 = MEDIUM.
- **1 layer available** (e.g., only Pastas): Results shown but fusion score replaced by single-layer confidence.
- **0 layers**: Pipeline fails with actionable error message.

Layer 3 (embeddings) depends on SoftCLT/TS2Vec which is currently in development. **Phase 1 ships with Layer 3 as optional** — if the encoder is unavailable, the pipeline runs Layers 1+2 only and the embedding panel shows "Encoder not available" in the UI.

### Error Handling Per Layer
| Layer | Failure mode | Fallback |
|-------|-------------|----------|
| Layer 1 — Pastas | EVP < 30% or solver diverges | Flag "poor physics model" in results. Still extract residuals (even if noisy). Layers 2+3 continue. Clean period selection falls back to seasonal heuristic (Nov-Mar = clean). |
| Layer 1 — BEAST/PELT | Rbeast install fails or crashes on gappy data | Fall back to PELT only (ruptures is pure Python, more robust). If both fail, skip changepoint detection, use ACF only. |
| Layer 2 — Clean period | No clean windows found (all periods have high residuals) | Iteratively relax threshold (2σ → 3σ → 4σ). If still < 1 year of clean data, skip Layer 2 with warning. |
| Layer 2 — TFT training | Insufficient clean data or training diverges | Skip Layer 2, report training failure. Fusion runs on Layers 1+3. |
| Layer 3 — Encoder | SoftCLT/TS2Vec not installed or model not trained | Skip Layer 3 entirely. Fusion adapts to 2-layer mode. |
| Layer 3 — Twins | No twin stations available in embedding pool | Skip twin comparison, use only single-station drift analysis. |

### Layer 1 — Physics (Pastas + Change Point Detection)

**Input**: Piezometric level + precipitation + ETP time series.

**Process**:
1. Calibrate Pastas TFN model with `RechargeModel` (precip + ETP) on the full series.
2. Extract residuals (observed - modeled).
3. Compute ACF/PACF on residuals — structured autocorrelation = missing stress signature.
4. Run BEAST (Bayesian change point detection) on residuals to locate temporal breakpoints.
5. Optionally run PELT as a second opinion.

**Output**:
- `residuals`: pd.Series — Pastas residuals time series.
- `acf_stats`: dict — ACF/PACF values + Ljung-Box test p-values.
- `changepoints`: list[dict] — Each with `start`, `end`, `confidence`, `type` (trend/seasonal).
- `pastas_fit_quality`: dict — EVP (Explained Variance Percentage), RMSE, AIC.

**Dependencies**: `pastas`, `Rbeast` (optional, fallback to `ruptures`), `statsmodels` (ACF/Ljung-Box).

**Note**: Reuses existing `PastasWrapper` from `dashboard/utils/counterfactual/pastas_validation.py`. The new `pastas_layer.py` imports and extends it, adding residual extraction and ACF/PACF analysis.

### Layer 2 — ML + XAI (TFT on clean period + attribution drift)

**Input**: Same time series + clean period mask from Layer 1.

**Process**:
1. **Identify clean periods** (see Clean Period Algorithm below).
2. **Train TFT model** on clean periods only (target = piézo level, past covariates = precip/temp/ETP). Standard temporal split within clean windows. Uses `ModelFactory` directly with a lightweight `train_transient_model()` function (no MLflow logging, no model registry persistence — this is a disposable analysis model, not a registered artifact).
3. **Predict on full series** — the model predicts everywhere, including suspect periods.
4. **Compute XAI attributions** per sliding window:
   - Integrated Gradients (Captum) — per-timestep, per-feature.
   - SHAP/TimeSHAP — feature-level importance.
   - TFT attention weights — variable selection + temporal attention.
5. **Compute attribution drift metrics** between clean windows and each test window:
   - Jensen-Shannon divergence of per-feature SHAP distributions.
   - Spearman rank correlation of feature importance rankings.
   - Feature Agreement (FA) — % of features in same top-K.
   - Temporal attribution anomaly — IG magnitude on timesteps where model struggles.
6. **Residual structure analysis** on ML model: ACF of prediction errors, correlation with Pastas residuals.

**Output**:
- `predictions`: TimeSeries — Full-series predictions from clean-trained model.
- `ml_residuals`: pd.Series — Prediction errors.
- `xai_attributions`: dict — Per-window attribution matrices (timesteps x features).
- `drift_metrics`: list[dict] — Per-window JS divergence, Spearman, FA scores.
- `training_metrics`: dict — MAE, RMSE, NSE on clean validation set.

**Dependencies**: Existing `model_factory.py`, `training.py`, `explainability/` modules. No new ML dependencies.

### Layer 3 — Embeddings (SoftCLT/TS2Vec drift + twin stations)

**Input**: Raw piezometric time series + pre-trained SoftCLT/TS2Vec encoder.

**Process**:
1. **Compute temporal embeddings**: Encode the station's time series in sliding windows → sequence of embedding vectors.
2. **Embedding trajectory analysis**: Track the station's position in latent space over time. Compute distance from historical distribution (Mahalanobis or z-score per dimension).
3. **Find twin stations**: From a pool of ADES stations with pre-computed embeddings, find k-nearest neighbors in embedding space (cosine similarity).
4. **Twin divergence**: Compare the target station's recent behavior with its twins. If twins are stable but target diverges → local anomaly (likely pumping, not climate).
5. **UMAP projection**: 2D projection of embedding trajectory for visualization, colored by time.

**Output**:
- `embedding_trajectory`: np.ndarray — (n_windows, embedding_dim).
- `drift_scores`: pd.Series — Per-window distance from historical distribution.
- `twin_stations`: list[dict] — Each with `station_id`, `similarity`, `divergence_score`.
- `umap_projection`: np.ndarray — (n_windows, 2) for visualization.

**Dependencies**: Existing SoftCLT/TS2Vec code, `umap-learn`.

### Fusion Layer

**Input**: Outputs from all 3 layers.

**Process**:
1. **Temporal alignment**: All 3 layers produce per-window scores. Align on the same temporal grid.
2. **Per-window fusion**: For each window, combine:
   - Layer 1: Is there a BEAST changepoint? Is ACF significant?
   - Layer 2: Is JS divergence > threshold? Is Spearman < threshold?
   - Layer 3: Is embedding drift > threshold? Do twins diverge?
3. **Concordance scoring**:
   - 3/3 layers agree → HIGH confidence (red).
   - 2/3 layers agree → MEDIUM confidence (orange).
   - 1/3 layers → LOW confidence (yellow).
   - 0/3 → clean (green).
4. **Window merging**: Merge adjacent suspect windows into contiguous periods.

**Output**:
- `suspect_windows`: list[dict] — Each with `start`, `end`, `confidence` (high/medium/low), `layer_details` (which layers flagged it and why).
- `global_score`: float — Overall suspicion score for the station (0-1).

### Clean Period Identification Algorithm

The clean period selector identifies temporal windows where Pastas explains the data well (i.e., no hidden stress is needed).

**Algorithm**:
1. Compute Pastas residuals `r(t) = observed(t) - modeled(t)`.
2. Set amplitude threshold: `T = clean_residual_threshold` (default: `2 * std(r)`).
3. Compute rolling Ljung-Box test on residuals with:
   - Rolling window: 180 days.
   - Max lag: 30 days.
   - Significance level: alpha = 0.05.
4. A day `t` is "clean" if:
   - `|r(t)| < T`, AND
   - The Ljung-Box p-value for the 180-day window centered on `t` is > 0.05 (no significant autocorrelation).
5. Merge contiguous clean days into windows. Discard windows < 90 days.
6. If total clean data < 365 days: relax threshold iteratively (2σ → 3σ → 4σ).
7. If still < 365 days after 4σ: fall back to **seasonal heuristic** (November 1 — March 31 = presumed clean for agricultural pumping).
8. If seasonal heuristic also yields < 180 days: skip Layer 2 with warning.

**Output**: Binary mask `is_clean(t)` over the full time range.

### Fusion Temporal Grid

All layers produce scores at different temporal resolutions. The fusion layer aligns them on a **monthly grid** (30-day steps):

- **Layer 1** (Pastas): Daily residuals → aggregated to monthly (mean |residual|, Ljung-Box per month). BEAST changepoints mapped to months.
- **Layer 2** (XAI): 90-day windows with 30-day stride → each window mapped to its center month.
- **Layer 3** (Embeddings): 365-day windows → drift score assigned to the last month of each window.
- **Fusion**: Per-month, each layer provides a binary flag (suspect/clean) + a continuous score. Concordance computed per month.

Adjacent suspect months are merged into contiguous periods with a configurable merge gap (default: 30 days).

## 3. Data Flow

```
User selects dataset (existing)
       │
       ▼
POST /api/v1/pumping-detection/analyze
  body: { dataset_id, config }
       │
       ▼
Backend creates async task (with cancellation support via stop_event)
  → SSE stream: /api/v1/pumping-detection/{task_id}/stream
       │
  SSE event types (compatible with existing useSSE.ts pattern):
    event: progress    data: { stage, pct, message }
    event: metrics     data: { stage, partial_result }
    event: error       data: { stage, error_message, recoverable }
    event: done        data: { full_results }

  Stage progression:
    progress: { stage: "pastas",     pct: 0.15, message: "Calibrating Pastas..." }
    metrics:  { stage: "pastas",     partial_result: { evp, rmse, n_residuals } }
    progress: { stage: "changepoint", pct: 0.25, message: "Running BEAST..." }
    metrics:  { stage: "changepoint", partial_result: { n_changepoints, [...] } }
    progress: { stage: "clean",      pct: 0.30, message: "Selecting clean periods..." }
    metrics:  { stage: "clean",      partial_result: { n_clean_days, pct_clean } }
    progress: { stage: "ml_train",   pct: 0.50, message: "Training TFT on clean data..." }
    progress: { stage: "xai",        pct: 0.70, message: "Computing attributions..." }
    progress: { stage: "embedding",  pct: 0.85, message: "Analyzing embeddings..." }
    progress: { stage: "fusion",     pct: 0.95, message: "Computing fusion scores..." }
    done:     { full_results }

  The frontend hook `usePumpingDetection.ts` extends `useSSE.ts` with a typed
  discriminated union for stages, enabling partial result rendering as each
  layer completes (e.g., show Pastas panel as soon as Layer 1 finishes
  while Layer 2 is still training).
```

## 4. API Endpoints

### `POST /api/v1/pumping-detection/analyze`

Start a pumping detection analysis.

**Request**:
```json
{
  "dataset_id": "string",
  "config": {
    "pastas": {
      "response_function": "Gamma",  // or "Hantush"
      "noise_model": true
    },
    "changepoint": {
      "method": "beast",  // or "pelt" or "both"
      "min_segment_length": 90  // days
    },
    "ml": {
      "model_type": "TFTModel",  // from ModelFactory
      "input_chunk_length": 365,
      "output_chunk_length": 30,
      "max_epochs": 100,
      "clean_residual_threshold": "auto"  // or float
    },
    "xai": {
      "methods": ["integrated_gradients", "shap", "attention"],
      "window_size": 90,  // days
      "stride": 30  // days
    },
    "embeddings": {
      "encoder": "softclt",  // or "ts2vec"
      "window_size": 365,
      "n_twins": 5
    },
    "fusion": {
      "js_divergence_threshold": 0.3,
      "spearman_threshold": 0.5,
      "embedding_drift_threshold": 2.0,
      "acf_significance": 0.05,
      "min_layers_for_high": "all",  // "all" adapts to available layers
      "merge_gap_days": 30
    }
  }
}
```

**Response**: `{ "task_id": "string" }`

### `GET /api/v1/pumping-detection/{task_id}/stream`

SSE stream of analysis progress and results.

### `GET /api/v1/pumping-detection/{task_id}/results`

Full results after completion.

### `POST /api/v1/pumping-detection/{task_id}/cancel`

Cancel a running analysis. Each layer checks the task's `stop_event` at reasonable intervals (Pastas: after solver, TFT: every epoch, embeddings: every window). Returns partial results from completed layers.

### `GET /api/v1/pumping-detection/{task_id}/layer/{layer_name}`

Fetch partial results from a specific layer (pastas, ml, embeddings) before the full pipeline completes. Enables progressive rendering in the frontend.

### `GET /api/v1/pumping-detection/bnpe-context?lat={lat}&lon={lon}&radius_km={r}`

Fetch nearby BNPE declared pumping facilities for context overlay. Results cached in-memory with 24h TTL (keyed by geohash at ~1km precision). Timeout: 10s. If Hub'Eau is unavailable, returns empty list with `"bnpe_available": false` flag.

## 5. Frontend

### Page: `PumpingDetectionPage.tsx`

**Layout**:

```
┌─────────────────────────────────────────────────────┐
│  [Dataset selector]  [Config panel]  [Analyze btn]  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  CHRONIQUE ANNOTÉE (Plotly, full width)             │
│  - Piézo original (blue line)                       │
│  - Pastas reconstruction (dashed gray)              │
│  - Suspect windows (colored rectangles)             │
│  - BNPE declared pumping overlay (if available)     │
│                                                     │
├──────────┬──────────┬───────────────────────────────┤
│ PASTAS   │ XAI      │ EMBEDDINGS                    │
│          │          │                               │
│ Residuals│ Heatmap  │ UMAP trajectory               │
│ ACF/PACF │ features │ colored by time               │
│ BEAST    │ ×temps   │                               │
│ change-  │          │ Twin stations                 │
│ points   │ JS div   │ divergence chart              │
│          │ curve    │                               │
├──────────┴──────────┴───────────────────────────────┤
│  VERDICT                                            │
│  Score: 0.82 [HIGH]                                 │
│  Suspect periods: Jun-Sep 2019, Jul-Aug 2020        │
│  Concordance: 3/3 layers for summer 2019            │
│  BNPE context: 2 declared pumps within 3km          │
└─────────────────────────────────────────────────────┘
```

### Hook: `usePumpingDetection.ts`

```typescript
// Follows same pattern as useCounterfactual.ts
interface PumpingDetectionConfig { ... }
interface PumpingDetectionResult {
  suspect_windows: SuspectWindow[];
  global_score: number;
  pastas: PastasResult;
  xai: XAIResult;
  embeddings: EmbeddingResult;
}

function usePumpingDetection() {
  const analyzeMutation = useMutation(...)  // POST /analyze
  const { data, stages } = useSSE(taskId)   // SSE stream
  return { analyze, results, progress, stages }
}
```

### Components:
- `AnnotatedChroniquePlot.tsx` — Main Plotly chart with suspect windows
- `PastasPanel.tsx` — Residuals + ACF + changepoints
- `XAIDriftPanel.tsx` — Attribution heatmap + divergence curve
- `EmbeddingPanel.tsx` — UMAP trajectory + twin stations
- `VerdictPanel.tsx` — Fusion score + summary

## 6. Backend Module Structure

```
dashboard/utils/pumping_detection/
  __init__.py
  pastas_layer.py        # PastasAnalyzer class
  changepoint.py         # ChangepointDetector (BEAST + PELT)
  clean_period.py        # CleanPeriodSelector (uses Pastas residuals)
  ml_layer.py            # MLAnalyzer (train TFT on clean, predict full)
  xai_layer.py           # XAIDriftAnalyzer (IG, SHAP, attention + metrics)
  embedding_layer.py     # EmbeddingAnalyzer (SoftCLT drift + twins)
  fusion.py              # FusionEngine (concordance scoring)
  bnpe_client.py         # Hub'Eau Prélèvements API client

api/routers/pumping_detection.py   # FastAPI router
api/schemas/pumping_detection.py   # Pydantic models
```

## 7. Configuration Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Pastas response function | Gamma | Standard for recharge; Hantush if well distances known |
| Changepoint method | BEAST | Handles seasonality natively, provides uncertainty |
| Min segment length | 90 days | Avoid detecting short noise bursts as pumping |
| Clean residual threshold | auto (2σ) | Pastas residuals within 2 standard deviations |
| TFT input chunk | 365 days | Full year of context for seasonal patterns |
| TFT output chunk | 30 days | Monthly prediction horizon |
| XAI window size | 90 days | Quarterly analysis windows |
| XAI stride | 30 days | Monthly sliding step |
| Embedding window | 365 days | Full year context for contrastive embeddings |
| Number of twins | 5 | Enough for robust comparison |
| Fusion high threshold | 3/3 layers | Conservative — all layers must agree |

## 8. New Dependencies

| Package | Purpose | Size |
|---------|---------|------|
| `ruptures` | PELT change point detection | Lightweight, pure Python |
| `Rbeast` | BEAST Bayesian change points | PyPI, ~5MB |

All other dependencies already in the project: `pastas`, `captum`, `timeshap`, `umap-learn`, `statsmodels`.

## 9. What We Reuse

| Existing module | Reused for |
|----------------|------------|
| `dashboard/utils/model_factory.py` | Create TFT model for Layer 2 |
| `dashboard/utils/training.py` | `run_training_pipeline()` on clean periods |
| `dashboard/utils/explainability/` | IG, SHAP, attention computation |
| `dashboard/utils/counterfactual/pastas_validation.py` | `PastasWrapper` for Layer 1 |
| `dashboard/utils/dataset_registry.py` | Load existing datasets |
| `frontend/src/hooks/useSSE.ts` | SSE streaming pattern |
| `frontend/src/lib/api.ts` | API client |
| SoftCLT/TS2Vec (in progress) | Embedding encoder for Layer 3 |

## 10. Out of Scope (Phase 2)

- Batch analysis of multiple stations (screening mode)
- Automatic BNPE × ADES cross-referencing via BDLISA
- Pumping characterization (Hantush inverse → estimated flow rate)
- Export rapport PDF
- Embeddings as TFT static covariates
- ROC/AUC with synthetic ground truth validation
- Multi-model comparison (LSTM, NHiTS alongside TFT)

## 11. Success Criteria

1. Pipeline runs end-to-end on a single ADES station in < 15 minutes.
2. All 3 layers produce interpretable, visualizable output.
3. Fusion score distinguishes known "clean" stations from stations near BNPE-declared pumping.
4. XAI attribution drift is measurably different between clean and suspect windows.
5. Frontend displays results with the same quality as existing Forecasting/Counterfactual pages.

## 12. Dependency Validation

Before starting implementation, validate in the Docker environment:
```bash
pip install ruptures Rbeast
python -c "import ruptures; print(ruptures.__version__)"
python -c "import Rbeast; print('BEAST OK')"
```
If Rbeast fails (Fortran build issues on Ubuntu 24.04), Phase 1 proceeds with PELT only via `ruptures`.

## 13. Testing Strategy

### Unit Tests (`tests/pumping_detection/`)
- `test_pastas_layer.py` — Pastas calibration on synthetic data, residual extraction, ACF computation.
- `test_changepoint.py` — BEAST/PELT on synthetic series with known changepoints.
- `test_clean_period.py` — Clean period selection with known clean/suspect windows.
- `test_ml_layer.py` — Transient TFT training on small dataset, prediction output shape.
- `test_xai_layer.py` — Attribution computation, JS divergence, Spearman on known distributions.
- `test_embedding_layer.py` — Drift score computation (skip if encoder unavailable).
- `test_fusion.py` — Concordance scoring with mocked layer outputs (2-layer and 3-layer modes).
- `test_bnpe_client.py` — Hub'Eau API call with mocked responses + timeout handling.

### Integration Test
- `test_pipeline_e2e.py` — Full pipeline on a known ADES station. Validate that all layers produce non-null output, fusion score is in [0,1], and frontend-compatible JSON is emitted.

### Validation Pairs
Identify at least 2 station pairs for qualitative validation:
- **Known clean station**: Far from any BNPE-declared pumping, stable Pastas residuals.
- **Known influenced station**: Near BNPE-declared pumping, with visible drawdown patterns.
- The pipeline should produce a higher suspicion score for the influenced station.

## 14. References

Full literature review: `docs/SOTA_pumping_detection.md` (~50 papers across 10 categories).

Key references:
- Collenteur et al. (2019) — Pastas TFN modeling
- Zhao et al. (2019) — BEAST change point detection
- Hsieh et al. (2023, 2024) — HHT+EOF pumping signal extraction
- Sundararajan et al. (2017) — Integrated Gradients
- Duckworth et al. (2021) — Concept drift via SHAP
- Vater et al. (2025) — SHAP interactions for anomaly detection
- Clark et al. (2025) — XAI for groundwater interpretation
- Lin et al. (2024) — Pumping detection without pumping data
