# Pastas Lab — Design

**Date**: 2026-04-17
**Status**: Draft
**Scope**:
- New top-level section `/pastas` in the frontend with 3 tabs (Fit, Scenarios, Batch).
- New backend module `dashboard/utils/pastas/` (pure Python, no framework dependency).
- New API router `api/routers/pastas.py` with Pydantic schemas in `api/schemas/pastas.py`.
- New MLflow experiment `pastas` for model tracking & artifact storage.
- Lab section (`/lab/*`) removed from navigation and routes redirect to `/`.

## Problem Statement

Hydrogeologists need a first-class interface to build, calibrate and exploit Pastas Transfer Function Noise (TFN) models inside Junon. Today Pastas is buried as an internal subcomponent of two Lab features (Counterfactual dual validation, Pumping Detection layer 1) and has no user-facing exposure: no training, no persistence, no what-if scenario tooling.

The Lab section contains three pages (Latent Space, Counterfactual, Pumping Detection) that are not production-ready and create clutter in the main navigation. They will be hidden from users but the code is kept in the repo for later reuse.

Goal: ship a progressively-built **Pastas Lab** that covers the full Pastas workflow — fit, analyze, simulate what-if scenarios (pumping, climate trend, stress scaling), persist to MLflow, and eventually batch process regions of stations — while hiding the broken Lab section.

## Non-Goals

- Re-using or refactoring `dashboard/utils/counterfactual/pastas_validation.py` or `dashboard/utils/pumping_detection/pastas_layer.py`. Those stay in place for the dormant Lab code and the new module is independent.
- Re-enabling any Lab feature. Specs for Lab work are preserved as historical documents.
- Replacing Darts-based forecasting. Pastas is complementary (physics-informed) and lives in parallel.
- Deep authentication / role-based access for the lab. Uses the existing Junon auth posture.

## Phased Delivery

| Phase | Ships | Value |
|-------|-------|-------|
| **P1** — Fit & scenarios (MVP) | Single-station fit (LeastSquares/Lmfit), MLflow persistence, 4 scenario modifications (pumping synthetic, pumping CSV, linear trend, scale stress), diagnostics panel. Lab hidden. | Immediately useful for hydrogeologists. Autonomous release. |
| **P2** — Uncertainty & signatures | MCMC solver (EmceeSolve, SSE stream, CI bands), hydrological signatures, multi-model compare, step modification, more recharge/response/noise options. | Adds scientific rigor and confidence intervals. |
| **P3** — Batch & cartography | Pastastore-backed batch fit across N stations, GeoJSON output, station map with parameters & EVP. | Regional-scale analysis. |

Each phase has its own implementation plan. This spec describes the end-state architecture; Phase boundaries are called out inline.

## Architecture Overview

```
Frontend                          Backend (FastAPI)              Core services
────────                          ────────────────               ─────────────
/pastas/fit         ─── POST /api/v1/pastas/fit ──→  fit_service.run_fit()
                                                          │
                                                          ├── builder.build_model()
                                                          ├── solver.solve()
                                                          └── MLflow: log params+metrics+model.pas artifact

/pastas/scenarios   ─── POST /api/v1/pastas/simulate ─→ scenario.simulate_scenario()
                                                          │
                                                          ├── io.load_model(run_id)   (cached)
                                                          ├── apply_modification()    (per mod)
                                                          └── model.simulate()

/pastas/batch (P3)  ─── POST /api/v1/pastas/batch/fit ─→ batch.run_batch()
                                                          │
                                                          └── Pastastore (DictConnector) bulk_solve

                     MLflow experiment `pastas`
                     ├── runs (one per fit)
                     └── model registry (filter by station_id tag)
```

**Layering invariants**
- `dashboard/utils/pastas/` is pure Python — no FastAPI, no React, no Streamlit imports.
- `api/routers/pastas.py` is a thin wrapper: validates input, calls services, shapes response.
- `frontend/src/pages/pastas/*` consumes only `/api/v1/pastas/*` — no direct Python bridge.

## Backend Module: `dashboard/utils/pastas/`

### `config.py`
Maps string identifiers to Pastas classes so the UI can drive configuration via enums and the service layer can instantiate the correct types. The registries also serve a `GET /pastas/options` endpoint consumed by the frontend to populate dropdowns dynamically.

```python
RECHARGE_REGISTRY = {
    "Linear": ps.rch.Linear,
    "FlexModel": ps.rch.FlexModel,
    "Berendrecht": ps.rch.Berendrecht,      # P2
    "Peterson": ps.rch.Peterson,            # P2
}
RFUNC_REGISTRY = {
    "Gamma": ps.Gamma,
    "Exponential": ps.Exponential,
    "Hantush": ps.Hantush,
    "HantushWellModel": ps.HantushWellModel,  # P2
    "DoubleExponential": ps.DoubleExponential,  # P2
    "FourParam": ps.FourParam,             # P2
    "One": ps.One,
}
NOISE_REGISTRY = {
    "ArNoiseModel": ps.ArNoiseModel,
    "ArmaNoiseModel": ps.ArmaNoiseModel,   # P2
}
SOLVER_REGISTRY = {
    "LeastSquares": ps.LeastSquares,
    "Lmfit": ps.LmfitSolve,
    "Emcee": ps.EmceeSolve,                # P2 only
}
```

### `builder.py`
`build_model(request: FitRequest, series: StationSeries) -> ps.Model`

1. Loads the dataset via `dataset_registry.get(request.dataset_id)`. The `PreparedDataset` has a `target_column` (piezometric head) and `covariate_columns`. The request specifies `precip_column` and `evap_column` (which must be in `covariate_columns`) so the builder knows which covariates to wire to the RechargeModel.
2. Validates series: minimum 365 observations, temporal overlap between observation and stresses, NaN ratio < 20%. Raises `ValueError` with an explicit message on failure.
3. Instantiates `ps.Model(gwl_series, name=...)`.
4. Builds `ps.RechargeModel(precip, evap, rfunc=<RF>, recharge=<RechargeModel>)` and adds it.
5. Optionally adds a noise model from `NoiseConfig`.
6. Returns the unsolved model together with the applied `tmin/tmax` window.

### `fit_service.py`
Thin orchestrator that runs the solver and persists to MLflow.

```python
def run_fit(request: FitRequest) -> FitResponse:
    series = load_station_series(request.dataset_id)
    model = build_model(request, series)

    solver_cls = SOLVER_REGISTRY[request.solver.type]
    solver = solver_cls(**request.solver.kwargs)

    mlflow.set_experiment("pastas")
    with mlflow.start_run(run_name=request.name) as run:
        model.solve(
            solver=solver,
            tmin=request.tmin,
            tmax=request.tmax,
            report=False,
        )

        metrics = extract_metrics(model)
        mlflow.log_params(flatten_params(request))
        mlflow.log_metrics(metrics)
        mlflow.set_tag("station_id", request.dataset_id)
        mlflow.set_tag("pastas_version", ps.__version__)
        mlflow.set_tag("series_hash", series.hash())   # sha256 over concatenated (index,values) of the three series

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.pas"
            model.to_file(str(path))
            mlflow.log_artifact(str(path))

        return build_fit_response(model, run.info.run_id, metrics)
```

`run_fit_mcmc(request, on_progress)` (P2) is a variant that instantiates `EmceeSolve`, accepts a progress callback invoked per chain step (used by SSE stream), and logs the posterior samples as an additional artifact.

### `io.py`
`load_model(run_id: str) -> ps.Model` downloads `model.pas` from MLflow artifacts into `/tmp/pastas_models/<run_id>.pas`, memoizes the parsed `ps.Model` in an LRU cache (maxsize 32) keyed by `run_id`, and guards against version skew: if `ps.__version__` differs from the tag `pastas_version`, raises `ModelVersionMismatch` translated to HTTP 409 by the router.

Cache invalidation: on `DELETE /models/{run_id}`, evict the cache entry.

### `scenario.py`
`simulate_scenario(req: ScenarioRequest) -> ScenarioResponse`

1. Loads base model via `load_model(req.run_id)`.
2. Simulates baseline on `[tmin, tmax]`.
3. Deep-copies the model, applies modifications in order using `apply_modification`, simulates scenario.
4. Returns baseline, scenario, delta, per-stress contributions (both baseline and scenario), and (P2) uncertainty bands if the base run was MCMC.

`apply_modification(model, mod)` dispatches on `mod.type`:

| `mod.type` | Action |
|---|---|
| `pumping_synthetic` | Generate Q(t) pd.Series per pattern (constant/seasonal/pulse), build `ps.WellModel` with `rfunc` and distance/depth, `model.add_stressmodel(...)`. |
| `pumping_upload` | Same as above, Q(t) parsed from the uploaded CSV rows. |
| `linear_trend` | `ps.LinearTrend(start, end, rfunc=ps.One())`, add. |
| `step` (P2) | `ps.StepModel(date, rfunc=ps.One())`, add. |
| `scale_stress` | Rebuild the `RechargeModel` with stress multiplied by `factor` on `[start, end]`. Pastas has no clean in-place stress update, so this replaces the stress model entirely, preserving the calibrated parameters via `parameters=model.parameters`. |

Invariants checked per modification:
- Dates within the simulation window (warn, do not fail, if partial).
- `rate_m3d >= 0` and `factor > 0`.
- `distance_m > 0` for pumping modifications (distance to the well is user-provided, no lat/lon needed — `WellModel` takes `distances=[d]` directly).

### `signatures.py` (P2)
Wraps `ps.stats.signatures.summary(head)` to compute the 30+ hydrological signatures on observed and simulated series. Exposed through `GET /pastas/models/{run_id}/signatures`.

### `batch.py` (P3)
Instantiates a `PastaStore(connector=DictConnector())`, imports series for the selected stations, runs `create_models` + `solve_models` in parallel (bounded by CPU count), aggregates parameters + metrics per station, and produces a GeoJSON FeatureCollection with station coordinates and model attributes. Streams progress via SSE.

Pre-requisite: station lat/lon available in the `gold` schema of `brgm-postgres`. To be validated at the start of P3.

## API Contract

All endpoints are prefixed by `/api/v1/pastas/`. Phase indicated.

| Method | Path | Phase | Purpose |
|---|---|---|---|
| `GET`  | `/options` | P1 | Returns the content of `config.py` registries for UI dropdowns |
| `POST` | `/fit` | P1 | Synchronous fit (`LeastSquares`, `Lmfit`), returns `FitResponse` |
| `POST` | `/fit/mcmc` | P2 | SSE stream (progress per chain step) + final `FitResponse` |
| `GET`  | `/models` | P1 | Lists runs in MLflow experiment `pastas`, filterable by `station_id` |
| `GET`  | `/models/{run_id}` | P1 | Returns the full `FitResponse` for a stored run |
| `DELETE` | `/models/{run_id}` | P1 | Deletes the run and evicts the cache |
| `POST` | `/simulate` | P1 | Runs a scenario; returns `ScenarioResponse` |
| `POST` | `/compare` | P2 | Compares N runs; returns metrics side-by-side + overlay series |
| `GET`  | `/models/{run_id}/signatures` | P2 | Hydrological signatures on observed vs simulated |
| `POST` | `/batch/fit` | P3 | Launches batch fit; SSE progress; returns job_id |
| `GET`  | `/batch/{job_id}/status` | P3 | Job status + per-station results |
| `GET`  | `/batch/{job_id}/map` | P3 | GeoJSON of parameters + EVP |

### Pydantic Schemas (`api/schemas/pastas.py`)

```python
# ---------- Fit ----------
class RechargeConfig(BaseModel):
    type: Literal["Linear", "FlexModel", "Berendrecht", "Peterson"]
    kwargs: dict[str, Any] = {}

class ResponseConfig(BaseModel):
    type: Literal["Gamma", "Exponential", "Hantush", "HantushWellModel",
                  "DoubleExponential", "FourParam", "One"]
    kwargs: dict[str, Any] = {}

class NoiseConfig(BaseModel):
    type: Literal["ArNoiseModel", "ArmaNoiseModel", "none"] = "ArNoiseModel"

class SolverConfig(BaseModel):
    type: Literal["LeastSquares", "Lmfit", "Emcee"] = "LeastSquares"
    kwargs: dict[str, Any] = {}

class FitRequest(BaseModel):
    dataset_id: str
    station_id: str | None = None         # if multi-station dataset, filter to this station
    precip_column: str                     # covariate column to use as precipitation
    evap_column: str                       # covariate column to use as evapotranspiration
    tmin: date | None = None
    tmax: date | None = None
    recharge: RechargeConfig
    response: ResponseConfig
    noise: NoiseConfig = NoiseConfig()
    solver: SolverConfig = SolverConfig()
    name: str | None = None

class TimeSeries(BaseModel):
    index: list[date]
    values: list[float]

class FitParameter(BaseModel):
    name: str
    optimal: float
    stderr: float | None
    initial: float
    pmin: float | None
    pmax: float | None
    vary: bool

class FitResponse(BaseModel):
    run_id: str
    metrics: dict[str, float]            # evp, rmse, aic, bic, ljung_box_pvalue, n_obs
    parameters: list[FitParameter]
    observed: TimeSeries
    simulated: TimeSeries
    residuals: TimeSeries
    contributions: dict[str, TimeSeries]
    step_response: TimeSeries
    block_response: TimeSeries
    acf: dict[str, Any]                   # values, pacf_values, nlags, ljung_box_pvalue
    warnings: list[str] = []
    pastas_version: str

# ---------- Scenario ----------
class PumpingSynthetic(BaseModel):
    type: Literal["pumping_synthetic"] = "pumping_synthetic"
    pattern: Literal["constant", "seasonal", "pulse"]
    rate_m3d: float
    start: date
    end: date
    distance_m: float
    screen_depth_m: float | None = None
    rfunc: Literal["Hantush", "Exponential"] = "Hantush"
    period_days: int = 365               # for seasonal

class PumpingRow(BaseModel):
    date: date
    Q_m3d: float

class PumpingUpload(BaseModel):
    type: Literal["pumping_upload"] = "pumping_upload"
    csv_rows: list[PumpingRow]
    distance_m: float
    rfunc: Literal["Hantush", "Exponential"] = "Hantush"

class LinearTrendMod(BaseModel):
    type: Literal["linear_trend"] = "linear_trend"
    start: date
    end: date
    slope_m_per_year: float

class StepMod(BaseModel):                 # P2
    type: Literal["step"] = "step"
    date: date
    magnitude_m: float

class ScaleStressMod(BaseModel):
    type: Literal["scale_stress"] = "scale_stress"
    stress: Literal["precip", "evap"]
    factor: float
    start: date
    end: date

Modification = Annotated[
    PumpingSynthetic | PumpingUpload | LinearTrendMod | StepMod | ScaleStressMod,
    Field(discriminator="type"),
]

class ScenarioRequest(BaseModel):
    run_id: str
    tmin: date
    tmax: date
    modifications: list[Modification]

class ScenarioResponse(BaseModel):
    baseline: TimeSeries
    scenario: TimeSeries
    delta: TimeSeries
    contributions_baseline: dict[str, TimeSeries]
    contributions_scenario: dict[str, TimeSeries]
    uncertainty: dict[str, TimeSeries] | None = None   # {lower, upper} when P2 MCMC
    warnings: list[str] = []
```

## Frontend

### Routes & navigation
- `routes.tsx`:
  - Replace the `/lab` block with `{ path: '/lab/*', element: <Navigate to="/" replace /> }` (keeps bookmarks from 404ing).
  - Add `/pastas` with children `fit`, `scenarios`, `batch`; index redirects to `/pastas/fit`.
- `TopNav.tsx`:
  - Remove `{ to: '/lab', icon: FlaskConical, label: 'Lab' }`.
  - Add `{ to: '/pastas', icon: Waves, label: 'Pastas' }` (icon from lucide-react).

### Pages (`frontend/src/pages/pastas/`)
- `PastasLayout.tsx` — tab bar (Fit / Scenarios / Batch).
- `FitPage.tsx` — station picker + `PastasConfigForm` + `FitResultsPanel`.
- `ScenariosPage.tsx` — model picker (lists MLflow runs) + `ScenarioComposer` + `ScenarioResultsPanel`.
- `BatchPage.tsx` — P3.

### Components (`frontend/src/components/pastas/`)
- `StationPicker` — reuses `useDatasets` hook.
- `PastasConfigForm` — collapsible sections for Recharge, Response, Noise, Solver, Calibration window.
- `FitResultsPanel` — four sub-panels: Metrics (cards), Parameters (table), Fit plot (Plotly obs/sim + residuals subplot), Responses (step + block + stacked decomposition).
- `ScenarioComposer` — drag-reorderable list (`@dnd-kit/sortable`) of `ModificationCard`s with an "+ Add modification" menu.
- `ModificationCard` — header (type badge, delete button) + type-specific editor:
  - `PumpingSyntheticEditor`
  - `PumpingUploadEditor` (CSV drop zone)
  - `LinearTrendEditor`
  - `StepEditor` (P2)
  - `ScaleStressEditor`
- `ScenarioResultsPanel` — overlay plot baseline vs scenario with CI band (P2), delta plot, stacked contributions.

### Hooks (`frontend/src/hooks/usePastas.ts`)
```ts
usePastasOptions()                  // GET /pastas/options (cached 1h, drives UI dropdowns)
usePastasFit()                      // useMutation POST /fit
usePastasMcmcFit()                  // useSSE wrapper, P2
usePastasModels(stationId?)         // GET /models
usePastasModel(runId)               // GET /models/{run_id}
usePastasSimulate()                 // useMutation POST /simulate
usePastasSignatures(runId)          // GET /models/{run_id}/signatures, P2
usePastasCompare(runIds)            // POST /compare, P2
```

Cache keys align with React Query conventions used elsewhere in the project (`['pastas', 'models', stationId]`, `['pastas', 'model', runId]`, etc.).

### New frontend dependencies
- `@dnd-kit/core` + `@dnd-kit/sortable` — drag-reorder for ScenarioComposer.

### Validation
Frontend uses Zod to validate modification payloads before sending. Server-side validation (Pydantic + builder checks) remains the source of truth.

## Error Handling

Validation-at-the-boundary strategy: Pydantic + `builder` + `apply_modification` catch bad inputs early and raise explicit errors; core services do not wrap Pastas calls in defensive try/except.

| Source | Symptom | Handling |
|---|---|---|
| Missing or short series, too many NaNs, no obs/stress overlap | `builder.build_model` raises `ValueError` | Router returns 422 with the error message. |
| Solver did not converge or parameters hit their bounds | `model.fit.success` false or `abs(optimal - bound) < eps` | Appended to `FitResponse.warnings`. Frontend shows a yellow banner. Not blocking. |
| `run_id` not found in MLflow | `MlflowException` | Router returns 404. React Query invalidates the list cache. |
| Pumping modification with `distance_m <= 0` | `apply_modification` raises `ValueError` | 422 with explicit "distance must be positive" message. |
| MCMC run is very long / SSE drops | SSE reconnection logic | The run is persisted to MLflow before the stream ends, so reload recovers it. Client polls `/models/{run_id}` on SSE drop. |
| Pastas version mismatch on `load_model` | `ModelVersionMismatch` | 409 Conflict with current vs stored version in the response body. |
| Dataset series changed in DB since fit | `series_hash` tag differs on load | Warning tag on the run listed in `/models`. Not blocking (fit remains reproducible from artifact). |
| Unexpected Pastas crash | Unhandled exception | 500 with generic message; full stack logged server-side. |

## Testing

### Unit tests (`tests/pastas/`)
- `test_config.py` — registries expose all declared classes and map to the correct Pastas symbols.
- `test_builder.py` — each combination (recharge × rfunc × noise) produces a valid `ps.Model` on a short synthetic fixture. Validators reject short series, NaN-heavy series, non-overlapping stresses.
- `test_fit_service.py` — MLflow mocked; synthetic series generated from known Gamma + Linear recharge parameters; solver recovers parameters within tolerance (evp > 95%, parameters within 10%).
- `test_scenario.py` — per modification type:
  - `pumping_synthetic` with `rate=0` yields scenario ≈ baseline.
  - Positive pumping → GWL decrease sign invariant.
  - `linear_trend` → monotone drift sign matches `slope`.
  - `scale_stress(precip, factor=1)` is a no-op.
- `test_io.py` — round-trip save → load → simulate returns identical series. Version mismatch raises `ModelVersionMismatch`.
- `test_apply_modification.py` — invariants and input validation (dates, units, coordinates).

### Integration tests (`tests/test_api_pastas.py`)
- Spin up the FastAPI app with a local MLflow tracking URI and a fixture station.
- `POST /fit` → 200 + `run_id`.
- `GET /models/{run_id}` returns the same metrics.
- `POST /simulate` with 2 modifications → response shape and sign invariants.
- `DELETE /models/{run_id}` removes the run and returns 404 on subsequent GET.

### E2E tests (`e2e/`)
- `pastas-fit.spec.ts` — open `/pastas/fit`, pick a station, submit fit, assert results panels render with non-zero metrics.
- `pastas-scenarios.spec.ts` — load the fitted run, add a `pumping_synthetic` modification, simulate, assert baseline/scenario/delta plots render.
- `lab-disabled.spec.ts` — `/lab/latent-space`, `/lab/counterfactual`, `/lab/pumping-detection` redirect to `/`.

### Coverage target
- `>85%` on `dashboard/utils/pastas/`.
- `100%` on validators (`builder.py` validation functions, `apply_modification` invariants).

## Open Questions / Validated at Phase Entry

- **Station coordinates availability** (P3 pre-req): `lat/lon` needed only for batch cartography (not for P1/P2 — WellModel uses user-provided distance). Verify existence in `gold` schema at start of P3; adjust scope if missing.
- **Icon for the Pastas nav entry**: placeholder `Waves`. User can choose differently at implementation time.
- **Units conversions UI-side**: pumping in m³/d server-side; UI can expose m³/h or L/s via a lightweight converter on the editor (non-normative, cosmetic only).
- **MCMC chain configuration defaults** (P2): number of walkers, steps, burn-in — to be benchmarked at P2 design time.
