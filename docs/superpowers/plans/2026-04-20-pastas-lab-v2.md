# Pastas Lab v2 — Complete Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Pastas Lab from a basic fit+scenario tool into a comprehensive experimentation dashboard with data preview, cal/val split, full diagnostics, response visualization, multi-stress models, model gallery, comparison, hydrological signatures, and export.

**Architecture:** Extends existing `dashboard/utils/pastas/` (pure Python) + `api/routers/pastas.py` (FastAPI) + `frontend/src/pages/pastas/` (React). New pages added to PastasLayout tabs. New backend services as focused modules. All data sourced from BRGM `gold` schema via `station_loader.py`.

**Tech Stack:** Pastas 1.10+, MLflow, FastAPI, React 19, TanStack React Query 5, Plotly.js, Tailwind CSS 4

**Existing codebase (P1):**
- Backend: `dashboard/utils/pastas/{config,builder,fit_service,io,scenario,station_loader}.py`
- API: `api/routers/pastas.py`, `api/schemas/pastas.py`
- Frontend: `frontend/src/pages/pastas/{PastasLayout,FitPage,ScenariosPage}.tsx`
- Components: `frontend/src/components/pastas/{StationPicker,PastasConfigForm,FitResultsPanel,ScenarioComposer,ModificationCard,ScenarioResultsPanel}.tsx`
- Hooks: `frontend/src/hooks/usePastas.ts`
- API client: `frontend/src/lib/api.ts` (pastas namespace)

---

## File Map

### New backend files
```
dashboard/utils/pastas/diagnostics.py      # QQ, PACF, Durbin-Watson, runs test, normality
dashboard/utils/pastas/signatures.py       # Wrap ps.stats.signatures
dashboard/utils/pastas/comparison.py       # Multi-model comparison
dashboard/utils/pastas/stress_builder.py   # Build WellModel, custom stress series
```

### New API schemas + router extensions
```
api/schemas/pastas.py                      # Extended with new request/response models
api/routers/pastas.py                      # Extended with new endpoints
```

### New frontend pages
```
frontend/src/pages/pastas/GalleryPage.tsx  # Model gallery with table
frontend/src/pages/pastas/ComparePage.tsx  # Multi-model comparison
```

### New frontend components
```
frontend/src/components/pastas/DataPreviewPanel.tsx     # Series plots + stats before fit
frontend/src/components/pastas/StationMap.tsx            # Mini Plotly scattermapbox
frontend/src/components/pastas/ContributionsChart.tsx    # Stacked area decomposition
frontend/src/components/pastas/DiagnosticsPanel.tsx      # QQ, PACF, histogram, test cards
frontend/src/components/pastas/ResponsePanel.tsx         # Step/block/impulse response plots
frontend/src/components/pastas/CalValToggle.tsx           # Split controls + validation metrics
frontend/src/components/pastas/SignaturesPanel.tsx        # Radar chart of hydro signatures
frontend/src/components/pastas/StressListEditor.tsx      # Multi-stress config (well, custom)
frontend/src/components/pastas/ModelTable.tsx             # Sortable/filterable model table
frontend/src/components/pastas/ComparisonView.tsx        # Side-by-side metrics + overlay plots
frontend/src/components/pastas/ExportMenu.tsx             # Download .pas, CSV, PNG buttons
```

### Modified files
```
frontend/src/pages/pastas/PastasLayout.tsx  # Add Gallery, Compare tabs
frontend/src/pages/pastas/FitPage.tsx       # Add preview, cal/val, contributions, diagnostics, response, signatures, export
frontend/src/routes.tsx                     # Add gallery, compare routes
frontend/src/hooks/usePastas.ts             # Add new hooks
frontend/src/lib/api.ts                     # Add new endpoints
frontend/src/lib/types.ts                   # Add new types
```

---

## Group A: Data Preview & Station Context

### Task 1: Backend — station preview endpoint

**Files:**
- Modify: `api/routers/pastas.py`
- Modify: `api/schemas/pastas.py`

- [ ] **Step 1: Add preview schema to `api/schemas/pastas.py`**

```python
class StationPreview(BaseModel):
    code_bss: str
    metadata: dict[str, Any]
    piezo: TimeSeriesData
    precip: TimeSeriesData
    evap: TimeSeriesData
    stats: dict[str, Any]  # n_obs, date_range, gaps, nan_pct, mean, std, seasonality_strength
```

- [ ] **Step 2: Add preview endpoint to `api/routers/pastas.py`**

```python
@router.get("/preview/{code_bss}", response_model=StationPreview)
def preview_station(code_bss: str):
    from dashboard.utils.pastas.station_loader import load_station_series

    db_url = _brgm_url()

    try:
        station = load_station_series(code_bss, db_url)
    except ValueError as exc:
        raise HTTPException(404, str(exc))

    # Compute stats
    piezo = station.piezo
    stats = {
        "n_obs_piezo": len(piezo),
        "n_obs_precip": len(station.precip),
        "date_range": [str(piezo.index.min()), str(piezo.index.max())],
        "piezo_mean": float(piezo.mean()),
        "piezo_std": float(piezo.std()),
        "piezo_nan_pct": float(piezo.isna().mean() * 100),
        "precip_mean_mm_d": float(station.precip.mean()),
        "evap_mean_mm_d": float(station.evap.mean()),
    }

    # Gap analysis on piezo
    if len(piezo) > 1:
        gaps = piezo.index.to_series().diff().dt.days.dropna()
        stats["piezo_median_gap_days"] = float(gaps.median())
        stats["piezo_max_gap_days"] = float(gaps.max())
        stats["piezo_pct_daily"] = float((gaps == 1).mean() * 100)

    return StationPreview(
        code_bss=code_bss,
        metadata=station.metadata,
        piezo=_series_to_ts(piezo),
        precip=_series_to_ts(station.precip),
        evap=_series_to_ts(station.evap),
        stats=stats,
    )
```

Also extract `_brgm_url()` helper (used by both `/fit` and `/preview`):

```python
def _brgm_url() -> str:
    return (
        f"postgresql://{settings.brgm_db_user}:{settings.brgm_db_password}"
        f"@{settings.brgm_db_host}:{settings.brgm_db_port}/{settings.brgm_db_name}"
    )
```

- [ ] **Step 3: Add hook + API client**

In `frontend/src/hooks/usePastas.ts`:
```ts
export function usePastasPreview(codeBss: string | null) {
  return useQuery({
    queryKey: ['pastas', 'preview', codeBss],
    queryFn: () => api.pastas.preview(codeBss!),
    enabled: !!codeBss,
    staleTime: 10 * 60 * 1000,
  })
}
```

In `frontend/src/lib/api.ts` pastas namespace:
```ts
preview: (codeBss: string) => fetchJson<PastasStationPreview>(`/pastas/preview/${codeBss}`, { timeout: 30_000 }),
```

In `frontend/src/lib/types.ts`:
```ts
export interface PastasStationPreview {
  code_bss: string
  metadata: Record<string, unknown>
  piezo: TimeSeriesData
  precip: TimeSeriesData
  evap: TimeSeriesData
  stats: Record<string, number | string | string[]>
}
```

- [ ] **Step 4: Commit**

```bash
git add api/routers/pastas.py api/schemas/pastas.py frontend/src/hooks/usePastas.ts frontend/src/lib/api.ts frontend/src/lib/types.ts
git commit -m "feat(pastas): station preview endpoint with series + stats"
```

### Task 2: Frontend — DataPreviewPanel + StationMap

**Files:**
- Create: `frontend/src/components/pastas/DataPreviewPanel.tsx`
- Create: `frontend/src/components/pastas/StationMap.tsx`
- Modify: `frontend/src/pages/pastas/FitPage.tsx`

- [ ] **Step 1: Create DataPreviewPanel**

Shows 3 Plotly subplots (piezo, precip, evap) stacked vertically with shared x-axis, plus a stats summary grid. Uses `PastasStationPreview` type. Displays:
- Stacked time series (piezo as scatter, precip as bar, evap as line)
- Stats cards: n observations, date range, mean level, max gap, daily coverage %
- Plotly rangeslider on bottom subplot for interactive date selection

```tsx
import Plot from 'react-plotly.js'
import type { PastasStationPreview } from '@/lib/types'

interface Props {
  preview: PastasStationPreview
  onRangeChange?: (tmin: string, tmax: string) => void
}

export function DataPreviewPanel({ preview, onRangeChange }: Props) {
  const { piezo, precip, evap, stats, metadata } = preview

  return (
    <div className="space-y-3">
      {/* Station info header */}
      <div className="flex items-baseline gap-3">
        <span className="text-sm font-mono text-accent-cyan">{preview.code_bss}</span>
        <span className="text-xs text-text-muted">
          {metadata.nom_commune as string} ({metadata.code_departement as string})
        </span>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-5 gap-2">
        {[
          { label: 'Observations', value: stats.n_obs_piezo },
          { label: 'Période', value: Array.isArray(stats.date_range) ? `${(stats.date_range[0] as string)?.slice(0,4)}–${(stats.date_range[1] as string)?.slice(0,4)}` : '—' },
          { label: 'Niveau moyen', value: typeof stats.piezo_mean === 'number' ? `${stats.piezo_mean.toFixed(2)} m` : '—' },
          { label: 'Gap max', value: typeof stats.piezo_max_gap_days === 'number' ? `${stats.piezo_max_gap_days} j` : '—' },
          { label: 'Couv. journalière', value: typeof stats.piezo_pct_daily === 'number' ? `${stats.piezo_pct_daily.toFixed(0)}%` : '—' },
        ].map(({ label, value }) => (
          <div key={label} className="bg-bg-primary rounded-lg p-2 border border-white/5 text-center">
            <div className="text-[10px] text-text-muted">{label}</div>
            <div className="text-sm font-semibold text-text-primary">{value}</div>
          </div>
        ))}
      </div>

      {/* Stacked subplots */}
      <div className="bg-bg-card rounded-lg border border-white/5 p-2">
        <Plot
          data={[
            { x: piezo.index, y: piezo.values, name: 'Piézo (m)', type: 'scatter', mode: 'lines', line: { color: '#60a5fa', width: 1 }, xaxis: 'x', yaxis: 'y' },
            { x: precip.index, y: precip.values, name: 'Précip (mm/j)', type: 'bar', marker: { color: 'rgba(59,130,246,0.3)' }, xaxis: 'x', yaxis: 'y2' },
            { x: evap.index, y: evap.values, name: 'ETP (mm/j)', type: 'scatter', mode: 'lines', line: { color: '#f97316', width: 1 }, xaxis: 'x', yaxis: 'y3' },
          ]}
          layout={{
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9ca3af', size: 10 },
            margin: { t: 10, r: 20, b: 40, l: 50 },
            height: 380,
            showlegend: false,
            grid: { rows: 3, columns: 1, subplots: [['xy'], ['xy2'], ['xy3']], roworder: 'top to bottom' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.03)', rangeslider: { visible: true, thickness: 0.06 },
              ...(onRangeChange ? {} : {}),
            },
            yaxis: { title: 'Piézo (m)', gridcolor: 'rgba(255,255,255,0.05)', domain: [0.7, 1] },
            yaxis2: { title: 'P (mm/j)', gridcolor: 'rgba(255,255,255,0.05)', domain: [0.38, 0.65] },
            yaxis3: { title: 'ETP (mm/j)', gridcolor: 'rgba(255,255,255,0.05)', domain: [0.0, 0.3] },
          }}
          useResizeHandler
          className="w-full"
          onRelayout={(e: Record<string, unknown>) => {
            if (onRangeChange && e['xaxis.range[0]'] && e['xaxis.range[1]']) {
              onRangeChange(
                String(e['xaxis.range[0]']).slice(0, 10),
                String(e['xaxis.range[1]']).slice(0, 10),
              )
            }
          }}
        />
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create StationMap**

```tsx
import Plot from 'react-plotly.js'

interface Props {
  lat: number | null
  lon: number | null
  label: string
}

export function StationMap({ lat, lon, label }: Props) {
  if (lat == null || lon == null) return null

  return (
    <div className="bg-bg-card rounded-lg border border-white/5 overflow-hidden">
      <Plot
        data={[{
          type: 'scattermapbox',
          lat: [lat],
          lon: [lon],
          mode: 'markers',
          marker: { size: 12, color: '#22d3ee' },
          text: [label],
          hoverinfo: 'text',
        }]}
        layout={{
          mapbox: {
            style: 'carto-darkmatter',
            center: { lat, lon },
            zoom: 9,
          },
          margin: { t: 0, r: 0, b: 0, l: 0 },
          height: 180,
          paper_bgcolor: 'transparent',
        }}
        useResizeHandler
        className="w-full"
        config={{ displayModeBar: false }}
      />
    </div>
  )
}
```

- [ ] **Step 3: Integrate into FitPage**

In `FitPage.tsx`, after the StationPicker card:
- Call `usePastasPreview(codeBss)`
- When `preview` is loaded, show `<DataPreviewPanel>` and `<StationMap>` in the right column (above the fit results)
- Wire `onRangeChange` to auto-set tmin/tmax in the config form

- [ ] **Step 4: Verify frontend builds**

Run: `cd frontend && npm run build`

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/pastas/DataPreviewPanel.tsx \
  frontend/src/components/pastas/StationMap.tsx \
  frontend/src/pages/pastas/FitPage.tsx
git commit -m "feat(pastas): data preview panel with series plots, stats, and station map"
```

---

## Group B: Enhanced Fit — Cal/Val Split + Contributions

### Task 3: Backend — calibration/validation split

**Files:**
- Modify: `api/schemas/pastas.py`
- Modify: `dashboard/utils/pastas/fit_service.py`
- Modify: `api/routers/pastas.py`

- [ ] **Step 1: Add split fields to FitRequest + FitResponse**

In `api/schemas/pastas.py`, add to `FitRequest`:
```python
    val_split: Optional[float] = None  # fraction [0-1] for validation (e.g. 0.3 = last 30%)
```

Add to `FitResponse`:
```python
    validation_metrics: Optional[dict[str, float]] = None  # NSE, RMSE, etc. on val period
    cal_period: Optional[list[str]] = None   # [tmin_cal, tmax_cal]
    val_period: Optional[list[str]] = None   # [tmin_val, tmax_val]
```

- [ ] **Step 2: Implement split logic in fit_service.py**

In `run_fit()`, after building the model:
- If `val_split` is provided, compute `tmax_cal` as `piezo.index[int(len(piezo) * (1 - val_split))]`
- Solve on `[tmin, tmax_cal]`
- Simulate on `[tmax_cal, tmax]` for validation
- Compute validation metrics (NSE, RMSE, KGE, R²) on the validation window

```python
def _compute_val_metrics(model: ps.Model, tmin_val: str, tmax_val: str) -> dict[str, float]:
    """Compute metrics on a validation period (out-of-sample)."""
    val_metrics = {}
    for stat_name in ("nse", "kge", "rsq", "rmse", "evp"):
        try:
            val_metrics[stat_name] = float(getattr(model.stats, stat_name)(tmin=tmin_val, tmax=tmax_val))
        except Exception:
            pass
    return val_metrics
```

Add to `FitResult` dataclass:
```python
    validation_metrics: Optional[dict[str, float]] = None
    cal_period: Optional[list[str]] = None
    val_period: Optional[list[str]] = None
```

- [ ] **Step 3: Update router to pass val_split through**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(pastas): calibration/validation split with out-of-sample metrics"
```

### Task 4: Frontend — CalValToggle + ContributionsChart

**Files:**
- Create: `frontend/src/components/pastas/CalValToggle.tsx`
- Create: `frontend/src/components/pastas/ContributionsChart.tsx`
- Modify: `frontend/src/pages/pastas/FitPage.tsx`
- Modify: `frontend/src/components/pastas/FitResultsPanel.tsx`

- [ ] **Step 1: Create CalValToggle**

Simple slider/input for validation split percentage (0-50%), default off. Shows cal/val periods and validation metrics when available.

```tsx
interface Props {
  valSplit: number | null  // 0-1 or null for no split
  onChange: (v: number | null) => void
}

export function CalValToggle({ valSplit, onChange }: Props) {
  const enabled = valSplit !== null

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-text-secondary">Cal/Val split</label>
        <button
          onClick={() => onChange(enabled ? null : 0.3)}
          className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${
            enabled ? 'border-accent-cyan text-accent-cyan bg-accent-cyan/10' : 'border-white/10 text-text-muted'
          }`}
        >
          {enabled ? 'On' : 'Off'}
        </button>
      </div>
      {enabled && (
        <div>
          <input
            type="range"
            min={10}
            max={50}
            step={5}
            value={(valSplit ?? 0.3) * 100}
            onChange={(e) => onChange(+e.target.value / 100)}
            className="w-full accent-accent-cyan"
          />
          <div className="flex justify-between text-xs text-text-muted">
            <span>Cal: {((1 - (valSplit ?? 0.3)) * 100).toFixed(0)}%</span>
            <span>Val: {((valSplit ?? 0.3) * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Create ContributionsChart**

Stacked area chart showing the contribution of each stress model over time, from the `contributions` field in `FitResponse`.

```tsx
import Plot from 'react-plotly.js'
import type { TimeSeriesData } from '@/lib/types'

const COLORS = ['#60a5fa', '#34d399', '#f97316', '#a78bfa', '#f43f5e']

interface Props {
  contributions: Record<string, TimeSeriesData>
  observed?: TimeSeriesData
}

export function ContributionsChart({ contributions, observed }: Props) {
  const entries = Object.entries(contributions)
  if (entries.length === 0) return null

  const traces = entries.map(([name, ts], i) => ({
    x: ts.index,
    y: ts.values,
    name,
    type: 'scatter' as const,
    mode: 'lines' as const,
    stackgroup: 'one',
    line: { color: COLORS[i % COLORS.length], width: 0 },
    fillcolor: COLORS[i % COLORS.length] + '40',
  }))

  if (observed) {
    traces.push({
      x: observed.index,
      y: observed.values,
      name: 'Observed',
      type: 'scatter' as const,
      mode: 'lines' as const,
      stackgroup: undefined as unknown as string,
      line: { color: '#ffffff', width: 1.5 },
      fillcolor: undefined as unknown as string,
    })
  }

  return (
    <div className="bg-bg-card rounded-lg border border-white/5 p-2">
      <div className="text-xs font-semibold text-text-secondary mb-1 px-1">Decomposition</div>
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
          font: { color: '#9ca3af', size: 10 },
          margin: { t: 10, r: 20, b: 30, l: 50 },
          height: 220,
          xaxis: { gridcolor: 'rgba(255,255,255,0.03)' },
          yaxis: { title: 'm', gridcolor: 'rgba(255,255,255,0.05)' },
          legend: { orientation: 'h', y: -0.2, font: { size: 10 } },
        }}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
```

- [ ] **Step 3: Integrate into FitPage + FitResultsPanel**

- Add `CalValToggle` to FitPage config column, pass `val_split` in the fit mutation body
- Add `ContributionsChart` to `FitResultsPanel` after the obs/sim plot
- If `validation_metrics` is present in fit result, show a "Validation" section with metrics cards (same style, different color — green for good NSE, red for poor)
- Mark cal/val boundary with a vertical line on the obs/sim plot

- [ ] **Step 4: Update types.ts**

Add `val_split` to the fit body type. Add `validation_metrics`, `cal_period`, `val_period` to `PastasFitResponse`.

- [ ] **Step 5: Verify build, commit**

```bash
git commit -m "feat(pastas): cal/val split + contributions decomposition chart"
```

---

## Group C: Diagnostics Panel

### Task 5: Backend — diagnostics endpoint

**Files:**
- Create: `dashboard/utils/pastas/diagnostics.py`
- Modify: `api/routers/pastas.py`
- Modify: `api/schemas/pastas.py`

- [ ] **Step 1: Create diagnostics.py**

```python
"""Compute diagnostic statistics on a fitted Pastas model."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson, jarque_bera


def compute_diagnostics(residuals: pd.Series) -> dict[str, Any]:
    """Full diagnostic suite on model residuals."""
    clean = residuals.dropna()
    n = len(clean)

    result: dict[str, Any] = {"n_residuals": n}

    if n < 10:
        return result

    # Basic stats
    result["mean"] = float(clean.mean())
    result["std"] = float(clean.std())
    result["skewness"] = float(scipy_stats.skew(clean))
    result["kurtosis"] = float(scipy_stats.kurtosis(clean))

    # Normality tests
    if n >= 20:
        _, jb_pvalue = jarque_bera(clean)[:2]
        result["jarque_bera_pvalue"] = float(jb_pvalue)

        _, sw_pvalue = scipy_stats.shapiro(clean[:5000])  # shapiro limited to 5000
        result["shapiro_wilk_pvalue"] = float(sw_pvalue)

    # Durbin-Watson
    result["durbin_watson"] = float(durbin_watson(clean))

    # Ljung-Box at multiple lags
    for lag in [5, 10, 20]:
        if n > lag:
            try:
                lb = acorr_ljungbox(clean, lags=[lag], return_df=True)
                result[f"ljung_box_p_lag{lag}"] = float(lb["lb_pvalue"].iloc[0])
            except Exception:
                pass

    # Runs test for randomness
    median = clean.median()
    runs = ((clean > median).astype(int).diff().abs().sum() / 2) + 1
    result["runs_count"] = int(runs)

    # ACF / PACF
    nlags = min(40, n // 2 - 1)
    if nlags >= 2:
        result["acf_values"] = acf(clean, nlags=nlags, fft=True).tolist()
        result["pacf_values"] = pacf(clean, nlags=nlags).tolist()
        result["nlags"] = nlags
        result["confidence_bound"] = float(1.96 / np.sqrt(n))

    # QQ plot data (theoretical quantiles vs sorted residuals)
    sorted_res = np.sort(clean.values)
    theoretical = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, n))
    result["qq_theoretical"] = theoretical.tolist()
    result["qq_sample"] = sorted_res.tolist()

    # Histogram
    counts, bin_edges = np.histogram(clean, bins=30)
    result["hist_counts"] = counts.tolist()
    result["hist_bins"] = bin_edges.tolist()

    return result
```

- [ ] **Step 2: Add diagnostics endpoint**

```python
@router.get("/models/{run_id}/diagnostics")
def get_diagnostics(run_id: str):
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.diagnostics import compute_diagnostics

    try:
        model = load_model(run_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model '{run_id}' not found")

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tmin = run.data.params.get("tmin")
    tmax = run.data.params.get("tmax")

    residuals = model.residuals(tmin=tmin, tmax=tmax)
    return compute_diagnostics(residuals)
```

- [ ] **Step 3: Add hook + API client**

```ts
// hook
export function usePastasDiagnostics(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'diagnostics', runId],
    queryFn: () => api.pastas.diagnostics(runId!),
    enabled: !!runId,
  })
}

// api client
diagnostics: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/diagnostics`),
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(pastas): diagnostics endpoint — QQ, PACF, Durbin-Watson, normality tests"
```

### Task 6: Frontend — DiagnosticsPanel

**Files:**
- Create: `frontend/src/components/pastas/DiagnosticsPanel.tsx`
- Modify: `frontend/src/components/pastas/FitResultsPanel.tsx`

- [ ] **Step 1: Create DiagnosticsPanel**

Four sub-panels:
1. **QQ Plot** — scatter of theoretical vs sample quantiles with 1:1 line
2. **PACF Bar chart** — partial autocorrelation with confidence bands (±1.96/√n)
3. **Residual Histogram** — distribution with normal overlay
4. **Test results cards** — Durbin-Watson (ideal ~2), Jarque-Bera p, Shapiro-Wilk p, Ljung-Box p at lags 5/10/20, runs count
   - Color code: green if p > 0.05 (pass), red if p < 0.05 (fail)

```tsx
import Plot from 'react-plotly.js'

interface Props {
  diagnostics: Record<string, unknown>
}

export function DiagnosticsPanel({ diagnostics }: Props) {
  const qqTheoretical = diagnostics.qq_theoretical as number[] | undefined
  const qqSample = diagnostics.qq_sample as number[] | undefined
  const pacfValues = diagnostics.pacf_values as number[] | undefined
  const confBound = diagnostics.confidence_bound as number | undefined
  const histCounts = diagnostics.hist_counts as number[] | undefined
  const histBins = diagnostics.hist_bins as number[] | undefined

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
        Residual Diagnostics
      </div>

      {/* Test result badges */}
      <div className="flex flex-wrap gap-2">
        <TestBadge label="Durbin-Watson" value={diagnostics.durbin_watson as number} format={(v) => v.toFixed(2)} good={(v) => v > 1.5 && v < 2.5} />
        <TestBadge label="Jarque-Bera p" value={diagnostics.jarque_bera_pvalue as number} format={(v) => v.toFixed(3)} good={(v) => v > 0.05} />
        <TestBadge label="Shapiro-Wilk p" value={diagnostics.shapiro_wilk_pvalue as number} format={(v) => v.toFixed(3)} good={(v) => v > 0.05} />
        <TestBadge label="Ljung-Box p (lag 10)" value={diagnostics.ljung_box_p_lag10 as number} format={(v) => v.toFixed(3)} good={(v) => v > 0.05} />
      </div>

      <div className="grid grid-cols-2 gap-3">
        {/* QQ Plot */}
        {qqTheoretical && qqSample && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <Plot
              data={[
                { x: qqTheoretical, y: qqSample, type: 'scatter', mode: 'markers', marker: { color: '#60a5fa', size: 3 }, name: 'Residuals' },
                { x: [Math.min(...qqTheoretical), Math.max(...qqTheoretical)], y: [Math.min(...qqTheoretical), Math.max(...qqTheoretical)], type: 'scatter', mode: 'lines', line: { color: '#ef4444', dash: 'dash' }, name: '1:1' },
              ]}
              layout={{
                title: { text: 'QQ Plot', font: { size: 11 } },
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#9ca3af', size: 9 },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                height: 200,
                showlegend: false,
                xaxis: { title: 'Theoretical', gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: 'Sample', gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
            />
          </div>
        )}

        {/* PACF */}
        {pacfValues && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <Plot
              data={[
                { y: pacfValues, type: 'bar', marker: { color: '#60a5fa' }, name: 'PACF' },
                ...(confBound ? [
                  { y: Array(pacfValues.length).fill(confBound), type: 'scatter' as const, mode: 'lines' as const, line: { color: '#ef4444', dash: 'dash' as const, width: 1 }, name: '+95%', showlegend: false },
                  { y: Array(pacfValues.length).fill(-confBound), type: 'scatter' as const, mode: 'lines' as const, line: { color: '#ef4444', dash: 'dash' as const, width: 1 }, name: '-95%', showlegend: false },
                ] : []),
              ]}
              layout={{
                title: { text: 'Partial Autocorrelation', font: { size: 11 } },
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#9ca3af', size: 9 },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                height: 200,
                showlegend: false,
                xaxis: { title: 'Lag', gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
            />
          </div>
        )}

        {/* Histogram */}
        {histCounts && histBins && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2 col-span-2">
            <Plot
              data={[{
                x: histBins.slice(0, -1).map((b, i) => (b + histBins[i + 1]) / 2),
                y: histCounts,
                type: 'bar',
                marker: { color: 'rgba(96,165,250,0.4)', line: { color: '#60a5fa', width: 1 } },
                name: 'Residuals',
              }]}
              layout={{
                title: { text: 'Residual Distribution', font: { size: 11 } },
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#9ca3af', size: 9 },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                height: 180,
                xaxis: { title: 'Residual (m)', gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: 'Count', gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
            />
          </div>
        )}
      </div>
    </div>
  )
}

function TestBadge({ label, value, format, good }: {
  label: string; value: number | undefined; format: (v: number) => string; good: (v: number) => boolean
}) {
  if (value === undefined) return null
  const isGood = good(value)
  return (
    <div className={`px-2 py-1 rounded-md text-xs border ${
      isGood ? 'border-green-500/30 bg-green-500/10 text-green-400' : 'border-red-500/30 bg-red-500/10 text-red-400'
    }`}>
      {label}: {format(value)} {isGood ? '✓' : '✗'}
    </div>
  )
}
```

- [ ] **Step 2: Integrate into FitResultsPanel**

After the ACF chart, add a collapsible "Diagnostics" section that fetches diagnostics on demand via `usePastasDiagnostics(result.run_id)` and renders `<DiagnosticsPanel>`.

- [ ] **Step 3: Verify build, commit**

```bash
git commit -m "feat(pastas): diagnostics panel — QQ plot, PACF, histogram, stat tests"
```

---

## Group D: Response Function Visualization

### Task 7: Frontend — ResponsePanel

**Files:**
- Create: `frontend/src/components/pastas/ResponsePanel.tsx`
- Modify: `frontend/src/components/pastas/FitResultsPanel.tsx`

- [ ] **Step 1: Create ResponsePanel**

Shows step response and block response side-by-side. The data is already in `FitResponse.step_response` and `FitResponse.block_response`. Add:
- Step response plot with characteristic time annotation (time to 50% and 95% of final value)
- Block response plot
- Response parameters summary (A, a, n for Gamma)

```tsx
import Plot from 'react-plotly.js'
import type { TimeSeriesData, FitParameter } from '@/lib/types'

interface Props {
  stepResponse: TimeSeriesData
  blockResponse: TimeSeriesData
  parameters: FitParameter[]
  responseType: string
}

export function ResponsePanel({ stepResponse, blockResponse, parameters, responseType }: Props) {
  const hasStep = stepResponse?.values?.length > 0
  const hasBlock = blockResponse?.values?.length > 0

  // Extract response-specific parameters
  const responseParams = parameters.filter(p => p.name.startsWith('recharge_'))

  // Characteristic times from step response
  let t50: number | null = null
  let t95: number | null = null
  if (hasStep) {
    const finalVal = stepResponse.values[stepResponse.values.length - 1]
    if (finalVal !== 0) {
      t50 = stepResponse.values.findIndex(v => Math.abs(v) >= Math.abs(finalVal) * 0.5)
      t95 = stepResponse.values.findIndex(v => Math.abs(v) >= Math.abs(finalVal) * 0.95)
    }
  }

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
        Response Function — {responseType}
      </div>

      {/* Response parameters */}
      <div className="flex flex-wrap gap-2">
        {responseParams.map(p => (
          <div key={p.name} className="bg-bg-primary rounded px-2 py-1 text-xs border border-white/5">
            <span className="text-text-muted">{p.name.replace('recharge_', '')}</span>
            <span className="ml-1 text-text-primary font-mono">{p.optimal.toFixed(4)}</span>
            {p.stderr && <span className="text-text-muted"> ± {p.stderr.toFixed(4)}</span>}
          </div>
        ))}
        {t50 !== null && <div className="bg-bg-primary rounded px-2 py-1 text-xs border border-white/5"><span className="text-text-muted">t₅₀</span> <span className="font-mono text-text-primary">{t50}j</span></div>}
        {t95 !== null && <div className="bg-bg-primary rounded px-2 py-1 text-xs border border-white/5"><span className="text-text-muted">t₉₅</span> <span className="font-mono text-text-primary">{t95}j</span></div>}
      </div>

      <div className="grid grid-cols-2 gap-3">
        {hasStep && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <Plot
              data={[{ y: stepResponse.values, type: 'scatter', mode: 'lines', line: { color: '#34d399' }, name: 'Step' }]}
              layout={{
                title: { text: 'Step Response', font: { size: 11 } },
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#9ca3af', size: 9 },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                height: 200,
                xaxis: { title: 'Days', gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: 'm', gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
            />
          </div>
        )}
        {hasBlock && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <Plot
              data={[{ y: blockResponse.values, type: 'scatter', mode: 'lines', line: { color: '#f97316' }, name: 'Block' }]}
              layout={{
                title: { text: 'Block Response', font: { size: 11 } },
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#9ca3af', size: 9 },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                height: 200,
                xaxis: { title: 'Days', gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: 'm/d', gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
            />
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Add to FitResultsPanel after contributions**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(pastas): response function panel with step/block plots + characteristic times"
```

---

## Group E: Multi-Stress Models

### Task 8: Backend — stress builder + extended FitRequest

**Files:**
- Create: `dashboard/utils/pastas/stress_builder.py`
- Modify: `api/schemas/pastas.py`
- Modify: `dashboard/utils/pastas/builder.py`
- Modify: `api/routers/pastas.py`

- [ ] **Step 1: Create stress_builder.py**

```python
"""Build additional Pastas stress models (wells, custom series)."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import pastas as ps

from dashboard.utils.pastas.config import RFUNC_REGISTRY


def build_well_stress(
    q_series: pd.Series,
    name: str,
    rfunc_type: str = "Exponential",
    settings: str = "well",
) -> ps.StressModel:
    """Build a well pumping stress model."""
    rfunc_cls = RFUNC_REGISTRY.get(rfunc_type, ps.Exponential)
    return ps.StressModel(q_series, rfunc=rfunc_cls(), name=name, settings=settings)


def build_river_stress(
    river_series: pd.Series,
    name: str,
    rfunc_type: str = "Exponential",
) -> ps.StressModel:
    """Build a river/surface water stress model."""
    rfunc_cls = RFUNC_REGISTRY.get(rfunc_type, ps.Exponential)
    return ps.StressModel(river_series, rfunc=rfunc_cls(), name=name, settings="waterlevel")


def build_custom_stress(
    series: pd.Series,
    name: str,
    rfunc_type: str = "Gamma",
    settings: str = "prec",
) -> ps.StressModel:
    """Build a custom stress model from an arbitrary series."""
    rfunc_cls = RFUNC_REGISTRY.get(rfunc_type, ps.Gamma)
    return ps.StressModel(series, rfunc=rfunc_cls(), name=name, settings=settings)
```

- [ ] **Step 2: Add stress config schemas**

In `api/schemas/pastas.py`:
```python
class AdditionalStress(BaseModel):
    type: Literal["well", "river", "custom"]
    name: str
    rfunc: str = "Exponential"
    # For well/river: column name in hubeau_daily_chroniques or separate source
    source: Literal["upload", "db_column"] = "upload"
    column: Optional[str] = None  # if source=db_column
    csv_rows: Optional[list[dict]] = None  # if source=upload: [{date, value}]
```

Add to `FitRequest`:
```python
    additional_stresses: list[AdditionalStress] = []
```

- [ ] **Step 3: Update builder.py to add extra stresses**

After adding the RechargeModel, iterate `additional_stresses` and add each via `stress_builder.build_*_stress()`.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(pastas): multi-stress support — well, river, custom stress models"
```

### Task 9: Frontend — StressListEditor

**Files:**
- Create: `frontend/src/components/pastas/StressListEditor.tsx`
- Modify: `frontend/src/pages/pastas/FitPage.tsx`

- [ ] **Step 1: Create StressListEditor**

A list of additional stress configurations. Each entry has:
- Type selector (well / river / custom)
- Name input
- Response function selector
- CSV upload for the stress series (date, value)
- Delete button

Same pattern as ScenarioComposer — list of cards with "Add stress" button.

- [ ] **Step 2: Integrate into FitPage config column**

Add after the PastasConfigForm card, before CalValToggle.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(pastas): stress list editor for multi-stress fit configuration"
```

---

## Group F: Model Gallery & Comparison

### Task 10: Frontend — GalleryPage with ModelTable

**Files:**
- Create: `frontend/src/pages/pastas/GalleryPage.tsx`
- Create: `frontend/src/components/pastas/ModelTable.tsx`
- Create: `frontend/src/components/pastas/ExportMenu.tsx`
- Modify: `frontend/src/pages/pastas/PastasLayout.tsx`
- Modify: `frontend/src/routes.tsx`

- [ ] **Step 1: Create ModelTable**

Sortable/filterable table with columns:
- Name (link to detail), Station (code_bss), Response type, EVP, RMSE, NSE, Created, Actions (delete, export .pas)

Uses `usePastasModels()`. Sorting is client-side (TanStack Table or simple state-based sort). Filterable by station search.

```tsx
import { useState } from 'react'
import { Trash2, Download, ArrowUpDown } from 'lucide-react'
import { usePastasModels, usePastasDeleteModel } from '@/hooks/usePastas'
import { ExportMenu } from './ExportMenu'
import type { PastasModelSummary } from '@/lib/types'

type SortKey = 'name' | 'code_bss' | 'evp' | 'rmse' | 'created_at'

export function ModelTable() {
  const { data: models, isLoading } = usePastasModels()
  const deleteMutation = usePastasDeleteModel()
  const [sortKey, setSortKey] = useState<SortKey>('created_at')
  const [sortAsc, setSortAsc] = useState(false)
  const [filter, setFilter] = useState('')

  const filtered = (models ?? []).filter(m =>
    m.code_bss.toLowerCase().includes(filter.toLowerCase()) ||
    m.name.toLowerCase().includes(filter.toLowerCase())
  )

  const sorted = [...filtered].sort((a, b) => {
    const va = a[sortKey] ?? ''
    const vb = b[sortKey] ?? ''
    const cmp = typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb))
    return sortAsc ? cmp : -cmp
  })

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc)
    else { setSortKey(key); setSortAsc(true) }
  }

  // ... render table with sortable headers, rows, delete + export actions
}
```

- [ ] **Step 2: Create ExportMenu**

Dropdown with options:
- Download .pas file (link to MLflow artifact)
- Download metrics CSV
- Copy run ID

```tsx
import { useState } from 'react'
import { Download, ChevronDown } from 'lucide-react'

interface Props {
  runId: string
  modelName: string
}

export function ExportMenu({ runId, modelName }: Props) {
  const [open, setOpen] = useState(false)

  return (
    <div className="relative">
      <button onClick={() => setOpen(!open)} className="p-1 hover:bg-bg-hover rounded text-text-muted hover:text-text-primary">
        <Download className="w-4 h-4" />
      </button>
      {open && (
        <div className="absolute right-0 mt-1 bg-bg-card border border-white/10 rounded-lg shadow-xl z-10 py-1 w-44">
          <a href={`/api/v1/pastas/models/${runId}/export/pas`} download={`${modelName}.pas`}
            className="block px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover">
            Download .pas
          </a>
          <a href={`/api/v1/pastas/models/${runId}/export/csv`} download={`${modelName}_metrics.csv`}
            className="block px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover">
            Download metrics CSV
          </a>
          <button onClick={() => { navigator.clipboard.writeText(runId); setOpen(false) }}
            className="block w-full text-left px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover">
            Copy run ID
          </button>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Create GalleryPage**

```tsx
import { ModelTable } from '@/components/pastas/ModelTable'

export default function GalleryPage() {
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold text-text-primary mb-4">Pastas — Model Gallery</h1>
      <ModelTable />
    </div>
  )
}
```

- [ ] **Step 4: Add export endpoints to router**

```python
@router.get("/models/{run_id}/export/pas")
def export_pas(run_id: str):
    from dashboard.utils.pastas.io import load_model
    from fastapi.responses import FileResponse
    import tempfile

    model = load_model(run_id)
    with tempfile.NamedTemporaryFile(suffix=".pas", delete=False) as f:
        model.to_file(f.name)
        return FileResponse(f.name, filename=f"model_{run_id[:8]}.pas", media_type="application/octet-stream")

@router.get("/models/{run_id}/export/csv")
def export_csv(run_id: str):
    from fastapi.responses import Response
    import csv, io

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    params = run.data.params

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["key", "value"])
    for k, v in {**params, **{f"metric_{mk}": mv for mk, mv in metrics.items()}}.items():
        writer.writerow([k, v])

    return Response(content=output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=pastas_{run_id[:8]}.csv"})
```

- [ ] **Step 5: Update PastasLayout + routes**

Add Gallery tab to PastasLayout:
```tsx
const PASTAS_TABS = [
  { to: '/pastas/fit', icon: SlidersHorizontal, label: 'Fit' },
  { to: '/pastas/scenarios', icon: FlaskConical, label: 'Scenarios' },
  { to: '/pastas/gallery', icon: LayoutGrid, label: 'Gallery' },
  { to: '/pastas/compare', icon: GitCompareArrows, label: 'Compare' },
] as const
```

Add routes in `routes.tsx`.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(pastas): model gallery page with sortable table + export (.pas, CSV)"
```

### Task 11: Backend + Frontend — Model Comparison

**Files:**
- Create: `dashboard/utils/pastas/comparison.py`
- Modify: `api/routers/pastas.py`
- Create: `frontend/src/pages/pastas/ComparePage.tsx`
- Create: `frontend/src/components/pastas/ComparisonView.tsx`

- [ ] **Step 1: Create comparison.py**

```python
"""Compare multiple fitted Pastas models."""
from __future__ import annotations

from typing import Any

import pandas as pd
import pastas as ps

from dashboard.utils.pastas.io import load_model


def compare_models(run_ids: list[str]) -> dict[str, Any]:
    """Load N models and return side-by-side metrics + aligned series."""
    results = []

    for run_id in run_ids:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        model = load_model(run_id)

        tmin = run.data.params.get("tmin")
        tmax = run.data.params.get("tmax")

        sim = model.simulate(tmin=tmin, tmax=tmax)
        obs = model.observations(tmin=tmin, tmax=tmax)

        results.append({
            "run_id": run_id,
            "name": run.info.run_name or run_id[:8],
            "params": run.data.params,
            "metrics": run.data.metrics,
            "observed": obs,
            "simulated": sim,
        })

    return {"models": results}
```

- [ ] **Step 2: Add compare endpoint**

```python
@router.post("/compare")
def compare_models_endpoint(run_ids: list[str]):
    from dashboard.utils.pastas.comparison import compare_models

    if len(run_ids) < 2 or len(run_ids) > 5:
        raise HTTPException(422, "Provide 2-5 run IDs")

    result = compare_models(run_ids)

    return {
        "models": [
            {
                "run_id": m["run_id"],
                "name": m["name"],
                "params": m["params"],
                "metrics": m["metrics"],
                "observed": _series_to_ts(m["observed"]),
                "simulated": _series_to_ts(m["simulated"]),
            }
            for m in result["models"]
        ]
    }
```

- [ ] **Step 3: Create ComparisonView**

- Metrics comparison table (rows = metrics, columns = models)
- Overlay plot: all simulations + observed on same axes (different colors)
- Delta plots: each model's residuals

- [ ] **Step 4: Create ComparePage**

- Multi-select from model gallery (checkboxes)
- "Compare" button → shows ComparisonView
- Or: select from URL params

- [ ] **Step 5: Add hook + API, commit**

```bash
git commit -m "feat(pastas): multi-model comparison page with overlay plots + metrics table"
```

---

## Group G: Hydrological Signatures

### Task 12: Backend + Frontend — Signatures

**Files:**
- Create: `dashboard/utils/pastas/signatures.py`
- Modify: `api/routers/pastas.py`
- Create: `frontend/src/components/pastas/SignaturesPanel.tsx`
- Modify: `frontend/src/components/pastas/FitResultsPanel.tsx`

- [ ] **Step 1: Create signatures.py**

```python
"""Compute hydrological signatures using Pastas stats."""
from __future__ import annotations

from typing import Any

import pandas as pd


def compute_signatures(observed: pd.Series, simulated: pd.Series) -> dict[str, dict[str, float]]:
    """Compute groundwater signatures on both observed and simulated.

    Returns dict with keys 'observed' and 'simulated', each containing
    signature name → value.
    """
    import pastas as ps

    result = {}
    for label, series in [("observed", observed), ("simulated", simulated)]:
        sigs = {}
        # Core signatures available in pastas.stats
        sig_functions = [
            "cv_period_mean", "parde_seasonality", "avg_seasonal_fluctuation",
            "interannual_variation", "rise_rate", "fall_rate",
            "bimodality_coefficient", "recession_constant", "recovery_constant",
            "colwell_constancy", "colwell_contingency",
            "mean_annual_maximum", "autocorr_time",
        ]
        for sig_name in sig_functions:
            try:
                func = getattr(ps.stats.signatures, sig_name)
                val = float(func(series))
                sigs[sig_name] = val
            except Exception:
                pass
        result[label] = sigs

    return result
```

- [ ] **Step 2: Add signatures endpoint**

```python
@router.get("/models/{run_id}/signatures")
def get_signatures(run_id: str):
    from dashboard.utils.pastas.io import load_model
    from dashboard.utils.pastas.signatures import compute_signatures

    model = load_model(run_id)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    tmin = run.data.params.get("tmin")
    tmax = run.data.params.get("tmax")

    obs = model.observations(tmin=tmin, tmax=tmax)
    sim = model.simulate(tmin=tmin, tmax=tmax)

    return compute_signatures(obs, sim)
```

- [ ] **Step 3: Create SignaturesPanel**

Radar/spider chart comparing observed vs simulated signatures. Uses Plotly `scatterpolar`.

```tsx
import Plot from 'react-plotly.js'

interface Props {
  signatures: { observed: Record<string, number>; simulated: Record<string, number> }
}

export function SignaturesPanel({ signatures }: Props) {
  const keys = Object.keys(signatures.observed)
  const obsValues = keys.map(k => signatures.observed[k])
  const simValues = keys.map(k => signatures.simulated[k] ?? 0)

  // Normalize to [0,1] for radar
  const maxVals = keys.map((k, i) => Math.max(Math.abs(obsValues[i]), Math.abs(simValues[i]), 1e-6))
  const obsNorm = obsValues.map((v, i) => v / maxVals[i])
  const simNorm = simValues.map((v, i) => v / maxVals[i])

  // Readable labels
  const labels = keys.map(k => k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))

  return (
    <div className="bg-bg-card rounded-lg border border-white/5 p-3">
      <div className="text-xs font-semibold text-text-secondary mb-1">Hydrological Signatures</div>
      <Plot
        data={[
          { type: 'scatterpolar', r: [...obsNorm, obsNorm[0]], theta: [...labels, labels[0]], name: 'Observed', fill: 'toself', fillcolor: 'rgba(96,165,250,0.1)', line: { color: '#60a5fa' } },
          { type: 'scatterpolar', r: [...simNorm, simNorm[0]], theta: [...labels, labels[0]], name: 'Simulated', fill: 'toself', fillcolor: 'rgba(249,115,22,0.1)', line: { color: '#f97316' } },
        ]}
        layout={{
          paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
          font: { color: '#9ca3af', size: 9 },
          margin: { t: 30, r: 40, b: 30, l: 40 },
          height: 350,
          polar: {
            bgcolor: 'transparent',
            radialaxis: { visible: true, gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.05)' },
            angularaxis: { gridcolor: 'rgba(255,255,255,0.1)', linecolor: 'rgba(255,255,255,0.05)' },
          },
          legend: { orientation: 'h', y: -0.1, font: { size: 10 } },
        }}
        useResizeHandler className="w-full"
      />

      {/* Raw values table */}
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-text-muted border-b border-white/5">
              <th className="text-left px-2 py-1">Signature</th>
              <th className="text-right px-2 py-1">Observed</th>
              <th className="text-right px-2 py-1">Simulated</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k, i) => (
              <tr key={k} className="border-b border-white/5">
                <td className="px-2 py-1 text-text-secondary">{labels[i]}</td>
                <td className="px-2 py-1 text-right font-mono text-text-primary">{obsValues[i].toFixed(4)}</td>
                <td className="px-2 py-1 text-right font-mono text-text-primary">{simValues[i].toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Add hook + API client, integrate into FitResultsPanel**

```ts
export function usePastasSignatures(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'signatures', runId],
    queryFn: () => api.pastas.signatures(runId!),
    enabled: !!runId,
  })
}
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(pastas): hydrological signatures — radar chart observed vs simulated"
```

---

## Summary: Task List

| # | Group | Task | Key deliverable |
|---|-------|------|-----------------|
| 1 | A | Station preview endpoint | `GET /preview/{code_bss}` — series + stats |
| 2 | A | DataPreviewPanel + StationMap | 3 stacked plots + stats cards + mini map |
| 3 | B | Cal/val split backend | `val_split` in FitRequest, validation metrics |
| 4 | B | CalValToggle + ContributionsChart | Split slider + stacked decomposition |
| 5 | C | Diagnostics backend | `GET /models/{id}/diagnostics` — QQ, PACF, DW |
| 6 | C | DiagnosticsPanel | QQ plot, PACF, histogram, test badges |
| 7 | D | ResponsePanel | Step/block plots + characteristic times + params |
| 8 | E | Multi-stress backend | `stress_builder.py` + extended FitRequest |
| 9 | E | StressListEditor | UI to add well/river/custom stresses |
| 10 | F | GalleryPage + export | Model table + export .pas/.csv |
| 11 | F | ComparePage | Side-by-side metrics + overlay plots |
| 12 | G | Signatures | Radar chart + values table |

**Recommended execution order:** 1→2→3→4→5→6→7→10→12→8→9→11

Tasks 1-7 enhance the core fit workflow. Task 10 adds model management. Task 12 adds analytical depth. Tasks 8-9 (multi-stress) and 11 (comparison) are more complex and build on the gallery.
