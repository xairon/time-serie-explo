# Outlier Diagnostics Module — Design Spec

**Date**: 2026-04-23
**Scope**: Pastas residual outlier investigation tool
**Status**: Draft

## Problem

After fitting a Pastas groundwater model, residuals are displayed as a bar chart with red bars for |residual| > 2σ. Currently there is no way to investigate WHY a specific residual is an outlier. Users must manually cross-reference climate data, data quality, and neighboring stations to understand whether the issue is a model limitation, a data problem, or an extreme event.

## Goals

1. Automatically classify each outlier into a diagnostic category with a confidence score
2. Cross-reference residuals with climate data, drought indices, data quality, and neighboring stations
3. Let users click on a red bar to see a contextual diagnostic panel that expands below the chart
4. Discriminate local anomalies (station-specific) from regional signals (aquifer-wide) using BDLISA siblings' raw observations (no model required for neighbors)

## Non-Goals

- Automatic model re-fitting or parameter suggestions
- Requiring Pastas models on neighboring stations
- Real-time streaming diagnostics during fitting

---

## Architecture

### Overview

One new backend endpoint computes diagnostics for all outliers in batch. One new frontend component renders the detail panel. The existing residuals bar chart becomes interactive.

```
GET /api/v1/pastas/models/{run_id}/outlier-diagnostics
  → backend loads model (LRU cache)
  → extracts residuals, identifies outliers (|r| > 2σ)
  → for each outlier, builds OutlierDiagnostic by cross-referencing:
      - model contributions (per stress)
      - climate data vs historical calendar-month averages
      - data quality (gaps ±30 days)
      - BDLISA siblings raw observation z-scores
      - SPLI/SPI drought indices
  → classifies into category + secondary tags
  → returns list sorted by severity descending

Frontend:
  - FitResultsPanel fetches outlier-diagnostics (React Query, alongside existing diagnostics)
  - Red bars in residuals chart get cursor:pointer + hover effect
  - Click on bar → expand OutlierDetailPanel below chart (no network request)
  - Click again or ✕ → collapse
```

### Data Flow

```
[Pastas Model Cache] ─→ residuals + contributions
[gold.fct_monthly_chroniques] ─→ climate context (precip, temp, ETP, historical averages)
[gold.fct_monthly_chroniques] ─→ neighbor z-scores (same table, different code_bss)
[SPLI/SPI compute] ─→ drought indices for the station
[gold.hubeau_daily_chroniques] ─→ data quality / gap detection
         │
         ▼
  compute_outlier_diagnostics()
         │
         ▼
  List[OutlierDiagnostic] sorted by severity
```

---

## Backend

### New File: `dashboard/utils/pastas/outlier_diagnostics.py`

#### Main Function

```python
def compute_outlier_diagnostics(
    model: ps.Model,
    code_bss: str,
    cal_tmin: str,
    cal_tmax: str,
    engine: Engine,
) -> dict:
```

#### Outlier Detection

1. Extract residuals from model over calibration period
2. Compute σ (standard deviation) and threshold = 2σ
3. Resample residuals to monthly (mean) for alignment with monthly data sources
4. Identify months where |residual_monthly| > threshold

#### Climate Context (per outlier month)

For each outlier month, query `gold.fct_monthly_chroniques` for the station:
- `precipitation_totale`, `temperature_moyenne`, `evaporation_moyenne`
- Compute z-score against all historical values of the same calendar month (e.g., all Marches)
- Fetch pre-computed SPLI and SPI for that month (reuse `compute_spli` / `compute_spi`)

#### Data Quality (per outlier month)

Query `gold.hubeau_daily_chroniques` for ±30 days around the outlier:
- Count days with NULL `niveau_nappe_eau` → `gap_days`
- Compute `coverage_pct` in the 60-day window
- Find nearest gap > 1 day → `nearest_gap_distance_days`

#### Neighbor Comparison

1. Query BDLISA siblings (reuse sibling logic from observatory router)
2. For each sibling, query their `niveau_moyen` from `gold.fct_monthly_chroniques` for all months of the same calendar month
3. Compute z-score of the outlier month vs the sibling's own historical distribution
4. Flag sibling as "anomalous" if |z-score| > 1.5
5. Return count anomalous / total + per-sibling z-scores

#### Model Contributions (per outlier month)

For each stressmodel in the Pastas model:
- Call `model.get_contribution(name)` and extract the value for the outlier month
- This shows which stress drove the model prediction that month

#### Classification Rules

Classification is a two-pass process:

**Pass 1 — Per-outlier rules** (evaluated in priority order per outlier):

| Priority | Category | Rule |
|----------|----------|------|
| 1 | `DATA_GAP` | `gap_days >= 1` in ±30 day window |
| 2 | `CLIMATE_EXTREME` | Any climate variable z-score > 2.0 |
| 3 | `REGIONAL_SIGNAL` | ≥50% of BDLISA siblings have |z-score| > 1.5 |
| 4 | `DOMINANT_CONTRIBUTION` | One contribution accounts for >80% of the total contribution variance that month |

**Pass 2 — Global pattern rules** (evaluated after all outliers are classified):

| Priority | Category | Rule |
|----------|----------|------|
| 5 | `SEASONAL_BIAS` | ≥3 outliers of the same sign fall in the same calendar quarter. Applied as secondary tag to all affected outliers; becomes primary only if no Pass 1 rule matched. |
| 6 | `UNKNOWN` | No rule matched after both passes |

First match in Pass 1 = primary category. Subsequent matches (Pass 1 + Pass 2) = secondary tags.

#### Severity Score

```python
severity = min(1.0, abs(residual) / (3 * sigma))
```

#### Explanation Generation

Template-based string per category:

- `DATA_GAP`: "Data gap of {gap_days} days detected within ±30 days. Model interpolation may be unreliable."
- `CLIMATE_EXTREME`: "Monthly {variable} was {z}σ {above/below} normal ({value} vs {avg} avg)."
- `REGIONAL_SIGNAL`: "{n}/{total} neighboring stations also show anomalous levels this month."
- `SEASONAL_BIAS`: "{count} outliers with {sign} residuals cluster in Q{quarter}, suggesting systematic seasonal model error."
- `DOMINANT_CONTRIBUTION`: "The {stress_name} contribution ({value}m) dominates model response this month."
- `UNKNOWN`: "No clear cause identified. Residual is {z}σ from model expectation."

Multiple explanations are concatenated when secondary tags exist.

#### Summary Computation

After classifying all outliers:
- Count by category → `by_category`
- Count by calendar quarter → `seasonal_pattern`
- Compute `median_severity`

### API Response Schema

```python
class ClimateContext(BaseModel):
    precip_mm: float | None
    precip_zscore: float | None
    temp_c: float | None
    temp_zscore: float | None
    etp_mm: float | None
    etp_zscore: float | None
    spli: float | None
    spli_class: str | None
    spi: float | None
    spi_class: str | None

class DataQuality(BaseModel):
    gap_days: int
    coverage_pct: float
    nearest_gap_distance_days: int | None

class NeighborZscore(BaseModel):
    code_bss: str
    zscore: float

class NeighborContext(BaseModel):
    total: int
    anomalous: int
    neighbor_zscores: list[NeighborZscore]

class OutlierDiagnostic(BaseModel):
    date: str
    residual: float
    residual_zscore: float
    severity: float
    category: str
    category_label: str
    secondary_tags: list[str]
    explanation: str
    climate: ClimateContext
    contributions: dict[str, float]
    observed: float
    simulated: float
    data_quality: DataQuality
    neighbors: NeighborContext

class OutlierDiagnosticsResponse(BaseModel):
    run_id: str
    code_bss: str
    sigma: float
    threshold: float
    n_residuals: int
    n_outliers: int
    outliers: list[OutlierDiagnostic]
    summary: OutlierSummary

class OutlierSummary(BaseModel):
    by_category: dict[str, int]          # e.g. {"CLIMATE_EXTREME": 3, "DATA_GAP": 2}
    seasonal_pattern: dict[str, int]     # e.g. {"Q1": 1, "Q2": 3, "Q3": 0, "Q4": 4}
    median_severity: float
```

### New API Endpoint

In `api/routers/pastas.py`:

```
GET /api/v1/pastas/models/{run_id}/outlier-diagnostics
```

Loads model via existing `load_model()`, extracts `code_bss` and `cal_period` from MLflow tags/params, calls `compute_outlier_diagnostics()`, returns `OutlierDiagnosticsResponse`.

Cache: use existing `get_cached()` with TTL = 3600s (same as model detail).

---

## Frontend

### Modified: FitResultsPanel.tsx

1. Add React Query hook `usePastasOutlierDiagnostics(runId)` — fetches on mount alongside existing diagnostics
2. Pass `outlierDiagnostics` to the residuals chart section
3. Add state: `selectedOutlierDate: string | null`
4. Make red bars clickable (Plotly `plotly_click` event) → set `selectedOutlierDate`
5. Below the residuals chart, conditionally render `<OutlierDetailPanel>` when a date is selected
6. Above the residuals chart, render outlier summary line when data is loaded:
   ```
   8 outliers detected — 3 climate, 2 data gaps, 2 seasonal bias, 1 unknown
   ```

### Modified: Residuals bar chart (in FitResultsPanel)

- Add `customdata` to bar trace containing each date's outlier status
- Register `onClick` handler on Plot component
- Red bars get `cursor: pointer` via CSS on hover
- Selected bar gets a highlight ring (white border or brighter opacity)

### New Component: OutlierDetailPanel.tsx

Props:
```typescript
interface OutlierDetailPanelProps {
  outlier: OutlierDiagnostic
  onClose: () => void
}
```

Layout (expands below residuals chart with slide-down animation):

**Row 1 — Header**
- Category badge (colored by type: red=DATA_GAP, orange=CLIMATE_EXTREME, blue=REGIONAL_SIGNAL, yellow=SEASONAL_BIAS, purple=DOMINANT_CONTRIBUTION, gray=UNKNOWN)
- Secondary tag badges (smaller, outline style)
- Date + residual value + z-score
- Severity dots (1-4 filled circles)
- Close button (✕)

**Row 2 — Explanation**
- Natural language explanation string, full width, text-sm

**Row 3 — Context grid (3 equal columns)**

Column 1 "Climate":
- Precip: value + z-score badge (bold if |z| > 1.5)
- Temp: value + z-score badge
- ETP: value + z-score badge
- SPLI: value + classification badge (reuse existing ClassificationBadge)
- SPI: value + classification badge

Column 2 "Model":
- Per-contribution name: value in meters
- Observed vs Simulated values

Column 3 "Data Quality":
- Coverage %
- Gap days in ±30d window
- Nearest gap distance

**Row 4 — Neighbors**
- "BDLISA neighbors: {anomalous}/{total} anomalous this month"
- Inline badges per neighbor: `[code_bss: +2.1σ]` colored red if |z| > 1.5, gray otherwise
- If no siblings available: "No BDLISA neighbors found"

### New Hook: usePastasOutlierDiagnostics

In `frontend/src/hooks/usePastas.ts`:

```typescript
export function usePastasOutlierDiagnostics(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'outlier-diagnostics', runId],
    queryFn: () => api.pastas.outlierDiagnostics(runId!),
    enabled: !!runId,
    staleTime: 60 * 60 * 1000, // 1h
  })
}
```

### New API Client Method

In `frontend/src/lib/api.ts` (or wherever Pastas API methods live):

```typescript
outlierDiagnostics: (runId: string) =>
  fetchJson<OutlierDiagnosticsResponse>(`/pastas/models/${runId}/outlier-diagnostics`)
```

---

## Category Styling

| Category | Color | Icon suggestion |
|----------|-------|-----------------|
| `DATA_GAP` | Red 500 | Gap/broken line |
| `CLIMATE_EXTREME` | Orange 500 | Lightning/storm |
| `REGIONAL_SIGNAL` | Blue 500 | Globe/network |
| `SEASONAL_BIAS` | Yellow 500 | Calendar/cycle |
| `DOMINANT_CONTRIBUTION` | Purple 500 | Bar chart |
| `UNKNOWN` | Gray 500 | Question mark |

---

## Testing Strategy

### Backend

- Unit test `compute_outlier_diagnostics()` with a synthetic Pastas model (mock residuals + contributions)
- Test each classification rule in isolation (provide residuals that trigger exactly one rule)
- Test priority ordering (DATA_GAP should win over CLIMATE_EXTREME when both match)
- Test edge cases: 0 outliers, all outliers, station with no BDLISA siblings
- Test z-score computation with known distributions

### Frontend

- Component test for `OutlierDetailPanel` with mock OutlierDiagnostic objects per category
- Integration test: click red bar → panel expands with correct data
- Edge case: no outliers → summary line says "No outliers detected"

---

## Performance Considerations

- Neighbor z-score computation requires one query per sibling per outlier month. Batch into a single query: fetch all months × all siblings in one SQL call, then pivot in Python.
- SPLI/SPI computation is already cached (86400s TTL). Reuse cached values.
- Typical outlier count: 5-20 for a 10-30 year model. Batch computation should complete in 1-3s.
- Model loading is LRU-cached (maxsize=32). No extra cost if model was recently loaded.

---

## Files to Create/Modify

### New Files
- `dashboard/utils/pastas/outlier_diagnostics.py` — core computation
- `frontend/src/components/pastas/OutlierDetailPanel.tsx` — detail panel component
- `api/schemas/pastas_outlier.py` — Pydantic response models

### Modified Files
- `api/routers/pastas.py` — add endpoint
- `frontend/src/hooks/usePastas.ts` — add hook
- `frontend/src/lib/api.ts` — add API client method
- `frontend/src/components/pastas/FitResultsPanel.tsx` — integrate clickable bars + panel + summary
