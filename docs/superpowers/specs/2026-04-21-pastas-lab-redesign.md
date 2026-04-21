# Pastas Lab Redesign — Hydrogeologist-Oriented Fitting Pipeline

**Date:** 2026-04-21
**Status:** Approved
**Scope:** Complete redesign of the Pastas Lab (frontend + backend) to match a professional hydrogeologist workflow

---

## Problem Statement

The current Pastas Lab does a single blind fit with user-selected config, producing poor results (NSE ~0.15) on stations with 18k+ observations. No pre-fit diagnostics, no warm-up, no model selection, no quality screening. The tool doesn't match how hydrogeologists actually work.

## Target Users

- BRGM hydrogeologists familiar with Gardénia but not necessarily Pastas
- Researchers who want full control over TFN model parameters
- Both served by a dual-mode UI (Guided / Expert)

## Design Decisions

- **Dual-mode UX**: Guided (automatic, 1-click) and Expert (all parameters exposed). Same engine, different UI surface.
- **STOWA screening**: Dutch standard 4-criteria pass/fail assessment adopted as quality gate.
- **Two-pass optimization**: Solve without noise → re-solve with noise using first params as initial. Standard Pastas best practice.
- **Architecture extensible**: designed to accommodate RAMEAU (BRGM process-based model) as a future 3rd modeling backend.
- **AI comparison**: when a DL model exists for the same station, overlay and compare on the common period.

---

## Architecture: 5-Step Pipeline

```
[1. Station]  →  [2. Calibrate]  →  [3. Results]  →  [4. Scenarios]  →  [5. Gallery]
```

Each step is a sub-tab within the Pastas Lab. Navigation is linear but users can jump to any step.

---

## Step 1: Station — Selection & Pre-fit Diagnostics

### Station Selection
- Existing StationPicker (search by BSS code, commune, department)
- StationDetailPanel shows instant metadata (from dim_piezo_stations, ~35ms)
- Time series preview loads async (piezo + precip + PET, ~2s)

### Pre-fit Diagnostics (new)

Runs automatically when a station is selected. Produces 6 indicators:

| Indicator | Computation | Green | Orange | Red |
|-----------|-------------|-------|--------|-----|
| Coverage | % days with measurement over exploitable period | >80% | 50-80% | <50% |
| Gaps | Largest continuous gap in days | <30d | 30-180d | >180d |
| Trend | Mann-Kendall test (p-value + slope) | p>0.05 | p<0.05, weak slope | p<0.01, strong slope |
| Breakpoints | Pettitt test for change-point | None | 1 detected | >1 detected |
| Seasonality | Autocorrelation at lag-12 months | >0.3 (strong) | 0.1-0.3 (weak) | <0.1 (absent) |
| Record length | Years of exploitable data | >15 yr | 5-15 yr | <5 yr |

### Actionable Recommendations

Each indicator with orange/red status produces a recommendation with an "Apply" button:

- "Trend detected (slope -2.1 cm/yr since 2008)" → [Add LinearTrend]
- "Major gap 2003-2005 (487 days)" → [Start after 2005-06]
- "Breakpoint detected at 2012-03" → [Add StepModel at 2012-03]
- "Weak seasonality — consider FlexModel" → [Switch to FlexModel]
- "Recommended calibration period: 2005-2020 (15 yr, 94% coverage)" → [Apply period]

In **Guided mode**: recommendations apply automatically. User sees a summary.
In **Expert mode**: recommendations are displayed, user clicks to apply.

### Backend

New endpoint: `POST /api/v1/pastas/diagnose`
- Input: `code_bss`
- Output: 6 indicators + recommended tmin/tmax + recommended modifications
- Implementation: `dashboard/utils/pastas/diagnostics_prefit.py`
  - Mann-Kendall: `scipy.stats.kendalltau` or `pymannkendall`
  - Pettitt: custom implementation (rank-based change-point test)
  - Coverage/gaps: pandas index analysis
  - Seasonality: `statsmodels.tsa.stattools.acf` at lag 12

---

## Step 2: Calibrate — Smart Fit Engine

### Mode Guided: Auto-fit

Single button "Auto-fit". The engine:

1. **Warm-up**: reserves first N years (default 2, configurable) as spin-up. Model is solved over warm-up but metrics exclude it.

2. **Two-pass solve** for each candidate config:
   - Pass 1: solve WITHOUT noise model → stable parameter estimates
   - Pass 2: solve WITH noise model, using Pass 1 params as `initial` values
   
3. **Grid search** over relevant combinations (filtered by diagnostic):
   - Recharge: Linear + BDLISA-recommended (2 options)
   - Response: Gamma + Exponential + BDLISA-recommended (2-3 options)
   - Noise: ArNoiseModel + none (2 options)
   - If trend detected: with/without LinearTrend (×2)
   - Typical: 8-16 combinations (not exhaustive 40+)

4. **STOWA screening** — 4 hard criteria per model:
   - EVP >= 70% (or NSE >= 0.5)
   - Residuals: Runs test p > 0.05
   - Response time: t95 < 50% of calibration period length
   - Gain: significantly different from zero (stderr-based)

5. **Ranking**: models passing STOWA ranked by AIC. Best selected.

6. **Progress**: SSE stream — "Testing Linear/Gamma/Ar (3/12)... EVP 72% PASS"

### Mode Expert: Manual Fit

All parameters exposed:
- Recharge type, response function, noise model, solver
- Warm-up period slider (0-5 years)
- Two-pass solve checkbox (on by default)
- Parameter bounds (pmin/pmax) editable per parameter
- tmin/tmax date pickers
- Cal/val split slider
- Temperature stress checkbox
- Diagnostic panel visible as context sidebar

### Backend

New endpoint: `POST /api/v1/pastas/auto-fit`
- Input: `code_bss`, optional overrides (warm_up_years, configs_to_test, stowa_thresholds)
- Output: SSE stream of progress + final ranked results
- Implementation: `dashboard/utils/pastas/auto_fit.py`

Modified: `dashboard/utils/pastas/fit_service.py`
- Add `warm_up_years` parameter
- Add `two_pass` parameter (bool, default True)
- Add `initial_params` parameter (dict, for seeding from Pass 1)

New: `dashboard/utils/pastas/stowa.py`
- `assess_stowa(model, tmin, tmax, cal_period_days) -> StowaResult`
- Returns pass/fail per criterion + values

### Execution Time

- 1 config (two-pass): ~4-10s
- Grid search 12 configs: ~60-120s
- SSE streaming for real-time progress

---

## Step 3: Results — STOWA Verdict & Diagnostics

### STOWA Verdict Banner

Full-width banner at top with 4 pass/fail indicators:

```
[✓ EVP 78.3%]  [✓ No autocorrelation]  [✓ t95 = 847d]  [✓ Gain significant]
```

Green = pass, red = fail. Overall verdict: "Model accepted" or "Model needs attention" with specific fix suggestions.

In Guided mode, if model fails STOWA, auto-suggest: "Try adding noise model" or "Reduce calibration period" with one-click relaunch.

### Result Sections

1. **Performance Metrics** — Train/Test side-by-side (NSE, KGE, EVP, RMSE, AIC, BIC). Existing, unchanged.
2. **Observed vs Simulated** — Plotly chart with train/test split. Existing, unchanged.
3. **Stress Decomposition** — Per-stress contributions. Existing, unchanged.
4. **Response Function** — Step/block response with t50/t95 annotated and STOWA threshold line.
5. **Residuals & Diagnostics** — Existing, enriched with traffic-light per test (DW, Ljung-Box, Jarque-Bera, Shapiro-Wilk, Runs test).
6. **Parameters** — Table with bounds-hit warnings. Existing, enhanced.
7. **Hydrological Signatures** — Radar chart. Existing, unchanged.
8. **AI Model Comparison** — New section (see below).

### AI Model Comparison

When a DL model (Darts) exists for the same station:

1. **Time series overlay**: Observed + Pastas simulated + AI forecast on common period (AI test set)
2. **Metrics comparison table**: NSE, KGE, RMSE for both on the same window
3. **Strengths summary**: auto-generated text explaining what each approach provides

Backend: `POST /api/v1/pastas/compare-ai`
- Input: `pastas_run_id`, `ai_model_id`
- Output: aligned series + metrics on common period

### Mode Behavior

- **Guided**: sections 5-7 collapsed by default. User sees verdict + metrics + main plot.
- **Expert**: all sections expanded.

---

## Step 4: Scenarios — What-if Simulations

### Existing (kept as-is)
- Modification types: pumping (synthetic/CSV), linear trend, scale stress
- Presets: drought (-30% precip), pumping (100 m³/d), climate trend (-1 cm/yr), PET increase (+20%)
- Results: baseline vs scenario, delta stats, contribution comparison

### New Presets
- **Zero recharge**: total infiltration stop over a period
- **DRIAS climate scenarios**: scaling factors from French climate projections (+1.5°C, +2°C, +4°C with associated precip/PET changes)

### Re-solve Option
After a scenario simulation, button "Re-calibrate with modified stresses" re-fits the model with the altered inputs. Tests if the model structure holds under extreme conditions.

### No Backend Changes Needed
Existing `/api/v1/pastas/simulate` endpoint handles all modification types. DRIAS presets are frontend-only (predefined scaling factors).

---

## Step 5: Gallery — Model Library & Ranking

### Station Grouping
Models grouped by station. Within each station, ranked by AIC among STOWA-passing models. Best model marked with star (★).

### STOWA Badges
Each model card shows mini STOWA badge: `✓✓✓✓` (all pass) or `✓✗✓✓` (one fail with label).

### Enriched Cards (existing structure, new fields)
- Model name, station code (clickable → station page), config tags
- Train/Test NSE + EVP
- STOWA badge
- AIC value
- "View results" / "Use for scenario" / Export / Delete actions

### No Major Structural Changes
Grid/list toggle, sorting, filtering, export — all stay.

---

## Frontend Structure

### Tab Layout

```
Pastas Lab (/pastas)
├── Station    (/pastas/station)   — pick + diagnostic
├── Calibrate  (/pastas/calibrate) — auto-fit or manual
├── Results    (/pastas/results)   — verdict + analysis
├── Scenarios  (/pastas/scenarios) — what-if
├── Gallery    (/pastas/gallery)   — all models
```

### Guided/Expert Toggle

Global toggle in the Pastas Lab header bar, persisted in localStorage. Affects all steps.

### State Flow

Station selection → stores `codeBss` in URL param.
Calibrate → produces `runId` → stored in URL param.
Results/Scenarios read `runId` from URL.
Gallery is independent (lists all models).

---

## Backend New Files

| File | Purpose |
|------|---------|
| `dashboard/utils/pastas/diagnostics_prefit.py` | Pre-fit data analysis (Mann-Kendall, Pettitt, coverage, gaps) |
| `dashboard/utils/pastas/auto_fit.py` | Grid search engine with two-pass solve and STOWA screening |
| `dashboard/utils/pastas/stowa.py` | STOWA 4-criteria assessment |
| `api/routers/pastas.py` (modified) | New endpoints: `/diagnose`, `/auto-fit` (SSE), `/compare-ai` |

## Backend Modified Files

| File | Change |
|------|--------|
| `dashboard/utils/pastas/fit_service.py` | Add `warm_up_years`, `two_pass`, `initial_params` parameters |
| `dashboard/utils/pastas/builder.py` | Support StepModel in initial build (not just scenarios) |
| `api/schemas/pastas.py` | New schemas: DiagnoseResponse, AutoFitProgress, StowaResult, CompareAIResponse |

## Frontend New Files

| File | Purpose |
|------|---------|
| `components/pastas/DiagnosticPanel.tsx` | 6-indicator pre-fit diagnostic with action buttons |
| `components/pastas/StowaVerdictBanner.tsx` | 4-criteria pass/fail banner |
| `components/pastas/AutoFitProgress.tsx` | SSE progress for grid search |
| `components/pastas/AIComparisonPanel.tsx` | Pastas vs AI overlay and metrics |
| `components/pastas/GuidedExpertToggle.tsx` | Mode toggle component |
| `pages/pastas/StationStep.tsx` | New step 1 page |
| `pages/pastas/CalibrateStep.tsx` | New step 2 page |
| `pages/pastas/ResultsStep.tsx` | New step 3 page (refactored from FitResultsPanel) |

## Frontend Modified Files

| File | Change |
|------|--------|
| `pages/pastas/PastasLayout.tsx` | New 5-tab layout + Guided/Expert toggle in header |
| `pages/pastas/GalleryPage.tsx` | Add STOWA badges + station grouping |
| `components/pastas/ModelTable.tsx` | Add STOWA badge, AIC column, station grouping |
| `hooks/usePastas.ts` | Add hooks: usePastasDiagnose, usePastasAutoFit, usePastasCompareAI |
| `lib/api.ts` | Add API methods: diagnose, autoFit, compareAI |
| `lib/types.ts` | Add types: DiagnoseResult, StowaResult, AutoFitProgress, CompareAIResult |

---

## Dependencies

### Python (new)
- `pymannkendall` — Mann-Kendall trend test (pip install)

### No new frontend dependencies
- All charts use existing Plotly
- SSE uses existing EventSource pattern

---

## Out of Scope (Future)

- RAMEAU integration (process-based modeling) — architecture supports it via modeling backend abstraction
- EmceeSolve / Bayesian uncertainty — requires long MCMC runs + UI for posterior visualization
- Batch processing (multi-station auto-fit) — requires job queue infrastructure
- Spatial parameter maps — requires map integration with model outputs
- ECMWF SEAS5 seasonal forecast integration
