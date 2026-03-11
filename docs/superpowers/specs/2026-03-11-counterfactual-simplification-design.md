# Counterfactual Simplification — Quality Gate + Single-Class Transition

**Date**: 2026-03-11
**Status**: Approved
**Scope**: Frontend `CounterfactualPage`, `CFConfigForm`, `IPSMonthlyGrid`; minor backend adjustments

## Problem

The current multi-class per-month CF target configuration is:
- Too complex for users (7 IPS classes × N months = combinatorial explosion)
- Physically contradictory (asking for "very_high" in January and "very_low" in March with only 7 global physics parameters is over-constrained)
- Missing a prerequisite check: running CF on a window where the model predictions poorly match GT produces meaningless results (measures sensitivity of a bad model, not physical reality)

## Design

### Flow

```
1. Select model
2. Move slider to select test window
3. [AUTOMATIC] Quality check runs when slider stops
   - Forecast model on window
   - Classify IPS per month for GT and predictions
   - Check concordance (pred_class == gt_class for all months)
   - Display result: green/red badges per month + global verdict
4. [IF QUALIFIED] Show single-class target selector
   - Display current IPS class (uniform = GT class since concordant)
   - +1 / -1 class transition buttons
   - One step at a time (no multi-class jumps)
5. Click "Generer" → PhysCF + CoMTE in parallel (SSE)
6. Results in tabs (unchanged)
7. Pastas validation button (unchanged)
```

### Quality Gate (automatic)

**Trigger**: `startIdx` changes (slider mouseUp)

**Computation** (already exists, just needs verdict logic):
1. `forecastMutation` fetches model predictions for window
2. `computeMonthlyIps()` classifies GT and predictions
3. New: compare `gtIpsPerMonth` vs `predIpsPerMonth` for all months

**Verdict** (with adjacent-class tolerance):
- **Concordant**: `pred_class == gt_class` OR classes are adjacent (±1 step). Example: pred=normal, gt=moderately_high → concordant (one step apart)
- **Qualified** (all months concordant): green banner, CF controls enabled
- **Partially qualified** (some months diverge by >1 class): amber banner listing divergent months, CF button still enabled but with warning
- **Not qualified** (majority of months diverge by >1 class): red banner, CF button disabled

Rationale: strict class equality is too sensitive near thresholds (z=0.24 normal vs z=0.26 moderately_high would fail despite 0.02 difference). Adjacent tolerance is the minimum to avoid false rejections.

**Loading state**: While forecast is in-flight, show a spinner with "Verification en cours..." in the quality gate area. Quality gate controls are disabled until the forecast resolves.

**Display** in the IPS monthly grid (enhanced):
- Add a concordance indicator (checkmark / cross / warning) per month
- Show both "Observe" and "Modele" classes side by side (already done)

### Target Selection (replaces CFConfigForm)

New component `CFTargetSelector` replaces `CFConfigForm`:

```
┌─────────────────────────────────────────┐
│ IPS actuel : [Normal]  (z = 0.12)       │
│                                         │
│  [← Baisser]          [Monter →]        │
│                                         │
│ Cible : Moderement haut                 │
│ (tous les mois de la fenetre)           │
│                                         │
│ ▸ Hyperparametres avances               │
│   PhysCF: lambda_prox, n_iter, lr       │
│   CoMTE: num_distractors, tau            │
│                                         │
│ [▶ Generer le contrefactuel]            │
└─────────────────────────────────────────┘
```

- Current class display = mode (most frequent) of GT IPS classes across months
- If months are heterogeneous (e.g., [normal, normal, moderately_high]), display shows mode + range: "Normal (1 mois: Mod. haut)"
- +1/-1 shift is applied **per-month**: each month's class is shifted individually. Backend receives per-month targets via `target_ips_classes`
- Boundary clipping: months already at `very_high` stay at `very_high` when +1 is pressed (per-month clamp). Button is disabled only when ALL months are at the extreme
- One click = one class step. Multiple clicks allowed (normal → high = click +1 twice)
- Display after shift shows the shifted mode + range if heterogeneous

### Shared IPS Constants

Create `frontend/src/lib/ips.ts` with:
- `IPS_CLASS_ORDER: string[]` — ordered list of 7 classes (very_low → very_high)
- `IPS_LABELS: Record<string, string>` — French display names
- `IPS_COLORS: Record<string, string>` — hex colors per class
- `IPS_THRESHOLDS: [string, number, number][]` — z-score bounds
- `classifyZ(z: number): string` — classify a z-score
- `shiftClass(cls: string, delta: number): string` — shift class by ±N steps, clamped at extremes
- `areAdjacent(a: string, b: string): boolean` — true if classes are ≤1 step apart

This eliminates the current duplication across `CounterfactualPage.tsx`, `IPSMonthlyGrid.tsx`, and `CFConfigForm.tsx`.

### Frontend API Type Fix

The `api.counterfactual.run()` body type in `frontend/src/lib/api.ts` must include `target_ips_classes?: Record<string, string>` in its typed parameter (currently only has `target_ips_class` singular). The field is already accepted by the backend Pydantic schema.

### Backend Changes

Minimal. The `target_ips_classes: Record<string, string>` field already supports per-month targeting. The frontend computes the shifted classes per month and sends them.

No new endpoints needed. No schema changes needed.

### Slider Accessibility

Add `onKeyUp={() => setStartIdx(sliderDraft)}` alongside `onMouseUp`/`onTouchEnd` so keyboard navigation triggers the quality gate.

### SSE Race Condition Prevention

Add a generation counter (`generationRef`) incremented on each submit. SSE event handlers check if their generation matches the current one before updating state, preventing stale results from overwriting fresh ones on rapid re-fires.

### Components

| Component | Action |
|-----------|--------|
| `CFConfigForm.tsx` | **Delete** — replaced by `CFTargetSelector.tsx` |
| `CFTargetSelector.tsx` | **New** — quality badge + current class + ±1 buttons + advanced hyperparams |
| `CounterfactualPage.tsx` | **Simplify** — add quality gate verdict, wire new target selector, remove multi-class logic |
| `IPSMonthlyGrid.tsx` | **Enhance** — add concordance column (✓/✗) per month |
| `lib/ips.ts` | **New** — shared IPS constants, `classifyZ`, `shiftClass`, `areAdjacent` |
| `lib/api.ts` | **Fix** — add `target_ips_classes` to `counterfactual.run()` body type |
| `useCounterfactual.ts` | No change (already has `target_ips_classes`) |
| Backend | No change |

### What Disappears

- Per-month IPS class selectors
- Preset modes (prediction/observe/custom)
- "Tous → classe" dropdown
- Optuna method (already removed from UI)

### What Stays

- PhysCF + CoMTE parallel execution
- SSE streaming with progress
- Tabbed results view (CFResultView)
- Pastas validation
- Advanced hyperparameters (collapsible)
- JSON export

## Scientific Validation

All scientific foundations have been audited and verified:

| Element | Status | Reference |
|---------|--------|-----------|
| IPS z-score thresholds (7 classes) | Standard BRGM | Seguin (2014) RP-64147-FR, Seguin (2016) RP-67249-FR |
| Clausius-Clapeyron rate (cc=0.07) | Textbook value | Standard atmospheric physics, ~7%/K |
| PhysCF 7 parameters | Physically interpretable | 4 seasonal P, 1 T, 1 ETP residual, 1 temporal shift |
| CoMTE C=3 binary mask | Ates et al. 2021 discrete feature swapping | No CC coupling, distractor-based |
| Pastas Gamma + FlexModel | Standard choices | Collenteur et al. (2019), Groundwater |
| Pastas gamma=1.5 threshold | Reasonable heuristic | No published standard; already configurable |
| Quality gate (IPS concordance) | Novel but sound | Pre-check ensures CF operates on a faithful model window |

## Hyperparameter Defaults

### PhysCF
- `lambda_prox = 0.1` — proximity regularization weight
- `n_iter = 500` — Adam optimizer iterations
- `lr = 0.02` — learning rate
- `cc_rate = 0.07` — Clausius-Clapeyron coupling (fixed, not exposed)

### CoMTE
- `num_distractors = 5` — number of nearest distractors (k)
- `tau = 0.5` — in-band fraction threshold

Note: `n_iter` and `lr` are PhysCF-only controls. CoMTE uses discrete combinatorial search (no gradient optimization), so it only needs `num_distractors` and `tau`.

### Month Grouping

`computeMonthlyIps()` must group by **year-month** (e.g., "2024-01"), not by month number alone. This matches the `IPSMonthlyGrid` grouping and prevents averaging across different years. The z-score for each year-month uses the reference stats for that calendar month (1-12).

### Pastas
- `gamma = 1.5` — acceptance threshold ratio
