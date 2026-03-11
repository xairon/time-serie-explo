# Refonte Page Contrefactuelle — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Réécrire la page contrefactuelle React pour reproduire le flow Streamlit : sélection de fenêtre sur le test set, bandes IPS par mois, classification mensuelle, interprétation actionnable.

**Architecture:** Le backend a déjà toute la logique (PhysCF/Optuna/CoMTE, IPS, perturbation). Il manque : (1) un paramètre `start_idx` dans le schéma CF pour choisir la fenêtre, (2) un endpoint `/ips-bounds` pour les bandes mensuelles, (3) une refonte complète du frontend avec slider de fenêtre, bandes IPS, et interprétation.

**Tech Stack:** FastAPI (backend), React + TanStack Query + Plotly.js (frontend), Tailwind CSS (styling)

---

## Task 1: Backend — Ajouter `start_idx` au schéma CF + endpoint IPS bounds

**Files:**
- Modify: `api/schemas/counterfactual.py`
- Modify: `api/routers/counterfactual.py`

**Step 1: Add `start_idx` to CFGenerateRequest schema**

In `api/schemas/counterfactual.py`, add to `CFGenerateRequest`:
```python
    start_idx: Optional[int] = None  # Position in test set (None = auto middle)
```

**Step 2: Use `start_idx` from request in `_run_cf_thread()`**

In `api/routers/counterfactual.py`, replace the hardcoded window selection (~line 146):
```python
# Before:
start_idx = min(valid_end // 2, valid_end)

# After:
if req.start_idx is not None and 0 <= req.start_idx <= valid_end:
    start_idx = req.start_idx
else:
    start_idx = min(valid_end // 2, valid_end)
```

**Step 3: Add GET `/ips-bounds` endpoint**

New endpoint that returns per-month IPS class bounds for a date range. Add to `api/routers/counterfactual.py`:

```python
@router.get("/ips-bounds")
async def ips_bounds(model_id: str, window: int = 1):
    """Return monthly IPS class bounds (m NGF) for the test set date range."""
    from dashboard.utils.model_registry import ModelRegistry
    from dashboard.utils.counterfactual.ips import (
        compute_ips_reference_n, compute_monthly_ips_bounds,
        IPS_CLASSES, IPS_LABELS, IPS_COLORS,
    )
    import pandas as pd

    registry = ModelRegistry(checkpoints_dir=Path(settings.checkpoints_dir))
    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Load test data for date range
    test_df = registry.load_data(entry, "test")
    if test_df is None:
        raise HTTPException(status_code=404, detail="Test data not found")

    # Get IPS reference (try cached, fallback compute)
    ref_response = await ips_reference(model_id=model_id, window=window)
    ref_stats_raw = ref_response.get("ref_stats", {})
    ref_stats = {int(k): tuple(v) for k, v in ref_stats_raw.items()}

    # Compute bounds
    bounds_df = compute_monthly_ips_bounds(test_df.index, ref_stats)
    if bounds_df.empty:
        return {"bounds": [], "classes": IPS_LABELS, "colors": IPS_COLORS}

    # Serialize
    rows = []
    for _, row in bounds_df.iterrows():
        r = {
            "month_start": row["month_start"].isoformat(),
            "month_end": row["month_end"].isoformat(),
            "month": int(row["month"]),
            "mu": float(row["mu"]),
            "sigma": float(row["sigma"]),
        }
        for cls_name in IPS_CLASSES:
            r[f"{cls_name}_lower"] = float(row[f"{cls_name}_lower"])
            r[f"{cls_name}_upper"] = float(row[f"{cls_name}_upper"])
        rows.append(r)

    return {
        "bounds": rows,
        "classes": IPS_LABELS,
        "colors": IPS_COLORS,
    }
```

**Step 4: Verify**

```bash
docker compose up -d --build backend
# Test start_idx
curl -s -X POST localhost:8000/api/v1/counterfactual/run \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"...","start_idx":50}' | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])"
# Test ips-bounds
curl -s "localhost:8000/api/v1/counterfactual/ips-bounds?model_id=...&window=1" | python3 -c "import sys,json;d=json.load(sys.stdin);print(len(d['bounds']),'months')"
```

**Step 5: Commit**
```bash
git add api/schemas/counterfactual.py api/routers/counterfactual.py
git commit -m "feat(cf): add start_idx param + /ips-bounds endpoint"
```

---

## Task 2: Frontend — Types, API client, hooks

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/hooks/useCounterfactual.ts`

**Step 1: Add new types**

In `frontend/src/lib/types.ts`, add:

```typescript
// IPS constants
export interface IPSBoundsRow {
  month_start: string
  month_end: string
  month: number
  mu: number
  sigma: number
  [key: string]: unknown // {cls}_lower, {cls}_upper
}

export interface IPSBoundsResponse {
  bounds: IPSBoundsRow[]
  classes: Record<string, string>  // key -> label FR
  colors: Record<string, string>   // key -> hex color
}
```

Update `CounterfactualResult.result` to be more precise:
```typescript
export interface CFInnerResult {
  method: string
  original: number[]
  counterfactual: number[]
  dates: string[]
  theta: Record<string, number>
  metrics: Record<string, number | boolean | string>
  convergence?: number[]
  best_trial?: Record<string, unknown>
}
```

**Step 2: Add API methods**

In `frontend/src/lib/api.ts`, add to `counterfactual` section:

```typescript
  counterfactual: {
    // existing: run, stream, ipsReference
    ipsBounds: (modelId: string, window: number = 1) =>
      fetchJson<IPSBoundsResponse>(
        `/counterfactual/ips-bounds?model_id=${modelId}&window=${window}`
      ),
  },
```

**Step 3: Add hooks**

In `frontend/src/hooks/useCounterfactual.ts`, add:

```typescript
export function useIPSBounds(modelId: string | null, window: number = 1) {
  return useQuery({
    queryKey: ['ips-bounds', modelId, window],
    queryFn: () => api.counterfactual.ipsBounds(modelId!, window),
    enabled: !!modelId,
  })
}
```

**Step 4: Commit**
```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/src/hooks/useCounterfactual.ts
git commit -m "feat(cf): add IPS bounds types, API client, hooks"
```

---

## Task 3: Frontend — Rewrite CounterfactualPage layout + window slider

**Files:**
- Rewrite: `frontend/src/pages/CounterfactualPage.tsx`

**Goal:** New layout with 3 sections:
1. **Top**: Test set overview with slider (reuse TestSetOverview pattern from ForecastingPage)
2. **Left sidebar**: Simplified config (model, window position, IPS transition, method, generate button)
3. **Center**: Results (overlay plot with IPS bands, monthly classification, interpretation, details accordions)

**Step 1: Rewrite the page**

The page should:

1. Use `useModels()` + model selector at top
2. Fetch `testInfo` with `api.models.testInfo(modelId)` when model selected
3. Show **test set overview** chart with slider (dates, values, window highlight)
4. Below: 2-column layout — left config sidebar (narrow), right results area (wide)
5. Config sidebar contains:
   - Window position slider (min=valid_start_idx, max=valid_end_idx)
   - Caption: "Contexte: {date} → {date} ({L}j) | Prédiction: {date} → {date} ({H}j)"
   - IPS window selector: IPS-1/3/6/12 dropdown
   - IPS transition: "De [dropdown] → Vers [dropdown]" with colored badges
   - Method: PhysCF (default), collapsible advanced (Optuna, CoMTE)
   - Hyperparams: collapsible section (lambda_prox, n_iter, lr)
   - Generate button
6. Results area: placeholder until generation is done

**Key state:**
```typescript
const [modelId, setModelId] = useState<string | null>(null)
const [startIdx, setStartIdx] = useState(0)
const [ipsWindow, setIpsWindow] = useState(1)
const [ipsFrom, setIpsFrom] = useState('normal')
const [ipsTo, setIpsTo] = useState('moderately_low')
const [method, setMethod] = useState<'physcf' | 'optuna' | 'comte'>('physcf')
```

**Step 2: Commit**
```bash
git add frontend/src/pages/CounterfactualPage.tsx
git commit -m "feat(cf): rewrite page with window slider + IPS config"
```

---

## Task 4: Frontend — IPS bands overlay on test set chart

**Files:**
- Create: `frontend/src/components/counterfactual/IPSBandsChart.tsx`

**Goal:** A Plotly chart showing the full test set with:
- Ground truth line (blue)
- Sliding window rectangles (context=blue, prediction=yellow) — same as TestSetOverview
- **IPS class bands** as stacked semi-transparent rectangles per calendar month, using `IPSBoundsResponse`

**Step 1: Create the component**

```typescript
interface IPSBandsChartProps {
  testDates: string[]
  testValues: (number | null)[]
  ipsBounds: IPSBoundsRow[]
  ipsColors: Record<string, string>
  ipsLabels: Record<string, string>
  contextStart: string    // date
  contextEnd: string      // date
  predStart: string       // date
  predEnd: string         // date
  // Optional CF overlay
  cfDates?: string[]
  cfOriginal?: number[]
  cfCounterfactual?: number[]
}
```

The component renders:
1. For each month in `ipsBounds`, draw 7 stacked `shape` rectangles (one per IPS class) as background fills
2. Ground truth line
3. Context rectangle (vrect blue)
4. Prediction rectangle (vrect yellow)
5. If CF results exist, overlay original (gray) + counterfactual (cyan dashed)

**Step 2: Commit**
```bash
git add frontend/src/components/counterfactual/IPSBandsChart.tsx
git commit -m "feat(cf): IPS bands chart with monthly class boundaries"
```

---

## Task 5: Frontend — Monthly IPS classification grid

**Files:**
- Create: `frontend/src/components/counterfactual/IPSMonthlyGrid.tsx`

**Goal:** A grid of cards showing per-month IPS classification for the prediction window.

For each month in the prediction horizon, show:
- Month label (e.g., "Jan 2021")
- GT class (colored badge)
- Pred/CF class (colored badge)
- z-score value

This requires computing IPS classification client-side from the raw values + ref_stats. Since the prediction window is small (H ≈ 90 days = ~3 months), this is just a few cards.

```typescript
interface IPSMonthlyGridProps {
  predDates: string[]
  predValues: number[]        // model prediction or CF values
  gtValues: number[]          // ground truth
  refStats: Record<string, [number, number]>  // month -> [mu, sigma]
  ipsLabels: Record<string, string>
  ipsColors: Record<string, string>
  label?: string              // "Prediction" or "Contrefactuel"
}
```

Compute monthly means, z-scores, classify. Display as horizontal card row.

**Step 2: Commit**
```bash
git add frontend/src/components/counterfactual/IPSMonthlyGrid.tsx
git commit -m "feat(cf): monthly IPS classification grid"
```

---

## Task 6: Frontend — CF Result panel with interpretation

**Files:**
- Rewrite: `frontend/src/components/counterfactual/CFResultView.tsx`

**Goal:** Replace current minimal result view with:

1. **Status banner**: green "Convergé" / orange "Partiel" / red "Échoué" + wall time
2. **Indicateurs de plausibilité**: 4 cards (Validité %, Proximité, Convergence, CC)
3. **Interprétation en langage naturel**: Generate French text from theta:
   - "Réduction des précipitations hivernales de -30% (s_P_DJF=0.70)"
   - "Hausse de température de +1.5°C"
   - etc.
4. **Accordéon "Paramètres optimisés"**: Radar chart + table with FR labels
5. **Accordéon "Convergence"**: Loss curve (reuse ConvergencePlot)

**Theta label mapping:**
```typescript
const THETA_LABELS: Record<string, string> = {
  s_P_DJF: 'Précipitations hiver (DJF)',
  s_P_MAM: 'Précipitations printemps (MAM)',
  s_P_JJA: 'Précipitations été (JJA)',
  s_P_SON: 'Précipitations automne (SON)',
  delta_T: 'Température (°C)',
  delta_etp: 'Evapotranspiration résiduelle',
  delta_s: 'Décalage temporel (jours)',
}
```

**Interpretation generation** (client-side):
```typescript
function interpretTheta(theta: Record<string, number>): string[] {
  const lines: string[] = []
  // Precipitation by season
  for (const [key, label] of [['s_P_DJF','hiver'],['s_P_MAM','printemps'],['s_P_JJA','été'],['s_P_SON','automne']]) {
    const v = theta[key]
    if (v != null && Math.abs(v - 1) > 0.05) {
      const pct = Math.round((v - 1) * 100)
      lines.push(`Précipitations ${label} : ${pct > 0 ? '+' : ''}${pct}%`)
    }
  }
  // Temperature
  if (theta.delta_T != null && Math.abs(theta.delta_T) > 0.1) {
    lines.push(`Température : ${theta.delta_T > 0 ? '+' : ''}${theta.delta_T.toFixed(1)}°C`)
  }
  // Shift
  if (theta.delta_s != null && Math.abs(theta.delta_s) > 1) {
    lines.push(`Décalage temporel : ${theta.delta_s > 0 ? '+' : ''}${Math.round(theta.delta_s)} jours`)
  }
  return lines
}
```

**Step 2: Commit**
```bash
git add frontend/src/components/counterfactual/CFResultView.tsx
git commit -m "feat(cf): result panel with plausibility indicators + interpretation"
```

---

## Task 7: Frontend — Simplify CFConfigForm

**Files:**
- Rewrite: `frontend/src/components/counterfactual/CFConfigForm.tsx`

**Goal:** Simplify the form. Remove hardcoded covariates (the model's actual covariates come from the backend). The form should contain:

1. **IPS Window**: dropdown IPS-1/3/6/12
2. **IPS Transition**: two dropdowns with colored badges "De → Vers"
3. **Method**: PhysCF (default radio), Optuna, CoMTE
4. **Hyperparams**: collapsible, pre-filled defaults per method
5. **Generate button**

Remove: dataset_id selector (not used by backend), hardcoded 5-covariate sliders, device selector, seed.

The form receives `modelId` and `startIdx` as props (set by the parent page).

```typescript
interface CFConfigFormProps {
  modelId: string
  startIdx: number
  ipsWindow: number
  onIpsWindowChange: (w: number) => void
  onSubmit: (config: CFFormData) => void
  isPending: boolean
}

interface CFFormData {
  model_id: string
  start_idx: number
  method: string
  target_ips_class: string
  from_ips_class: string
  to_ips_class: string
  lambda_prox: number
  n_iter: number
  lr: number
  cc_rate: number
  n_trials: number
  k_sigma: number
  lambda_smooth: number
}
```

**Step 2: Commit**
```bash
git add frontend/src/components/counterfactual/CFConfigForm.tsx
git commit -m "feat(cf): simplified config form with IPS transition"
```

---

## Task 8: Integration — Wire everything together in CounterfactualPage

**Files:**
- Modify: `frontend/src/pages/CounterfactualPage.tsx`

**Goal:** Connect all components:

1. Model selection → fetch testInfo + ipsBounds
2. Slider → update startIdx → recompute dates shown in caption
3. IPSBandsChart shows test set + IPS bands + window
4. CFConfigForm → submit → SSE stream → update result state
5. On result: overlay CF on chart + show IPSMonthlyGrid + CFResultView
6. Accordions for details (radar, convergence)

**SSE handling** (keep existing pattern from current page):
```typescript
const es = new EventSource(`${API_BASE}/counterfactual/${taskId}/stream`)
es.addEventListener('progress', (e) => { ... })
es.addEventListener('done', (e) => { ... })
es.onerror = () => { ... }
```

**Step 2: Rebuild + test**
```bash
docker compose up -d --build frontend
```

**Step 3: Commit**
```bash
git add frontend/src/pages/CounterfactualPage.tsx
git commit -m "feat(cf): wire all components together"
```

---

## Task 9: Cleanup — Remove unused components

**Files:**
- Delete: `frontend/src/components/counterfactual/PastasPanel.tsx` (stub, not used)
- Review: `frontend/src/components/charts/CFOverlayPlot.tsx` (replaced by IPSBandsChart)
- Keep: `frontend/src/components/charts/RadarPlot.tsx` (used in CFResultView)
- Keep: `frontend/src/components/counterfactual/ConvergencePlot.tsx` (used in CFResultView)

**Step 1: Remove dead files, verify no imports broken**
```bash
grep -r "PastasPanel" frontend/src/
grep -r "CFOverlayPlot" frontend/src/
```

**Step 2: Commit**
```bash
git add -A frontend/src/components/
git commit -m "chore(cf): remove unused stub components"
```

---

## Task 10: Final verification

**Checklist:**
- [ ] Model selection loads test set + IPS bounds
- [ ] Slider moves window, caption updates with dates
- [ ] IPS bands render as colored backgrounds per month
- [ ] IPS transition dropdowns show 7 classes with FR labels + colors
- [ ] Generate button sends request with correct start_idx
- [ ] SSE streaming shows progress
- [ ] Result displays: overlay on chart, monthly IPS grid, plausibility indicators
- [ ] Theta interpretation in plain French
- [ ] Convergence accordion works
- [ ] Radar chart shows theta parameters
- [ ] No .toFixed() crashes on non-numeric values
- [ ] Page works with existing model `66347362177b4aefbf33aba3e0c1c6e7`

```bash
docker compose up -d --build backend frontend
# Manual test in browser at :49513/counterfactual
```
