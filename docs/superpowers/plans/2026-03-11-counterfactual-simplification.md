# Counterfactual Simplification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the complex multi-class per-month CF target UI with a quality gate + single-class ±1 transition buttons.

**Architecture:** Extract shared IPS constants/functions to `lib/ips.ts`, replace `CFConfigForm` with `CFTargetSelector`, add quality gate verdict logic to `CounterfactualPage`, enhance `IPSMonthlyGrid` with concordance indicators.

**Tech Stack:** React 18, TypeScript, TanStack Query, Plotly, Tailwind CSS, lucide-react icons.

**Spec:** `docs/superpowers/specs/2026-03-11-counterfactual-simplification-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `frontend/src/lib/ips.ts` | **Create** | Shared IPS constants, classifyZ, shiftClass, areAdjacent, computeMonthlyIps |
| `frontend/src/lib/api.ts` | **Modify** (line 249) | Add `target_ips_classes` to counterfactual.run() body type |
| `frontend/src/components/counterfactual/CFTargetSelector.tsx` | **Create** | Quality verdict display + ±1 buttons + advanced hyperparams + generate button |
| `frontend/src/components/counterfactual/IPSMonthlyGrid.tsx` | **Modify** | Add concordance indicator (✓/✗/⚠) per month |
| `frontend/src/pages/CounterfactualPage.tsx` | **Modify** | Wire quality gate, replace CFConfigForm, add generation counter, fix slider a11y |
| `frontend/src/components/counterfactual/CFConfigForm.tsx` | **Delete** | Replaced by CFTargetSelector |

---

## Chunk 1: Shared IPS Module + API Fix

### Task 1: Create `lib/ips.ts` — shared IPS constants and functions

**Files:**
- Create: `frontend/src/lib/ips.ts`

- [ ] **Step 1: Create `lib/ips.ts` with all IPS constants and functions**

```typescript
// frontend/src/lib/ips.ts

// Ordered from lowest to highest — index = ordinal position
export const IPS_CLASS_ORDER = [
  'very_low',
  'low',
  'moderately_low',
  'normal',
  'moderately_high',
  'high',
  'very_high',
] as const

export type IPSClass = (typeof IPS_CLASS_ORDER)[number]

export const IPS_LABELS: Record<string, string> = {
  very_low: 'Tres bas',
  low: 'Bas',
  moderately_low: 'Mod. bas',
  normal: 'Normal',
  moderately_high: 'Mod. haut',
  high: 'Haut',
  very_high: 'Tres haut',
}

export const IPS_COLORS: Record<string, string> = {
  very_low: '#d73027',
  low: '#fc8d59',
  moderately_low: '#fee08b',
  normal: '#ffffbf',
  moderately_high: '#d9ef8b',
  high: '#91cf60',
  very_high: '#1a9850',
}

// BRGM standard thresholds: Seguin (2014) RP-64147-FR
export const IPS_THRESHOLDS: [string, number, number][] = [
  ['very_low', -Infinity, -1.28],
  ['low', -1.28, -0.84],
  ['moderately_low', -0.84, -0.25],
  ['normal', -0.25, 0.25],
  ['moderately_high', 0.25, 0.84],
  ['high', 0.84, 1.28],
  ['very_high', 1.28, Infinity],
]

const MONTH_NAMES = ['', 'Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aou', 'Sep', 'Oct', 'Nov', 'Dec']

/** Classify a z-score into an IPS class */
export function classifyZ(z: number): string {
  for (const [cls, lo, hi] of IPS_THRESHOLDS) {
    if (z >= lo && z < hi) return cls
  }
  return 'normal'
}

/** Shift a class by delta steps (clamped at extremes). delta can be negative. */
export function shiftClass(cls: string, delta: number): string {
  const idx = IPS_CLASS_ORDER.indexOf(cls as IPSClass)
  if (idx === -1) return cls
  const newIdx = Math.max(0, Math.min(IPS_CLASS_ORDER.length - 1, idx + delta))
  return IPS_CLASS_ORDER[newIdx]
}

/** True if two classes are at most 1 step apart in the ordering */
export function areAdjacent(a: string, b: string): boolean {
  const idxA = IPS_CLASS_ORDER.indexOf(a as IPSClass)
  const idxB = IPS_CLASS_ORDER.indexOf(b as IPSClass)
  if (idxA === -1 || idxB === -1) return false
  return Math.abs(idxA - idxB) <= 1
}

/** Get the ordinal index of a class (0=very_low, 6=very_high). Returns -1 if unknown. */
export function classIndex(cls: string): number {
  return IPS_CLASS_ORDER.indexOf(cls as IPSClass)
}

/** Get the mode (most frequent value) of an array of strings */
export function mode(values: string[]): string | null {
  if (values.length === 0) return null
  const counts = new Map<string, number>()
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1)
  let best = values[0]
  let bestCount = 0
  for (const [v, c] of counts) {
    if (c > bestCount) { best = v; bestCount = c }
  }
  return best
}

export interface MonthIps {
  yearMonth: string    // "2024-01"
  monthNumber: number  // 1-12
  monthLabel: string   // "Jan 2024"
  cls: string          // IPS class key
  zScore: number       // raw z-score
  mean: number         // mean GWL for that month
}

/**
 * Classify daily values into per-year-month IPS classes.
 * Groups by year-month (not month number alone).
 * Returns map keyed by year-month string.
 */
export function computeMonthlyIps(
  dates: string[],
  values: number[],
  refStats: Record<string, [number, number]>,
): MonthIps[] {
  const groups = new Map<string, { vals: number[]; monthNumber: number; year: string }>()

  for (let i = 0; i < dates.length; i++) {
    const d = new Date(dates[i])
    const monthNumber = d.getMonth() + 1
    const year = String(d.getFullYear())
    const yearMonth = `${year}-${String(monthNumber).padStart(2, '0')}`
    if (!groups.has(yearMonth)) groups.set(yearMonth, { vals: [], monthNumber, year })
    if (!isNaN(values[i]) && values[i] !== null) groups.get(yearMonth)!.vals.push(values[i])
  }

  const result: MonthIps[] = []
  for (const [yearMonth, { vals, monthNumber, year }] of groups) {
    const stats = refStats[String(monthNumber)]
    if (!stats || vals.length === 0) continue
    const [mu, sigma] = stats
    if (sigma <= 0) continue
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length
    const zScore = (mean - mu) / sigma
    result.push({
      yearMonth,
      monthNumber,
      monthLabel: `${MONTH_NAMES[monthNumber]} ${year}`,
      cls: classifyZ(zScore),
      zScore,
      mean,
    })
  }

  return result
}

/**
 * Compute quality gate verdict from GT and pred IPS per month.
 * Returns per-month concordance and global verdict.
 */
export type ConcordanceStatus = 'exact' | 'adjacent' | 'divergent'
export type QualityVerdict = 'qualified' | 'partial' | 'not_qualified'

export interface MonthConcordance {
  yearMonth: string
  monthLabel: string
  gtClass: string
  predClass: string
  status: ConcordanceStatus
}

export function computeQualityGate(
  gtIps: MonthIps[],
  predIps: MonthIps[],
): { months: MonthConcordance[]; verdict: QualityVerdict } {
  const predMap = new Map(predIps.map((m) => [m.yearMonth, m]))
  const months: MonthConcordance[] = []

  for (const gt of gtIps) {
    const pred = predMap.get(gt.yearMonth)
    if (!pred) continue
    let status: ConcordanceStatus
    if (gt.cls === pred.cls) status = 'exact'
    else if (areAdjacent(gt.cls, pred.cls)) status = 'adjacent'
    else status = 'divergent'
    months.push({ yearMonth: gt.yearMonth, monthLabel: gt.monthLabel, gtClass: gt.cls, predClass: pred.cls, status })
  }

  const divergentCount = months.filter((m) => m.status === 'divergent').length
  const totalCount = months.length

  let verdict: QualityVerdict
  if (divergentCount === 0) verdict = 'qualified'
  else if (divergentCount < totalCount / 2) verdict = 'partial'
  else verdict = 'not_qualified'

  return { months, verdict }
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | grep -E 'ips\.ts|error' | head -20`
Expected: No errors from `ips.ts`

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/ips.ts
git commit -m "feat(cf): add shared IPS constants and quality gate functions"
```

---

### Task 2: Fix API type for `counterfactual.run()`

**Files:**
- Modify: `frontend/src/lib/api.ts` (line 249-266)

- [ ] **Step 1: Add `target_ips_classes` to the run body type**

In `frontend/src/lib/api.ts`, find the `counterfactual.run` body type and add the field:

```typescript
// Change the run method body type to include target_ips_classes
run: (body: {
  model_id: string
  method?: string
  target_ips_class?: string
  target_ips_classes?: Record<string, string>
  from_ips_class?: string
  to_ips_class?: string
  start_idx?: number
  modifications?: Record<string, number>
  lambda_prox?: number
  n_iter?: number
  lr?: number
  cc_rate?: number
  device?: string
  n_trials?: number
  seed?: number
  k_sigma?: number
  lambda_smooth?: number
}) => postJson<CounterfactualResult>('/counterfactual/run', body),
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | grep 'error' | head -10`
Expected: No new errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "fix(cf): add target_ips_classes to counterfactual.run() type"
```

---

## Chunk 2: CFTargetSelector Component

### Task 3: Create `CFTargetSelector` component

**Files:**
- Create: `frontend/src/components/counterfactual/CFTargetSelector.tsx`

This replaces `CFConfigForm`. It receives quality gate data and provides ±1 class transition buttons.

- [ ] **Step 1: Create `CFTargetSelector.tsx`**

```typescript
// frontend/src/components/counterfactual/CFTargetSelector.tsx
import { useState, useMemo } from 'react'
import { ChevronDown, ChevronRight, ChevronLeft, Play, Loader2 } from 'lucide-react'
import {
  IPS_CLASS_ORDER,
  IPS_LABELS,
  IPS_COLORS,
  shiftClass,
  classIndex,
  mode,
  type MonthIps,
  type QualityVerdict,
} from '@/lib/ips'

export interface CFTargetData {
  model_id: string
  start_idx: number
  target_ips_classes: Record<string, string>
  lambda_prox: number
  n_iter: number
  lr: number
  cc_rate: number
  k_sigma: number
  lambda_smooth: number
}

interface CFTargetSelectorProps {
  modelId: string
  startIdx: number
  gtIps: MonthIps[]
  verdict: QualityVerdict
  isForecastLoading: boolean
  onSubmit: (data: CFTargetData) => void
  isPending: boolean
}

export function CFTargetSelector({
  modelId,
  startIdx,
  gtIps,
  verdict,
  isForecastLoading,
  onSubmit,
  isPending,
}: CFTargetSelectorProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [delta, setDelta] = useState(0) // class shift: -N..+N

  // Hyperparams
  const [lambdaProx, setLambdaProx] = useState(0.1)
  const [nIter, setNIter] = useState(500)
  const [lr, setLr] = useState(0.02)
  const [kSigma, setKSigma] = useState(4)
  const [lambdaSmooth, setLambdaSmooth] = useState(0.1)

  // Current IPS: mode of GT classes
  const currentClasses = useMemo(() => gtIps.map((m) => m.cls), [gtIps])
  const currentMode = useMemo(() => mode(currentClasses), [currentClasses])
  const currentModeLabel = currentMode ? IPS_LABELS[currentMode] ?? currentMode : '—'

  // Minority classes (not the mode)
  const minorityInfo = useMemo(() => {
    if (!currentMode || currentClasses.length === 0) return null
    const others = currentClasses.filter((c) => c !== currentMode)
    if (others.length === 0) return null
    const unique = [...new Set(others)]
    return unique.map((c) => `${others.filter((o) => o === c).length} mois: ${IPS_LABELS[c] ?? c}`).join(', ')
  }, [currentClasses, currentMode])

  // Target classes after shift
  const targetClasses = useMemo(() => {
    const result: Record<string, string> = {}
    for (const m of gtIps) {
      result[m.yearMonth] = shiftClass(m.cls, delta)
    }
    return result
  }, [gtIps, delta])

  const targetMode = useMemo(() => {
    const vals = Object.values(targetClasses)
    return mode(vals)
  }, [targetClasses])
  const targetModeLabel = targetMode ? IPS_LABELS[targetMode] ?? targetMode : '—'

  // Button disable logic
  const allAtMin = currentClasses.length > 0 && currentClasses.every((c) => classIndex(shiftClass(c, delta)) === 0)
  const allAtMax = currentClasses.length > 0 && currentClasses.every((c) => classIndex(shiftClass(c, delta)) === IPS_CLASS_ORDER.length - 1)

  const canSubmit = verdict !== 'not_qualified' && !isPending && !isForecastLoading && !!modelId && delta !== 0

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!canSubmit) return
    // Convert yearMonth keys to month number keys for backend compatibility
    const targetByMonth: Record<string, string> = {}
    for (const m of gtIps) {
      targetByMonth[String(m.monthNumber)] = shiftClass(m.cls, delta)
    }
    onSubmit({
      model_id: modelId,
      start_idx: startIdx,
      target_ips_classes: targetByMonth,
      lambda_prox: lambdaProx,
      n_iter: nIter,
      lr,
      cc_rate: 0.07,
      k_sigma: kSigma,
      lambda_smooth: lambdaSmooth,
    })
  }

  if (isForecastLoading) {
    return (
      <div className="flex items-center gap-2 text-text-secondary text-sm py-4">
        <Loader2 className="w-4 h-4 animate-spin" />
        Verification en cours...
      </div>
    )
  }

  if (gtIps.length === 0) {
    return (
      <p className="text-[10px] text-text-secondary/40 italic py-2">
        Selectionnez une fenetre pour configurer la cible
      </p>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Current IPS display */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">IPS actuel de la fenetre</label>
        <div className="flex items-center gap-2">
          <span
            className="px-3 py-1.5 rounded-full text-sm font-medium"
            style={{
              backgroundColor: `${IPS_COLORS[currentMode ?? 'normal']}33`,
              color: IPS_COLORS[currentMode ?? 'normal'],
            }}
          >
            {currentModeLabel}
          </span>
          {minorityInfo && (
            <span className="text-[10px] text-text-secondary/60">({minorityInfo})</span>
          )}
        </div>
      </div>

      {/* ±1 transition buttons */}
      <div>
        <label className="block text-xs text-text-secondary mb-2">Transition de classe</label>
        <div className="flex items-center gap-3">
          <button
            type="button"
            disabled={allAtMin}
            onClick={() => setDelta((d) => d - 1)}
            className="flex items-center gap-1 px-3 py-2 rounded-lg border border-white/10 bg-bg-hover/30 hover:bg-bg-hover/60 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-sm text-text-primary"
          >
            <ChevronLeft className="w-4 h-4" />
            Baisser
          </button>

          <div className="flex-1 text-center">
            {delta === 0 ? (
              <span className="text-xs text-text-secondary/50">Aucun changement</span>
            ) : (
              <div>
                <span className="text-xs text-text-secondary">
                  {delta > 0 ? '+' : ''}{delta} classe{Math.abs(delta) > 1 ? 's' : ''}
                </span>
              </div>
            )}
          </div>

          <button
            type="button"
            disabled={allAtMax}
            onClick={() => setDelta((d) => d + 1)}
            className="flex items-center gap-1 px-3 py-2 rounded-lg border border-white/10 bg-bg-hover/30 hover:bg-bg-hover/60 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-sm text-text-primary"
          >
            Monter
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Reset */}
        {delta !== 0 && (
          <button
            type="button"
            onClick={() => setDelta(0)}
            className="mt-1 text-[10px] text-text-secondary/50 hover:text-text-secondary underline"
          >
            Reinitialiser
          </button>
        )}
      </div>

      {/* Target display */}
      {delta !== 0 && targetMode && (
        <div className="bg-bg-hover/20 rounded-lg p-3">
          <span className="text-xs text-text-secondary">Cible :</span>
          <span
            className="ml-2 px-2.5 py-1 rounded-full text-sm font-medium"
            style={{
              backgroundColor: `${IPS_COLORS[targetMode]}33`,
              color: IPS_COLORS[targetMode],
            }}
          >
            {targetModeLabel}
          </span>
        </div>
      )}

      {/* Warning for partial qualification */}
      {verdict === 'partial' && (
        <p className="text-[10px] text-amber-400/80 bg-amber-500/10 rounded px-2 py-1.5">
          Attention : le modele diverge de l'observe sur certains mois. Les resultats contrefactuels peuvent etre moins fiables.
        </p>
      )}

      {/* Info: dual method */}
      <p className="text-[10px] text-text-secondary/50 bg-bg-hover/20 rounded px-2 py-1.5">
        PhysCF (gradient continu) et CoMTE (substitution de features) seront lances en parallele.
      </p>

      {/* Advanced hyperparams */}
      <div className="bg-bg-hover/30 rounded-lg">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-3 py-2 flex items-center gap-2 text-xs text-text-secondary hover:text-text-primary transition-colors"
        >
          {showAdvanced ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          Hyperparametres avances
        </button>
        {showAdvanced && (
          <div className="px-3 pb-3 space-y-3">
            <p className="text-[10px] text-text-secondary/50 uppercase">PhysCF</p>
            <SliderParam label="lambda_prox" value={lambdaProx} min={0.001} max={2.0} step={0.001} onChange={setLambdaProx} />
            <SliderParam label="n_iter" value={nIter} min={50} max={2000} step={10} onChange={setNIter} integer />
            <SliderParam label="lr" value={lr} min={0.001} max={0.1} step={0.001} onChange={setLr} />
            <div className="border-t border-white/5 my-2" />
            <p className="text-[10px] text-text-secondary/50 uppercase">CoMTE</p>
            <SliderParam label="num_distractors" value={numDistractors} min={1} max={20} step={1} onChange={setNumDistractors} integer />
            <SliderParam label="tau" value={tau} min={0.1} max={1.0} step={0.1} onChange={setTau} />
          </div>
        )}
      </div>

      {/* Generate button */}
      <button
        type="submit"
        disabled={!canSubmit}
        className="w-full bg-accent-cyan text-white px-4 py-2.5 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium flex items-center justify-center gap-2"
      >
        <Play className="w-4 h-4" />
        {isPending ? 'Generation...' : delta === 0 ? 'Choisir une direction' : 'Generer le contrefactuel'}
      </button>
    </form>
  )
}

function SliderParam({ label, value, min, max, step, onChange, integer }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; integer?: boolean
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] text-text-secondary">{label}</span>
        <span className="text-[11px] text-text-primary font-mono">
          {integer ? Math.round(value) : value.toFixed(3)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-accent-cyan h-1"
      />
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | grep -E 'CFTargetSelector|error' | head -20`
Expected: No errors from `CFTargetSelector.tsx`

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/counterfactual/CFTargetSelector.tsx
git commit -m "feat(cf): add CFTargetSelector with ±1 class transition buttons"
```

---

## Chunk 3: Enhance IPSMonthlyGrid with Concordance

### Task 4: Add concordance indicators to `IPSMonthlyGrid`

**Files:**
- Modify: `frontend/src/components/counterfactual/IPSMonthlyGrid.tsx`

The grid already shows GT and Pred side by side. We add a concordance badge (✓ exact, ⚠ adjacent, ✗ divergent) and accept optional concordance data.

- [ ] **Step 1: Update `IPSMonthlyGrid` to accept and display concordance**

Add a new optional prop `concordance` to the component. When provided, each month row shows a status indicator.

In `frontend/src/components/counterfactual/IPSMonthlyGrid.tsx`, add these changes:

1. Add import at top:
```typescript
import type { MonthConcordance } from '@/lib/ips'
```

2. Add to `IPSMonthlyGridProps`:
```typescript
concordance?: MonthConcordance[]
```

3. Replace the hardcoded `IPS_CLASSES` array with import from `@/lib/ips`:
```typescript
import { IPS_THRESHOLDS, classifyZ } from '@/lib/ips'
```
And replace the local `IPS_CLASSES` and `classifyZScore` with the imported versions.

4. Inside each month card in the render, add after the GT/Pred columns div:
```typescript
{/* Concordance indicator */}
{concordanceMap && concordanceMap.has(m.key) && (() => {
  const c = concordanceMap.get(m.key)!
  const icon = c.status === 'exact' ? '✓' : c.status === 'adjacent' ? '≈' : '✗'
  const color = c.status === 'exact' ? 'text-emerald-400' : c.status === 'adjacent' ? 'text-amber-400' : 'text-red-400'
  return <span className={`text-xs font-bold ${color}`}>{icon}</span>
})()}
```

Where `concordanceMap` is a `useMemo` built from the concordance prop:
```typescript
const concordanceMap = useMemo(() => {
  if (!concordance) return null
  return new Map(concordance.map((c) => [c.yearMonth, c]))
}, [concordance])
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | grep -E 'IPSMonthlyGrid|error' | head -10`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/counterfactual/IPSMonthlyGrid.tsx
git commit -m "feat(cf): add concordance indicators to IPSMonthlyGrid"
```

---

## Chunk 4: Rewrite CounterfactualPage

### Task 5: Rewrite `CounterfactualPage` with quality gate and new target selector

**Files:**
- Modify: `frontend/src/pages/CounterfactualPage.tsx`

This is the main wiring task. Key changes:
1. Replace `computeMonthlyIps` (local, groups by month number) with imported version from `lib/ips.ts` (groups by year-month)
2. Replace `CFConfigForm` with `CFTargetSelector`
3. Add quality gate verdict computation via `computeQualityGate`
4. Add generation counter to prevent SSE race conditions
5. Add `onKeyUp` to slider for a11y
6. Remove all multi-class imports and types

- [ ] **Step 1: Update imports**

Replace:
```typescript
import { CFConfigForm } from '@/components/counterfactual/CFConfigForm'
import type { CFFormData } from '@/components/counterfactual/CFConfigForm'
```

With:
```typescript
import { CFTargetSelector } from '@/components/counterfactual/CFTargetSelector'
import type { CFTargetData } from '@/components/counterfactual/CFTargetSelector'
```

Add:
```typescript
import { computeMonthlyIps, computeQualityGate, type MonthIps, type QualityVerdict } from '@/lib/ips'
```

Remove the local `IPS_THRESHOLDS`, `classifyZ`, `computeMonthlyIps` (lines 18-56) since they're now in `lib/ips.ts`.

- [ ] **Step 2: Add generation counter ref**

After `const eventSourcesRef = useRef<EventSource[]>([])`, add:
```typescript
const generationRef = useRef(0)
```

- [ ] **Step 3: Replace `gtIpsPerMonth`/`predIpsPerMonth` with new types**

Replace the two `useMemo` blocks computing `gtIpsPerMonth` and `predIpsPerMonth` (lines 238-246) with:

```typescript
const gtIps = useMemo<MonthIps[]>(() => {
  if (!currentWindowData || Object.keys(refStatsForGrid).length === 0) return []
  return computeMonthlyIps(currentWindowData.dates, currentWindowData.values, refStatsForGrid)
}, [currentWindowData, refStatsForGrid])

const predIps = useMemo<MonthIps[]>(() => {
  if (!currentWindowData || !windowPredValues || Object.keys(refStatsForGrid).length === 0) return []
  return computeMonthlyIps(currentWindowData.dates, windowPredValues, refStatsForGrid)
}, [currentWindowData, windowPredValues, refStatsForGrid])
```

- [ ] **Step 4: Add quality gate computation**

After the `predIps` useMemo, add:
```typescript
const qualityGate = useMemo(() => {
  if (gtIps.length === 0 || predIps.length === 0) return null
  return computeQualityGate(gtIps, predIps)
}, [gtIps, predIps])

const verdict: QualityVerdict = qualityGate?.verdict ?? 'not_qualified'
```

- [ ] **Step 5: Remove `predictionMonths` useMemo**

Delete the `predictionMonths` useMemo block (lines 220-225) — it's no longer needed.

- [ ] **Step 6: Update `handleSubmit` to use `CFTargetData` and generation counter**

Replace the `handleSubmit` callback:

```typescript
const handleSubmit = useCallback(
  async (config: CFTargetData) => {
    // Increment generation to prevent stale SSE overwrites
    generationRef.current += 1
    const thisGeneration = generationRef.current

    // Close any existing streams
    eventSourcesRef.current.forEach((es) => es.close())
    eventSourcesRef.current = []
    setResults({})
    setStreaming({})
    setActiveTab('physcf')

    const baseBody = {
      model_id: config.model_id,
      target_ips_classes: config.target_ips_classes,
      start_idx: config.start_idx,
      lambda_prox: config.lambda_prox,
      n_iter: config.n_iter,
      lr: config.lr,
      cc_rate: config.cc_rate,
      k_sigma: config.k_sigma,
      lambda_smooth: config.lambda_smooth,
    }

    for (const method of METHODS) {
      try {
        const resp = await api.counterfactual.run({ ...baseBody, method })
        if (generationRef.current !== thisGeneration) return // stale
        if (resp.task_id) {
          if (resp.status === 'done' && resp.result) {
            setResults((prev) => ({ ...prev, [method]: resp }))
          } else {
            const es = startCFStream(resp.task_id, method, setResults, setStreaming, thisGeneration, generationRef)
            eventSourcesRef.current.push(es)
          }
        }
      } catch (err) {
        if (generationRef.current !== thisGeneration) return
        setResults((prev) => ({
          ...prev,
          [method]: { task_id: '', status: 'error', result: null, error: (err as Error).message },
        }))
      }
    }
  },
  [],
)
```

- [ ] **Step 7: Update `startCFStream` to accept generation counter**

Update the `startCFStream` function signature to accept generation tracking:

```typescript
function startCFStream(
  taskId: string,
  method: string,
  setResults: React.Dispatch<React.SetStateAction<Record<string, CounterfactualResult | null>>>,
  setStreaming: React.Dispatch<React.SetStateAction<Record<string, boolean>>>,
  generation: number,
  generationRef: React.MutableRefObject<number>,
): EventSource {
```

And in each event handler, add a guard before state updates:
```typescript
if (generationRef.current !== generation) { es.close(); return }
```

- [ ] **Step 8: Add `onKeyUp` to slider**

On the slider `<input>`, add `onKeyUp`:
```typescript
onKeyUp={() => setStartIdx(sliderDraft)}
```

- [ ] **Step 9: Update the quality gate banner in the test set section**

After the `IPSMonthlyGrid` inside the test set card, add a quality verdict banner:

```typescript
{qualityGate && (
  <div className={`rounded-lg p-3 flex items-center gap-2 text-sm ${
    verdict === 'qualified'
      ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400'
      : verdict === 'partial'
        ? 'bg-amber-500/10 border border-amber-500/20 text-amber-400'
        : 'bg-red-500/10 border border-red-500/20 text-red-400'
  }`}>
    {verdict === 'qualified' && 'Le modele reproduit fidelement l\'observe sur cette fenetre'}
    {verdict === 'partial' && 'Concordance partielle — certains mois divergent'}
    {verdict === 'not_qualified' && 'Le modele ne reproduit pas l\'observe — contrefactuelle non fiable'}
  </div>
)}
```

- [ ] **Step 10: Replace CFConfigForm with CFTargetSelector in the sidebar**

Replace:
```tsx
<CFConfigForm
  modelId={modelId}
  startIdx={startIdx}
  predictionMonths={predictionMonths}
  gtIpsPerMonth={gtIpsPerMonth}
  predIpsPerMonth={predIpsPerMonth}
  onSubmit={handleSubmit}
  isPending={isLoading}
/>
```

With:
```tsx
<CFTargetSelector
  modelId={modelId}
  startIdx={startIdx}
  gtIps={gtIps}
  verdict={verdict}
  isForecastLoading={forecastMutation.isPending}
  onSubmit={handleSubmit}
  isPending={isLoading}
/>
```

- [ ] **Step 11: Pass concordance to IPSMonthlyGrid**

Update the IPSMonthlyGrid in the test set section to pass concordance data:
```tsx
<IPSMonthlyGrid
  predDates={currentWindowData.dates}
  predValues={windowPredValues ?? currentWindowData.values}
  gtValues={currentWindowData.values}
  refStats={refStatsForGrid}
  ipsLabels={ipsBoundsData?.classes ?? {}}
  ipsColors={ipsBoundsData?.colors ?? {}}
  label="Classification IPS de la fenetre selectionnee"
  concordance={qualityGate?.months}
/>
```

- [ ] **Step 12: Verify TypeScript compiles**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | head -30`
Expected: No errors

- [ ] **Step 13: Commit**

```bash
git add frontend/src/pages/CounterfactualPage.tsx
git commit -m "feat(cf): rewrite CounterfactualPage with quality gate and ±1 target selector"
```

---

## Chunk 5: Cleanup

### Task 6: Delete `CFConfigForm.tsx`

**Files:**
- Delete: `frontend/src/components/counterfactual/CFConfigForm.tsx`

- [ ] **Step 1: Verify no other files import CFConfigForm**

Run: `grep -r 'CFConfigForm' frontend/src/ --include='*.ts' --include='*.tsx'`
Expected: No results (CounterfactualPage was already updated in Task 5)

- [ ] **Step 2: Delete the file**

```bash
rm frontend/src/components/counterfactual/CFConfigForm.tsx
```

- [ ] **Step 3: Verify TypeScript compiles with no errors**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add -u frontend/src/components/counterfactual/CFConfigForm.tsx
git commit -m "chore(cf): delete CFConfigForm replaced by CFTargetSelector"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run full TypeScript check**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty`
Expected: Clean compilation

- [ ] **Step 2: Run dev server to verify rendering**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx vite build 2>&1 | tail -10`
Expected: Build succeeds with no errors

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A frontend/src/
git commit -m "fix(cf): address final compilation issues from CF simplification"
```
