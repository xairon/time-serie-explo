# Pastas CSV Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-section CSV export buttons for Pastas fit results (observed/simulated/residuals, contributions) and scenario outputs (baseline/scenario/delta, contributions).

**Architecture:** Frontend-only. A shared `downloadCsv()` utility builds CSV from column arrays and triggers browser download. A reusable `ExportCsvButton` component wraps this as an icon button placed in section headers. Four integration points: two in FitResultsPanel (via UnifiedAnalysisChart section), two in ScenarioResultsPanel.

**Tech Stack:** React, lucide-react (Download icon), TypeScript, Blob API

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `frontend/src/lib/csv-export.ts` | **Create** | `CsvColumn` type + `downloadCsv()` utility |
| `frontend/src/components/pastas/ExportCsvButton.tsx` | **Create** | Reusable icon button component |
| `frontend/src/components/pastas/FitResultsPanel.tsx` | **Modify** | Add 2 export buttons in Model Analysis section header |
| `frontend/src/components/pastas/ScenarioResultsPanel.tsx` | **Modify** | Add `codeBss` prop + 2 export buttons in section headers |
| `frontend/src/pages/pastas/ScenariosPage.tsx` | **Modify** | Pass `codeBss` prop to ScenarioResultsPanel |
| `frontend/src/components/pastas/ScenarioWorkflow.tsx` | **Modify** | Pass `codeBss` prop to ScenarioResultsPanel |

---

### Task 1: CSV Download Utility

**Files:**
- Create: `frontend/src/lib/csv-export.ts`

- [ ] **Step 1: Create the csv-export utility**

```ts
// frontend/src/lib/csv-export.ts

export interface CsvColumn {
  header: string
  values: (string | number | null)[]
}

function escapeCell(value: string | number | null): string {
  if (value === null || value === undefined) return ''
  const str = String(value)
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return '"' + str.replace(/"/g, '""') + '"'
  }
  return str
}

export function downloadCsv(filename: string, columns: CsvColumn[]): void {
  if (columns.length === 0) return

  const headers = columns.map(c => escapeCell(c.header)).join(',')
  const rowCount = Math.max(...columns.map(c => c.values.length))
  const rows: string[] = [headers]

  for (let i = 0; i < rowCount; i++) {
    rows.push(columns.map(c => escapeCell(c.values[i] ?? null)).join(','))
  }

  const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors related to csv-export.ts

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/csv-export.ts
git commit -m "feat(pastas): add CSV download utility"
```

---

### Task 2: ExportCsvButton Component

**Files:**
- Create: `frontend/src/components/pastas/ExportCsvButton.tsx`

- [ ] **Step 1: Create the button component**

```tsx
// frontend/src/components/pastas/ExportCsvButton.tsx
import { Download } from 'lucide-react'
import { downloadCsv, type CsvColumn } from '@/lib/csv-export'

interface Props {
  filename: string
  columns: CsvColumn[]
  title?: string
}

export function ExportCsvButton({ filename, columns, title = 'Export CSV' }: Props) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation()
        downloadCsv(filename, columns)
      }}
      className="p-1 rounded text-text-muted hover:text-text-primary hover:bg-white/5 transition-colors"
      title={title}
    >
      <Download className="w-3.5 h-3.5" />
    </button>
  )
}
```

Note: `e.stopPropagation()` prevents the click from toggling the parent accordion `Section` when the button is placed inside the section header.

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors related to ExportCsvButton.tsx

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/pastas/ExportCsvButton.tsx
git commit -m "feat(pastas): add ExportCsvButton reusable component"
```

---

### Task 3: Export Buttons in FitResultsPanel

**Files:**
- Modify: `frontend/src/components/pastas/FitResultsPanel.tsx`

The FitResultsPanel has a `Section` component wrapping the Model Analysis chart. We need to add export buttons next to the section title for:
1. Observed/Simulated/Residuals
2. Contributions

Since `Section` renders a title string, we'll place the export buttons just inside the section body, before the chart — as a small toolbar row.

- [ ] **Step 1: Add imports**

At the top of `FitResultsPanel.tsx`, add:

```tsx
import { ExportCsvButton } from './ExportCsvButton'
import type { CsvColumn } from '@/lib/csv-export'
```

- [ ] **Step 2: Add export buttons before UnifiedAnalysisChart**

Find the section that renders `UnifiedAnalysisChart` (around line 277). Just before the `{selectedOutlierDate && (` block (line 278), insert an export toolbar:

```tsx
        <div className="flex items-center gap-1 mb-2">
          <ExportCsvButton
            filename={`${result.code_bss}_observed_simulated_residuals.csv`}
            title="Export observed, simulated & residuals as CSV"
            columns={(() => {
              const cols: CsvColumn[] = [
                { header: 'date', values: observed.index },
                { header: 'observed', values: observed.values },
                { header: 'simulated', values: simulated.values },
              ]
              const resVals = observed.values.map((obs, i) => {
                const sim = simulated.values[i]
                return obs != null && sim != null ? obs - sim : null
              })
              cols.push({ header: 'residuals', values: resVals })
              return cols
            })()}
          />
          <ExportCsvButton
            filename={`${result.code_bss}_contributions.csv`}
            title="Export stress contributions as CSV"
            columns={(() => {
              const entries = Object.entries(contributions)
              if (entries.length === 0) return []
              const cols: CsvColumn[] = [{ header: 'date', values: entries[0][1].index }]
              for (const [name, ts] of entries) {
                cols.push({ header: name, values: ts.values })
              }
              return cols
            })()}
          />
        </div>
```

The columns are computed inline via IIFE — no useMemo needed since this only runs on click (columns prop is read lazily by ExportCsvButton on click).

Wait — actually ExportCsvButton receives `columns` as a prop, which means the IIFE runs on every render. Let's fix this: instead of computing columns in the prop, we should use a callback pattern. Let me revise the ExportCsvButton to accept a `getColumns` function instead.

**Revised ExportCsvButton:**

```tsx
// frontend/src/components/pastas/ExportCsvButton.tsx
import { Download } from 'lucide-react'
import { downloadCsv, type CsvColumn } from '@/lib/csv-export'

interface Props {
  filename: string
  getColumns: () => CsvColumn[]
  title?: string
}

export function ExportCsvButton({ filename, getColumns, title = 'Export CSV' }: Props) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation()
        downloadCsv(filename, getColumns())
      }}
      className="p-1 rounded text-text-muted hover:text-text-primary hover:bg-white/5 transition-colors"
      title={title}
    >
      <Download className="w-3.5 h-3.5" />
    </button>
  )
}
```

Now the columns are only computed when clicked. The integration becomes:

```tsx
        <div className="flex items-center gap-1 mb-2">
          <ExportCsvButton
            filename={`${result.code_bss}_observed_simulated_residuals.csv`}
            title="Export observed, simulated & residuals as CSV"
            getColumns={() => {
              const cols: CsvColumn[] = [
                { header: 'date', values: observed.index },
                { header: 'observed', values: observed.values },
                { header: 'simulated', values: simulated.values },
              ]
              const resVals = observed.values.map((obs, i) => {
                const sim = simulated.values[i]
                return obs != null && sim != null ? obs - sim : null
              })
              cols.push({ header: 'residuals', values: resVals })
              return cols
            }}
          />
          <ExportCsvButton
            filename={`${result.code_bss}_contributions.csv`}
            title="Export stress contributions as CSV"
            getColumns={() => {
              const entries = Object.entries(contributions)
              if (entries.length === 0) return []
              const cols: CsvColumn[] = [{ header: 'date', values: entries[0][1].index }]
              for (const [name, ts] of entries) {
                cols.push({ header: name, values: ts.values })
              }
              return cols
            }}
          />
        </div>
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

- [ ] **Step 4: Build frontend in Docker and verify**

Run: `docker compose up -d --build frontend`
Expected: Build succeeds. Navigate to Pastas results page. Two small download icons appear above the analysis chart.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/pastas/FitResultsPanel.tsx
git commit -m "feat(pastas): add CSV export buttons for fit results (obs/sim/residuals + contributions)"
```

---

### Task 4: Export Buttons in ScenarioResultsPanel

**Files:**
- Modify: `frontend/src/components/pastas/ScenarioResultsPanel.tsx`
- Modify: `frontend/src/pages/pastas/ScenariosPage.tsx`
- Modify: `frontend/src/components/pastas/ScenarioWorkflow.tsx`

- [ ] **Step 1: Add `codeBss` prop to ScenarioResultsPanel**

In `ScenarioResultsPanel.tsx`, update the Props interface:

```tsx
interface Props {
  result: PastasScenarioResponse
  modifications?: ModificationData[]
  codeBss?: string
}
```

Update the destructuring:

```tsx
export function ScenarioResultsPanel({ result, modifications, codeBss }: Props) {
```

Add imports at the top:

```tsx
import { Download } from 'lucide-react'
import { ExportCsvButton } from './ExportCsvButton'
import type { CsvColumn } from '@/lib/csv-export'
```

Remove `Download` from existing `lucide-react` import if it's not already there — the existing import is `{ ChevronDown, TrendingDown, TrendingUp, Minus }`. Just add `Download` to it:

```tsx
import { ChevronDown, TrendingDown, TrendingUp, Minus, Download } from 'lucide-react'
```

Wait — actually `Download` is imported via the `ExportCsvButton` component, not directly used in ScenarioResultsPanel. So just add the `ExportCsvButton` import:

```tsx
import { ExportCsvButton } from './ExportCsvButton'
import type { CsvColumn } from '@/lib/csv-export'
```

- [ ] **Step 2: Add export button to "Baseline vs Scenario" section**

The `Section` component in ScenarioResultsPanel takes a `title` string. To add a button next to the title without modifying the Section component, we'll modify the Section component in this file to accept an optional `extra` React node in the header bar.

Update the local `Section` component:

```tsx
function Section({ title, defaultOpen = true, extra, children }: {
  title: string; defaultOpen?: boolean; extra?: React.ReactNode; children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="bg-bg-primary rounded-lg border border-white/5 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-bg-hover transition-colors"
      >
        <span className="flex items-center gap-2">
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">{title}</span>
          {extra}
        </span>
        <ChevronDown className={`w-4 h-4 text-text-muted transition-transform ${open ? '' : '-rotate-90'}`} />
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}
```

Then pass the export button via `extra` on the "Baseline vs Scenario" section:

```tsx
      <Section
        title="Baseline vs Scenario"
        extra={
          <ExportCsvButton
            filename={`${codeBss ?? 'scenario'}_baseline_vs_scenario.csv`}
            title="Export baseline, scenario & delta as CSV"
            getColumns={() => [
              { header: 'date', values: baseline.index },
              { header: 'baseline', values: baseline.values },
              { header: 'scenario', values: scenario.values },
              { header: 'delta', values: delta.values },
            ]}
          />
        }
      >
```

- [ ] **Step 3: Add export button to "Stress Contributions" section**

```tsx
      <Section
        title="Stress Contributions"
        defaultOpen={false}
        extra={
          <ExportCsvButton
            filename={`${codeBss ?? 'scenario'}_contributions.csv`}
            title="Export baseline & scenario contributions as CSV"
            getColumns={() => {
              const cols: CsvColumn[] = [{ header: 'date', values: (contributions_baseline[allContribNames[0]] ?? contributions_scenario[allContribNames[0]])?.index ?? [] }]
              for (const name of allContribNames) {
                const bl = contributions_baseline[name]
                const sc = contributions_scenario[name]
                if (bl) cols.push({ header: `${name}_baseline`, values: bl.values })
                if (sc) cols.push({ header: `${name}_scenario`, values: sc.values })
              }
              return cols
            }}
          />
        }
      >
```

- [ ] **Step 4: Pass codeBss in ScenariosPage.tsx**

In `frontend/src/pages/pastas/ScenariosPage.tsx`, find the line:

```tsx
<ScenarioResultsPanel result={simResult} modifications={modifications} />
```

Change to:

```tsx
<ScenarioResultsPanel result={simResult} modifications={modifications} codeBss={codeBss} />
```

`codeBss` is already defined at line 54: `const codeBss = selected?.code_bss`

- [ ] **Step 5: Pass codeBss in ScenarioWorkflow.tsx**

In `frontend/src/components/pastas/ScenarioWorkflow.tsx`, find:

```tsx
<ScenarioResultsPanel result={simResult} modifications={modifications} />
```

Change to pass `codeBss`. Check what prop is available in ScenarioWorkflow:

```bash
grep -n "code_bss\|codeBss" frontend/src/components/pastas/ScenarioWorkflow.tsx
```

If `codeBss` is available as a prop or local variable, pass it. If not, add it as a prop to ScenarioWorkflow and thread it through from the parent page.

- [ ] **Step 6: Verify TypeScript compiles**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | head -20`
Expected: No errors

- [ ] **Step 7: Build frontend in Docker and verify**

Run: `docker compose up -d --build frontend`
Expected: Build succeeds. Navigate to Scenario page, run a scenario. Export icons visible in "Baseline vs Scenario" and "Stress Contributions" section headers.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/pastas/ScenarioResultsPanel.tsx frontend/src/pages/pastas/ScenariosPage.tsx frontend/src/components/pastas/ScenarioWorkflow.tsx
git commit -m "feat(pastas): add CSV export buttons for scenario results"
```

---

### Task 5: Final Verification

- [ ] **Step 1: Full TypeScript check**

Run: `cd /home/ringuet/time-serie-explo/frontend && npx tsc --noEmit --pretty 2>&1 | tail -5`
Expected: No errors

- [ ] **Step 2: Docker build**

Run: `docker compose up -d --build frontend`
Expected: Builds and starts successfully

- [ ] **Step 3: Manual smoke test**

Test in browser:
1. Go to Pastas results page → verify 2 download icons above the analysis chart
2. Click the first icon → downloads CSV with date, observed, simulated, residuals columns
3. Click the second icon → downloads CSV with date + stress contribution columns
4. Go to Scenarios page → run a scenario
5. Click export in "Baseline vs Scenario" → downloads CSV with date, baseline, scenario, delta
6. Click export in "Stress Contributions" → downloads CSV with baseline & scenario contributions

- [ ] **Step 4: Final commit (if any fixes needed)**

```bash
git add -u
git commit -m "fix(pastas): CSV export polish"
```
