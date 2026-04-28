# Pastas CSV Export

## Goal

Allow users to export key time series data from Pastas results and scenarios as CSV files, for external analysis in Excel/R/Python.

## Scope

Four export points, all frontend-only (data already in component state):

| # | What | Location | CSV Columns |
|---|------|----------|-------------|
| 1 | Observed + Simulated + Residuals | UnifiedAnalysisChart header | `date, observed, simulated, residuals` |
| 2 | Stress Contributions | UnifiedAnalysisChart contributions section | `date, {stress_1}, {stress_2}, ...` |
| 3 | Scenario Comparison | ScenarioResultsPanel header | `date, baseline, scenario, delta` |
| 4 | Scenario Contributions | ScenarioResultsPanel contributions chart | `date, {stress}_baseline, {stress}_scenario, ...` |

## Design

### Shared Utility: `frontend/src/lib/csv-export.ts`

```ts
type CsvColumn = { header: string; values: (string | number | null)[] }

function downloadCsv(filename: string, columns: CsvColumn[]): void
```

- Builds CSV from columns array (handles null as empty cell, quotes values containing commas)
- Triggers browser download via Blob + `<a>` click pattern
- Filename convention: `{code_bss}_{type}_{period}.csv`

### Shared Component: `frontend/src/components/pastas/ExportCsvButton.tsx`

```tsx
interface Props {
  filename: string
  columns: CsvColumn[]
  title?: string  // tooltip, defaults to "Export CSV"
}
```

- Renders ArrowDownTrayIcon (Heroicons) as small icon button
- On click: calls `downloadCsv(filename, columns)`
- Styling: `text-text-muted hover:text-text-primary` — consistent with existing icon buttons

### Integration Points

**`UnifiedAnalysisChart.tsx`** — two ExportCsvButton instances:
1. In main chart header: exports observed/simulated/residuals
2. In contributions section: exports per-stress contributions

Data source: props passed from FitResultsPanel (fitResponse fields).

**`ScenarioResultsPanel.tsx`** — two ExportCsvButton instances:
1. In results header: exports baseline/scenario/delta
2. In contributions chart header: exports baseline vs scenario contributions

Data source: scenarioResponse fields already in component state.

## Files Changed

| File | Action |
|------|--------|
| `frontend/src/lib/csv-export.ts` | **New** — downloadCsv utility + CsvColumn type |
| `frontend/src/components/pastas/ExportCsvButton.tsx` | **New** — reusable icon button |
| `frontend/src/components/pastas/UnifiedAnalysisChart.tsx` | **Modified** — add 2 export buttons |
| `frontend/src/components/pastas/ScenarioResultsPanel.tsx` | **Modified** — add 2 export buttons |

## Out of Scope

- No backend changes
- No Excel/XLSX format
- No export for diagnostics, signatures, or response functions
- No bulk "export all" — each button exports exactly what the user sees
