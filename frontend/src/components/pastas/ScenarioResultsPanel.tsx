import { useState } from 'react'
import Plot from 'react-plotly.js'
import { ChevronDown, TrendingDown, TrendingUp, Minus } from 'lucide-react'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { PastasScenarioResponse } from '@/lib/types'
import type { Layout } from 'plotly.js-dist-min'
import type { ModificationData } from './ModificationCard'
import { ExportCsvButton } from './ExportCsvButton'
import type { CsvColumn } from '@/lib/csv-export'

const MOD_LABELS: Record<string, string> = {
  pumping_synthetic: 'Synthetic pumping',
  pumping_upload: 'Pumping (CSV)',
  linear_trend: 'Linear trend',
  scale_stress: 'Stress scaling',
}

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

const chartLayout: Partial<Layout> = {
  ...darkLayout,
  margin: { t: 20, r: 20, b: 40, l: 60 },
  height: 260,
}

function computeStats(values: number[]) {
  const valid = values.filter(v => Number.isFinite(v))
  if (valid.length === 0) return null
  const mean = valid.reduce((a, b) => a + b, 0) / valid.length
  const min = Math.min(...valid)
  const max = Math.max(...valid)
  return { mean, min, max }
}

function severityColor(meanDelta: number): { bg: string; border: string; text: string; label: string } {
  const abs = Math.abs(meanDelta)
  if (abs < 0.05) return { bg: 'bg-green-500/10', border: 'border-green-500/20', text: 'text-green-400', label: 'Negligible' }
  if (abs < 0.2) return { bg: 'bg-yellow-500/10', border: 'border-yellow-500/20', text: 'text-yellow-400', label: 'Moderate' }
  if (abs < 0.5) return { bg: 'bg-orange-500/10', border: 'border-orange-500/20', text: 'text-orange-400', label: 'Significant' }
  return { bg: 'bg-red-500/10', border: 'border-red-500/20', text: 'text-red-400', label: 'Severe' }
}

const CONTRIB_COLORS = ['#60a5fa', '#34d399', '#f97316', '#a78bfa', '#f43f5e', '#eab308', '#06b6d4']

interface Props {
  result: PastasScenarioResponse
  modifications?: ModificationData[]
  codeBss?: string
}

export function ScenarioResultsPanel({ result, modifications, codeBss }: Props) {
  const { baseline, scenario, delta, contributions_baseline, contributions_scenario, warnings } = result

  const deltaStats = computeStats(delta.values)
  const baselineStats = computeStats(baseline.values)
  const scenarioStats = computeStats(scenario.values)

  const contribBaselineNames = Object.keys(contributions_baseline ?? {})
  const contribScenarioNames = Object.keys(contributions_scenario ?? {})
  const allContribNames = [...new Set([...contribBaselineNames, ...contribScenarioNames])]
  const newContribs = contribScenarioNames.filter(n => !contribBaselineNames.includes(n))

  const severity = deltaStats ? severityColor(deltaStats.mean) : null
  const TrendIcon = deltaStats ? (deltaStats.mean < -0.01 ? TrendingDown : deltaStats.mean > 0.01 ? TrendingUp : Minus) : Minus

  return (
    <div className="space-y-4">
      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
          <p className="text-xs font-semibold text-yellow-400 mb-1">Warnings</p>
          {warnings.map((w, i) => <p key={i} className="text-xs text-yellow-300">{w}</p>)}
        </div>
      )}

      {/* Impact summary banner */}
      {deltaStats && severity && (
        <div className={`${severity.bg} border ${severity.border} rounded-xl p-4`}>
          <div className="flex items-center gap-4">
            <TrendIcon className={`w-8 h-8 ${severity.text} shrink-0`} />
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className={`text-lg font-bold ${severity.text}`}>
                  {deltaStats.mean > 0 ? '+' : ''}{deltaStats.mean.toFixed(3)} m
                </span>
                <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${severity.bg} ${severity.text} border ${severity.border}`}>
                  {severity.label}
                </span>
              </div>
              <p className="text-xs text-text-secondary mt-0.5">
                Average level change over the simulation period
                {' · '}Range: {deltaStats.min.toFixed(3)} m to {deltaStats.max > 0 ? '+' : ''}{deltaStats.max.toFixed(3)} m
              </p>
            </div>
            <div className="flex gap-4 shrink-0">
              <div className="text-center">
                <div className="text-[10px] text-text-muted">Baseline</div>
                <div className="text-sm font-mono text-text-primary">{baselineStats?.mean.toFixed(2)} m</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-text-muted">Scenario</div>
                <div className={`text-sm font-mono ${severity.text}`}>{scenarioStats?.mean.toFixed(2)} m</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Modification recap */}
      {modifications && modifications.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {modifications.map((mod, i) => (
            <span key={i} className="inline-flex items-center gap-1 px-2 py-1 rounded-lg border border-white/10 text-[10px] text-text-secondary bg-bg-card">
              <span className="font-medium">{MOD_LABELS[mod.type] ?? mod.type}</span>
              {mod.type === 'scale_stress' && <span className="text-text-muted">({mod.stress} ×{mod.factor})</span>}
              {mod.type === 'pumping_synthetic' && <span className="text-text-muted">({mod.rate_m3d} m³/d, {mod.pattern})</span>}
              {mod.type === 'linear_trend' && <span className="text-text-muted">({mod.slope_m_per_year > 0 ? '+' : ''}{mod.slope_m_per_year} m/yr)</span>}
            </span>
          ))}
        </div>
      )}

      {/* Baseline vs Scenario — side by side */}
      <Section
        title="Baseline vs Scenario"
        extra={
          <ExportCsvButton
            filename={`${codeBss ?? 'scenario'}_baseline_vs_scenario.csv`}
            title="Export baseline, scenario & delta as CSV"
            label="CSV"
            getColumns={() => [
              { header: 'date', values: baseline.index },
              { header: 'baseline', values: baseline.values },
              { header: 'scenario', values: scenario.values },
              { header: 'delta', values: delta.values },
            ]}
          />
        }
      >
        <Plot
          data={[
            {
              x: baseline.index, y: baseline.values, type: 'scatter', mode: 'lines',
              name: 'Baseline', line: { color: '#6b7280', width: 1.5 },
            },
            {
              x: scenario.index, y: scenario.values, type: 'scatter', mode: 'lines',
              name: 'Scenario', line: { color: '#22d3ee', width: 2 },
            },
          ]}
          layout={{
            ...chartLayout,
            legend: { orientation: 'h', y: -0.15, font: { size: 10, color: '#9ca3af' } },
          }}
          config={plotlyConfig} style={{ width: '100%' }}
        />
      </Section>

      {/* Delta / Impact */}
      {delta?.index?.length > 0 && (
        <Section title="Impact (Δh = scenario − baseline)">
          <Plot
            data={[{
              x: delta.index,
              y: delta.values,
              type: 'scatter',
              mode: 'lines',
              fill: 'tozeroy',
              name: 'Delta',
              line: { color: '#f59e0b', width: 1.5 },
              fillcolor: 'rgba(245,158,11,0.12)',
            }]}
            layout={{
              ...chartLayout, height: 200,
              yaxis: { ...chartLayout.yaxis, title: { text: 'Δh (m)' }, gridcolor: 'rgba(255,255,255,0.05)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.2)' },
              shapes: [{
                type: 'line', x0: delta.index[0], x1: delta.index[delta.index.length - 1],
                y0: 0, y1: 0, line: { color: 'rgba(255,255,255,0.15)', width: 1 },
              }],
            }}
            config={plotlyConfig} style={{ width: '100%' }}
          />
        </Section>
      )}

      {/* Per-stress contributions — separate charts */}
      {allContribNames.length > 0 && (
        <Section
          title="Stress Contributions"
          defaultOpen={false}
          extra={
            <ExportCsvButton
              filename={`${codeBss ?? 'scenario'}_scenario_contributions.csv`}
              title="Export baseline & scenario contributions as CSV"
              label="CSV"
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
          <p className="text-xs text-text-muted mb-3">
            Each stress contribution shown independently. Gray dashed = baseline, solid color = scenario.
            {newContribs.length > 0 && <span className="text-accent-cyan"> New: {newContribs.join(', ')}</span>}
          </p>
          <div className="space-y-2">
            {allContribNames.map((name, i) => {
              const color = CONTRIB_COLORS[i % CONTRIB_COLORS.length]
              const bl = contributions_baseline?.[name]
              const sc = contributions_scenario?.[name]
              const traces: Plotly.Data[] = []

              if (bl?.index?.length) {
                traces.push({
                  x: bl.index, y: bl.values, type: 'scatter' as const, mode: 'lines' as const,
                  name: 'Baseline', line: { color: '#6b7280', width: 1, dash: 'dot' as const },
                })
              }
              if (sc?.index?.length) {
                traces.push({
                  x: sc.index, y: sc.values, type: 'scatter' as const, mode: 'lines' as const,
                  name: 'Scenario', line: { color, width: 2 },
                  fill: 'tozeroy' as const, fillcolor: color + '15',
                })
              }

              if (traces.length === 0) return null

              const isNew = newContribs.includes(name)

              return (
                <div key={name} className="bg-bg-card rounded-lg border border-white/5 p-2">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-[10px] font-semibold" style={{ color }}>{name}</span>
                    {isNew && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20">new</span>
                    )}
                  </div>
                  <Plot
                    data={traces}
                    layout={{
                      paper_bgcolor: 'transparent',
                      plot_bgcolor: 'transparent',
                      font: { color: '#9ca3af', size: 9 },
                      margin: { t: 5, r: 15, b: 25, l: 50 },
                      height: 120,
                      xaxis: { gridcolor: 'rgba(255,255,255,0.03)' },
                      yaxis: { gridcolor: 'rgba(255,255,255,0.05)', title: { text: 'm', font: { size: 9 } } },
                      legend: { orientation: 'h' as const, y: -0.3, font: { size: 9, color: '#9ca3af' } },
                      showlegend: !!bl && !!sc,
                    }}
                    useResizeHandler
                    className="w-full"
                    config={{ displayModeBar: false }}
                  />
                </div>
              )
            })}
          </div>
        </Section>
      )}
    </div>
  )
}
