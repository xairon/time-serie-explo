import { useState } from 'react'
import Plot from 'react-plotly.js'
import { ChevronDown, TrendingDown, TrendingUp, Minus } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { PastasScenarioResponse } from '@/lib/types'
import type { Layout } from 'plotly.js-dist-min'
import type { ModificationData } from './ModificationCard'
import { ExportCsvButton } from './ExportCsvButton'
import type { CsvColumn } from '@/lib/csv-export'

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

function argExtreme(values: number[], dates: string[]): { date: string; value: number } | null {
  let idx = -1
  let bestAbs = -Infinity
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) continue
    if (Math.abs(v) > bestAbs) {
      bestAbs = Math.abs(v)
      idx = i
    }
  }
  if (idx < 0) return null
  return { date: dates[idx], value: values[idx] }
}

function severityColor(meanDelta: number, t: (k: string) => string): { bg: string; border: string; text: string; label: string } {
  const abs = Math.abs(meanDelta)
  if (abs < 0.05) return { bg: 'bg-green-500/10', border: 'border-green-500/20', text: 'text-green-400', label: t('pastas.scenarioResults.negligible') }
  if (abs < 0.2) return { bg: 'bg-yellow-500/10', border: 'border-yellow-500/20', text: 'text-yellow-400', label: t('pastas.scenarioResults.moderate') }
  if (abs < 0.5) return { bg: 'bg-orange-500/10', border: 'border-orange-500/20', text: 'text-orange-400', label: t('pastas.scenarioResults.significant') }
  return { bg: 'bg-red-500/10', border: 'border-red-500/20', text: 'text-red-400', label: t('pastas.scenarioResults.severe') }
}

const CONTRIB_COLORS = ['#60a5fa', '#34d399', '#f97316', '#a78bfa', '#f43f5e', '#eab308', '#06b6d4']

interface Props {
  result: PastasScenarioResponse
  modifications?: ModificationData[]
  codeBss?: string
}

export function ScenarioResultsPanel({ result, modifications, codeBss }: Props) {
  const { t } = useTranslation()
  const MOD_LABELS: Record<string, string> = {
    pumping_synthetic: t('pastas.scenarioResults.modLabelPumpingSynthetic'),
    pumping_upload: t('pastas.scenarioResults.modLabelPumpingUpload'),
    linear_trend: t('pastas.scenarioResults.modLabelLinearTrend'),
    scale_stress: t('pastas.scenarioResults.modLabelScaleStress'),
  }
  const { baseline, scenario, delta, contributions_baseline, contributions_scenario, warnings } = result

  const deltaStats = computeStats(delta.values)
  const baselineStats = computeStats(baseline.values)
  const scenarioStats = computeStats(scenario.values)

  const contribBaselineNames = Object.keys(contributions_baseline ?? {})
  const contribScenarioNames = Object.keys(contributions_scenario ?? {})
  const allContribNames = [...new Set([...contribBaselineNames, ...contribScenarioNames])]
  const newContribs = contribScenarioNames.filter(n => !contribBaselineNames.includes(n))

  const severity = deltaStats ? severityColor(deltaStats.mean, t) : null
  const TrendIcon = deltaStats ? (deltaStats.mean < -0.01 ? TrendingDown : deltaStats.mean > 0.01 ? TrendingUp : Minus) : Minus
  const peakDelta = argExtreme(delta.values, delta.index)
  const peakDeltaIso = peakDelta?.date?.slice(0, 10) ?? null

  return (
    <div className="space-y-4">
      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
          <p className="text-xs font-semibold text-yellow-400 mb-1">{t('pastas.scenarioResults.warnings')}</p>
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
                {t('pastas.scenarioResults.meanChange')}
                {' · '}{t('pastas.scenarios.deltaRange')} : {deltaStats.min.toFixed(3)} m {t('common.to')} {deltaStats.max > 0 ? '+' : ''}{deltaStats.max.toFixed(3)} m
              </p>
            </div>
            <div className="flex gap-4 shrink-0">
              <div className="text-center">
                <div className="text-[10px] text-text-muted">{t('pastas.scenarioResults.reference')}</div>
                <div className="text-sm font-mono text-text-primary">{baselineStats?.mean.toFixed(2)} m</div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-text-muted">{t('pastas.scenarioResults.scenario')}</div>
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
              {mod.type === 'pumping_synthetic' && <span className="text-text-muted">({mod.rate_m3d} m³/j, {mod.pattern})</span>}
              {mod.type === 'linear_trend' && <span className="text-text-muted">({mod.slope_m_per_year > 0 ? '+' : ''}{mod.slope_m_per_year} m/an)</span>}
            </span>
          ))}
        </div>
      )}

      {/* Impact (Δh) — PRIMARY chart, big and prominent */}
      {delta?.index?.length > 0 && (
        <Section
          title={t('pastas.scenarioResults.impactTitle')}
          extra={
            <ExportCsvButton
              filename={`${codeBss ?? 'scenario'}_baseline_vs_scenario.csv`}
              title={t('pastas.scenarioResults.exportBaseVsScen')}
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
            data={[{
              x: delta.index,
              y: delta.values,
              type: 'scatter',
              mode: 'lines',
              fill: 'tozeroy',
              name: 'Δh',
              line: { color: '#f59e0b', width: 2 },
              fillcolor: 'rgba(245,158,11,0.25)',
              hovertemplate: '%{x|%d %b %Y}<br>Δh = %{y:.3f} m<extra></extra>',
            }]}
            layout={{
              ...chartLayout,
              height: 340,
              yaxis: {
                ...chartLayout.yaxis,
                title: { text: t('pastas.scenarioResults.deltaAxis') },
                gridcolor: 'rgba(255,255,255,0.05)',
                zeroline: true,
                zerolinecolor: 'rgba(255,255,255,0.3)',
                zerolinewidth: 1.5,
              },
              shapes: [{
                type: 'line', x0: delta.index[0], x1: delta.index[delta.index.length - 1],
                y0: 0, y1: 0, line: { color: 'rgba(255,255,255,0.3)', width: 1.5 },
              }],
              annotations: peakDelta && peakDeltaIso ? [{
                x: peakDeltaIso,
                y: peakDelta.value,
                text: t('pastas.scenarioResults.peakAnnotation', { value: `${peakDelta.value >= 0 ? '+' : ''}${peakDelta.value.toFixed(3)}` }),
                showarrow: true,
                arrowhead: 2,
                arrowsize: 0.8,
                arrowcolor: 'rgba(245,158,11,0.8)',
                ax: 0,
                ay: peakDelta.value < 0 ? -30 : 30,
                font: { size: 10, color: '#f59e0b' },
                bgcolor: 'rgba(0,0,0,0.4)',
                bordercolor: 'rgba(245,158,11,0.5)',
                borderwidth: 1,
                borderpad: 3,
              }] : [],
            }}
            config={plotlyConfig} style={{ width: '100%' }}
          />
          {peakDelta && (
            <div className="grid grid-cols-3 gap-2 mt-2">
              <div className="bg-bg-card rounded-md border border-white/5 px-3 py-2">
                <div className="text-[10px] text-text-muted">{t('pastas.scenarios.deltaMean')}</div>
                <div className={`text-sm font-mono ${severity?.text ?? ''}`}>{deltaStats!.mean >= 0 ? '+' : ''}{deltaStats!.mean.toFixed(3)} m</div>
              </div>
              <div className="bg-bg-card rounded-md border border-white/5 px-3 py-2">
                <div className="text-[10px] text-text-muted">{t('pastas.scenarios.deltaPeak')}</div>
                <div className="text-sm font-mono text-text-primary">{peakDelta.value >= 0 ? '+' : ''}{peakDelta.value.toFixed(3)} m</div>
                <div className="text-[9px] text-text-muted">{peakDeltaIso}</div>
              </div>
              <div className="bg-bg-card rounded-md border border-white/5 px-3 py-2">
                <div className="text-[10px] text-text-muted">{t('pastas.scenarios.deltaRange')}</div>
                <div className="text-sm font-mono text-text-primary">{deltaStats!.min.toFixed(3)} → {deltaStats!.max >= 0 ? '+' : ''}{deltaStats!.max.toFixed(3)} m</div>
              </div>
            </div>
          )}
        </Section>
      )}

      {/* Baseline vs Scenario — secondary view, collapsed by default */}
      <Section
        title={t('pastas.scenarios.absoluteLevels')}
        defaultOpen={false}
      >
        <Plot
          data={[
            {
              x: baseline.index, y: baseline.values, type: 'scatter', mode: 'lines',
              name: t('pastas.scenarioResults.reference'), line: { color: '#6b7280', width: 1.5 },
            },
            {
              x: scenario.index, y: scenario.values, type: 'scatter', mode: 'lines',
              name: t('pastas.scenarioResults.scenario'), line: { color: '#22d3ee', width: 2 },
              fill: 'tonexty' as const, fillcolor: 'rgba(245,158,11,0.25)',
            },
          ]}
          layout={{
            ...chartLayout,
            legend: { orientation: 'h', y: -0.15, font: { size: 10, color: '#9ca3af' } },
            yaxis: { ...chartLayout.yaxis, title: { text: t('pastas.scenarioResults.levelAxis') } },
          }}
          config={plotlyConfig} style={{ width: '100%' }}
        />
      </Section>

      {/* Per-stress contributions — separate charts */}
      {allContribNames.length > 0 && (
        <Section
          title={t('pastas.scenarios.contributions')}
          defaultOpen={false}
          extra={
            <ExportCsvButton
              filename={`${codeBss ?? 'scenario'}_scenario_contributions.csv`}
              title={t('pastas.scenarioResults.exportContributions')}
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
          <div className="space-y-2">
            {allContribNames.map((name, i) => {
              const color = CONTRIB_COLORS[i % CONTRIB_COLORS.length]
              const bl = contributions_baseline?.[name]
              const sc = contributions_scenario?.[name]
              const traces: Plotly.Data[] = []

              if (bl?.index?.length) {
                traces.push({
                  x: bl.index, y: bl.values, type: 'scatter' as const, mode: 'lines' as const,
                  name: t('pastas.scenarioResults.reference'), line: { color: '#6b7280', width: 1, dash: 'dot' as const },
                })
              }
              if (sc?.index?.length) {
                traces.push({
                  x: sc.index, y: sc.values, type: 'scatter' as const, mode: 'lines' as const,
                  name: t('pastas.scenarioResults.scenario'), line: { color, width: 2 },
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
                      <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20">{t('pastas.scenarioResults.newBadge')}</span>
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
