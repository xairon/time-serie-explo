import { useState } from 'react'
import { ChevronDown, Brain } from 'lucide-react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { PastasFitResponse } from '@/lib/types'
import type { Layout } from 'plotly.js-dist-min'
import { ContributionsChart } from '@/components/pastas/ContributionsChart'
import { usePastasDiagnostics, usePastasSignatures, usePastasOutlierDiagnostics, usePastasConfidenceBands, usePastasRecession, usePastasBaseflow, usePastasSpectral, usePastasDecomposition, usePastasCrossCorrelation, usePastasRegionalResiduals, usePastasInputQuality } from '@/hooks/usePastas'
import { useModels } from '@/hooks/useModels'
import { DiagnosticsPanel } from './DiagnosticsPanel'
import { ResponsePanel } from './ResponsePanel'
import { SignaturesPanel } from './SignaturesPanel'
import { OutlierDetailPanel } from './OutlierDetailPanel'
import { RecessionPanel } from './RecessionPanel'
import { BaseflowPanel } from './BaseflowPanel'
import { SpectralPanel } from './SpectralPanel'
import { DecompositionPanel } from './DecompositionPanel'
import { CrossCorrelationPanel } from './CrossCorrelationPanel'
import { RegionalResidualsPanel } from './RegionalResidualsPanel'

// --- Accordion section ---

function Section({ title, defaultOpen = true, children }: {
  title: string; defaultOpen?: boolean; children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="bg-bg-primary rounded-lg border border-white/5 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-bg-hover transition-colors"
      >
        <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">{title}</span>
        <ChevronDown className={`w-4 h-4 text-text-muted transition-transform ${open ? '' : '-rotate-90'}`} />
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}

// --- Metric definitions ---

const METRIC_DEFS: Record<string, {
  label: string; tooltip: string
  format: (v: number) => string
  quality: (v: number) => 'good' | 'ok' | 'poor'
}> = {
  nse: {
    label: 'NSE', tooltip: 'Nash-Sutcliffe Efficiency. 1 = perfect, 0 = as good as mean, <0 = worse.',
    format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor',
  },
  kge: {
    label: 'KGE', tooltip: 'Kling-Gupta Efficiency. Combines correlation, bias and variability. >0.7 = good.',
    format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor',
  },
  evp: {
    label: 'EVP (%)', tooltip: 'Explained Variance Percentage. 100% = fully explained variance.',
    format: v => v.toFixed(1), quality: v => v > 70 ? 'good' : v > 40 ? 'ok' : 'poor',
  },
  rmse: {
    label: 'RMSE', tooltip: 'Root Mean Square Error (m). Lower is better.',
    format: v => v.toFixed(4), quality: () => 'ok',
  },
  rsq: {
    label: 'R²', tooltip: 'Coefficient of determination. 1 = perfect correlation.',
    format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor',
  },
  mae: {
    label: 'MAE', tooltip: 'Mean Absolute Error (m). Average absolute error.',
    format: v => v.toFixed(4), quality: () => 'ok',
  },
}

const Q_COLORS = { good: 'text-green-400', ok: 'text-accent-cyan', poor: 'text-red-400' }
const Q_BORDERS = { good: 'border-green-500/20', ok: 'border-white/5', poor: 'border-red-500/20' }

function MetricCard({ metricKey, value }: { metricKey: string; value: number | null | undefined }) {
  const def = METRIC_DEFS[metricKey]
  if (!def) return null
  const hasValue = value !== null && value !== undefined && Number.isFinite(value)
  const quality = hasValue ? def.quality(value!) : 'ok'
  return (
    <div className={`bg-bg-card rounded-lg p-2.5 border ${Q_BORDERS[quality]}`} title={def.tooltip}>
      <div className="text-[10px] text-text-muted mb-0.5 flex items-center gap-1">
        {def.label}<span className="cursor-help opacity-50">ⓘ</span>
      </div>
      <div className={`text-base font-semibold ${hasValue ? Q_COLORS[quality] : 'text-text-muted'}`}>
        {hasValue ? def.format(value!) : '—'}
      </div>
    </div>
  )
}

function MetricGrid({ metrics, title, period, borderColor }: {
  metrics: Record<string, number>; title?: string; period?: string[] | null; borderColor?: string
}) {
  return (
    <div className={`rounded-lg border ${borderColor ?? 'border-white/5'} p-3`}>
      {title && (
        <div className={`text-xs font-semibold uppercase tracking-wide mb-2 flex items-center gap-2 ${
          borderColor?.includes('orange') ? 'text-orange-400' : 'text-accent-cyan'
        }`}>
          {title}
          {period && <span className="text-text-muted font-normal normal-case text-[10px]">{period[0]} → {period[1]}</span>}
        </div>
      )}
      <div className="grid grid-cols-3 gap-2">
        {['nse', 'kge', 'evp', 'rmse', 'rsq', 'mae'].map(k => (
          <MetricCard key={k} metricKey={k} value={metrics[k]} />
        ))}
      </div>
    </div>
  )
}

const chartLayout: Partial<Layout> = {
  ...darkLayout,
  margin: { t: 20, r: 20, b: 40, l: 60 },
  height: 220,
}

// --- Main component ---

interface FitResultsPanelProps {
  result: PastasFitResponse
  codeBss?: string
}

export function FitResultsPanel({ result, codeBss }: FitResultsPanelProps) {
  const { data: diagnosticsData } = usePastasDiagnostics(result.run_id)
  const { data: signaturesData } = usePastasSignatures(result.run_id)
  const { data: outlierData } = usePastasOutlierDiagnostics(result.run_id) as { data: any }
  const { data: confidenceData } = usePastasConfidenceBands(result.run_id) as { data: any }
  const { data: recessionData } = usePastasRecession(result.run_id) as { data: any }
  const { data: baseflowData } = usePastasBaseflow(result.run_id) as { data: any }
  const { data: spectralData } = usePastasSpectral(result.run_id) as { data: any }
  const { data: decompositionData } = usePastasDecomposition(result.run_id) as { data: any }
  const { data: crossCorrData } = usePastasCrossCorrelation(result.run_id) as { data: any }
  const { data: regionalData } = usePastasRegionalResiduals(result.run_id) as { data: any }
  const { data: inputQualityData } = usePastasInputQuality(result.run_id) as { data: any }
  const [selectedOutlierDate, setSelectedOutlierDate] = useState<string | null>(null)
  const { data: aiModels } = useModels()
  const aiModel = aiModels?.find(m => m.stations?.includes(codeBss ?? '') || m.primary_station === codeBss)

  const {
    metrics, parameters, observed, simulated, residuals,
    contributions, step_response, block_response,
    warnings, validation_metrics, cal_period, val_period,
  } = result

  const hasStepResponse = step_response?.index?.length > 0 && step_response?.values?.length > 0

  return (
    <div className="space-y-3">
      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
          <p className="text-xs font-semibold text-yellow-400 mb-1">Warnings</p>
          {warnings.map((w, i) => <p key={i} className="text-xs text-yellow-300">{w}</p>)}
        </div>
      )}

      {/* AI model cross-link */}
      {aiModel && (
        <div className="flex items-center gap-3 bg-purple-500/10 border border-purple-500/20 rounded-lg px-4 py-2.5">
          <Brain className="w-4 h-4 text-purple-400 shrink-0" />
          <div className="flex-1 min-w-0">
            <span className="text-xs text-purple-300">
              AI model available: <span className="font-medium text-purple-200">{aiModel.model_name}</span>
              {' '}({aiModel.model_type})
              {aiModel.metrics?.NSE != null && <span className="text-text-muted"> — NSE {aiModel.metrics.NSE.toFixed(3)}</span>}
            </span>
          </div>
          <a href={`/ai/forecasting`} className="text-[10px] text-purple-400 hover:text-purple-300 hover:underline shrink-0">
            View forecast →
          </a>
        </div>
      )}

      {/* 1. Metrics */}
      <Section title="Performance Metrics">
        {validation_metrics ? (
          <div className="grid grid-cols-2 gap-3">
            <MetricGrid metrics={metrics} title="Training" period={cal_period} borderColor="border-accent-cyan/20" />
            <MetricGrid metrics={validation_metrics} title="Test (unseen data)" period={val_period} borderColor="border-orange-500/20" />
          </div>
        ) : (
          <MetricGrid metrics={metrics} />
        )}
      </Section>

      {/* Input quality banner */}
      {inputQualityData && inputQualityData.n_flagged > 0 && (
        <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
          <p className="text-xs font-semibold text-orange-400 mb-1">Input Data Anomalies — {inputQualityData.n_flagged} suspicious months detected</p>
          <div className="flex flex-wrap gap-1">
            {inputQualityData.flagged?.map((f: any) => (
              <span key={f.month} className="px-2 py-0.5 rounded-full text-[10px] border border-orange-500/30 bg-orange-500/10 text-orange-300" title={f.reason}>
                {new Date(f.month).toLocaleDateString('en-GB', { year: 'numeric', month: 'short' })}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 2. Time series */}
      <Section title="Time Series — Observed vs Simulated">
        {selectedOutlierDate && (
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[10px] text-orange-400 font-medium">
              Zoomed to outlier: {new Date(selectedOutlierDate).toLocaleDateString('en-GB', { year: 'numeric', month: 'short' })}
            </span>
            <button onClick={() => setSelectedOutlierDate(null)} className="text-[10px] text-accent-cyan hover:underline">Reset zoom</button>
          </div>
        )}
        {observed?.index?.length > 0 && (() => {
          const splitDate = val_period?.[0]

          // Outlier highlight: zoom ±6 months + red band
          const outlierHighlight = selectedOutlierDate ? (() => {
            const d = new Date(selectedOutlierDate)
            const zoomStart = new Date(d); zoomStart.setMonth(zoomStart.getMonth() - 6)
            const zoomEnd = new Date(d); zoomEnd.setMonth(zoomEnd.getMonth() + 7)
            const monthEnd = new Date(d); monthEnd.setMonth(monthEnd.getMonth() + 1)
            return {
              range: [zoomStart.toISOString().slice(0, 10), zoomEnd.toISOString().slice(0, 10)],
              shape: {
                type: 'rect' as const,
                x0: selectedOutlierDate,
                x1: monthEnd.toISOString().slice(0, 10),
                y0: 0, y1: 1, yref: 'paper' as const,
                fillcolor: 'rgba(239,68,68,0.1)',
                line: { color: 'rgba(239,68,68,0.4)', width: 1, dash: 'dot' as const },
              },
            }
          })() : null

          const applyHighlight = (baseLayout: any) => {
            if (!outlierHighlight) return baseLayout
            return {
              ...baseLayout,
              xaxis: { ...(baseLayout.xaxis ?? {}), range: outlierHighlight.range },
              shapes: [...(baseLayout.shapes ?? []), outlierHighlight.shape],
            }
          }

          if (splitDate) {
            const obsTrainX: string[] = [], obsTrainY: number[] = []
            const obsTestX: string[] = [], obsTestY: number[] = []
            observed.index.forEach((d, i) => {
              if (d < splitDate) { obsTrainX.push(d); obsTrainY.push(observed.values[i]) }
              else { obsTestX.push(d); obsTestY.push(observed.values[i]) }
            })

            const simTrainX: string[] = [], simTrainY: number[] = []
            const simTestX: string[] = [], simTestY: number[] = []
            simulated.index.forEach((d, i) => {
              if (d < splitDate) { simTrainX.push(d); simTrainY.push(simulated.values[i]) }
              else { simTestX.push(d); simTestY.push(simulated.values[i]) }
            })

            const trainTraces = [
              ...(confidenceData?.p5 ? [
                { x: confidenceData.index, y: confidenceData.p95, type: 'scatter' as const, mode: 'lines' as const,
                  line: { width: 0 }, showlegend: false },
                { x: confidenceData.index, y: confidenceData.p5, type: 'scatter' as const, mode: 'lines' as const,
                  fill: 'tonexty' as const, fillcolor: 'rgba(34,211,238,0.08)', line: { width: 0 }, name: '90% CI' },
                { x: confidenceData.index, y: confidenceData.p75, type: 'scatter' as const, mode: 'lines' as const,
                  line: { width: 0 }, showlegend: false },
                { x: confidenceData.index, y: confidenceData.p25, type: 'scatter' as const, mode: 'lines' as const,
                  fill: 'tonexty' as const, fillcolor: 'rgba(34,211,238,0.15)', line: { width: 0 }, name: '50% CI' },
              ] : []),
              { x: obsTrainX, y: obsTrainY, type: 'scatter' as const, mode: 'lines' as const,
                name: 'Observed', line: { color: '#6b7280', width: 1 } },
              { x: simTrainX, y: simTrainY, type: 'scatter' as const, mode: 'lines' as const,
                name: 'Simulated', line: { color: '#22d3ee', width: 2 } },
            ]
            const testTraces = [
              { x: obsTestX, y: obsTestY, type: 'scatter' as const, mode: 'lines' as const,
                name: 'Observed', line: { color: '#6b7280', width: 1 }, showlegend: false },
              { x: simTestX, y: simTestY, type: 'scatter' as const, mode: 'lines' as const,
                name: 'Simulated', line: { color: '#f97316', width: 2 } },
            ]

            const outlierInTrain = !selectedOutlierDate || selectedOutlierDate < splitDate
            const outlierInTest = selectedOutlierDate && selectedOutlierDate >= splitDate

            return (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-accent-cyan mb-1">
                    Train {cal_period && <span className="font-normal normal-case text-text-muted">{cal_period[0]} → {cal_period[1]}</span>}
                  </div>
                  <Plot data={trainTraces} layout={outlierInTrain ? applyHighlight({ ...chartLayout, height: 220 }) : { ...chartLayout, height: 220 }} config={plotlyConfig} style={{ width: '100%' }} />
                </div>
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-orange-400 mb-1">
                    Test {val_period && <span className="font-normal normal-case text-text-muted">{val_period[0]} → {val_period[1]}</span>}
                  </div>
                  <Plot data={testTraces} layout={outlierInTest ? applyHighlight({ ...chartLayout, height: 220 }) : { ...chartLayout, height: 220 }} config={plotlyConfig} style={{ width: '100%' }} />
                </div>
              </div>
            )
          }

          // No validation split — single chart
          return (
            <Plot
              data={[
                ...(confidenceData?.p5 ? [
                  { x: confidenceData.index, y: confidenceData.p95, type: 'scatter' as const, mode: 'lines' as const,
                    line: { width: 0 }, showlegend: false },
                  { x: confidenceData.index, y: confidenceData.p5, type: 'scatter' as const, mode: 'lines' as const,
                    fill: 'tonexty' as const, fillcolor: 'rgba(34,211,238,0.08)', line: { width: 0 }, name: '90% CI' },
                  { x: confidenceData.index, y: confidenceData.p75, type: 'scatter' as const, mode: 'lines' as const,
                    line: { width: 0 }, showlegend: false },
                  { x: confidenceData.index, y: confidenceData.p25, type: 'scatter' as const, mode: 'lines' as const,
                    fill: 'tonexty' as const, fillcolor: 'rgba(34,211,238,0.15)', line: { width: 0 }, name: '50% CI' },
                ] : []),
                { x: observed.index, y: observed.values, type: 'scatter' as const, mode: 'lines' as const,
                  name: 'Observed', line: { color: '#6b7280', width: 1 } },
                { x: simulated.index, y: simulated.values, type: 'scatter' as const, mode: 'lines' as const,
                  name: 'Simulated', line: { color: '#22d3ee', width: 2 } },
              ]}
              layout={applyHighlight({ ...chartLayout, height: 280 })}
              config={plotlyConfig} style={{ width: '100%' }}
            />
          )
        })()}
      </Section>

      {/* 3. Contributions */}
      {contributions && Object.keys(contributions).length > 0 && (
        <Section title="Stress Contributions">
          <ContributionsChart contributions={contributions} />
        </Section>
      )}

      {/* 4. Response function */}
      {(hasStepResponse || block_response?.values?.length > 0) && (
        <Section title="Response Function">
          <ResponsePanel stepResponse={step_response} blockResponse={block_response} parameters={parameters} responseType="" />
        </Section>
      )}

      {/* 5. Residuals & diagnostics */}
      <Section title="Residuals & Diagnostics">
        {(() => {
          const calOutliers = outlierData?.cal ?? outlierData
          const valOutliers = outlierData?.val
          const nCal = calOutliers?.n_outliers ?? 0
          const nVal = valOutliers?.n_outliers ?? 0
          const nTotal = nCal + nVal

          return nTotal > 0 ? (
          <div className="flex items-center gap-2 mb-2 text-xs text-text-muted">
            <span className="font-medium text-text-secondary">{nTotal} outliers</span>
            <span>—</span>
            <span className="text-accent-cyan">{nCal} train</span>
            {nVal > 0 && <><span>,</span><span className="text-orange-400">{nVal} test</span></>}
            <span className="text-[10px]">— click a red bar to investigate</span>
          </div>
          ) : null
        })()}
        {residuals?.index?.length > 0 && (() => {
          const vals = residuals.values.filter(v => Number.isFinite(v))
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length
          const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length)
          const threshold = 2 * std

          const calOutliers = outlierData?.cal ?? outlierData
          const valOutliers = outlierData?.val
          const allOutliersList = [
            ...(calOutliers?.outliers ?? []),
            ...(valOutliers?.outliers ?? []),
          ]
          const outlierMonths = new Set<string>(
            allOutliersList.map((o: any) => o.date.slice(0, 7))
          )
          const selectedYM = selectedOutlierDate?.slice(0, 7)

          const barColors = residuals.values.map((_v, i) => {
            const ym = residuals.index[i].slice(0, 7)
            if (outlierMonths.size > 0) {
              if (!outlierMonths.has(ym)) return 'rgba(245,158,11,0.5)'
              return selectedYM === ym ? 'rgba(239,68,68,1.0)' : 'rgba(239,68,68,0.7)'
            }
            if (Math.abs(residuals.values[i]) <= threshold) return 'rgba(245,158,11,0.5)'
            return 'rgba(239,68,68,0.7)'
          })

          return (
            <div className="mb-3">
              <p className="text-xs text-text-muted mb-1">
                {outlierData ? 'Click a red bar to see outlier diagnostics' : 'Red bars = error exceeding 2 standard deviations'}
              </p>
              <Plot
                data={[{
                  x: residuals.index, y: residuals.values, type: 'bar', name: 'Residuals',
                  marker: { color: barColors },
                }]}
                layout={{
                  ...chartLayout, height: 160,
                  shapes: [
                    { type: 'line', x0: residuals.index[0], x1: residuals.index[residuals.index.length - 1], y0: threshold, y1: threshold, line: { color: 'rgba(239,68,68,0.3)', dash: 'dot', width: 1 } },
                    { type: 'line', x0: residuals.index[0], x1: residuals.index[residuals.index.length - 1], y0: -threshold, y1: -threshold, line: { color: 'rgba(239,68,68,0.3)', dash: 'dot', width: 1 } },
                  ],
                }}
                config={plotlyConfig}
                style={{ width: '100%' }}
                onClick={(event: any) => {
                  if (!allOutliersList.length || !event.points?.[0]) return
                  const clickedYM = String(event.points[0].x).slice(0, 7)
                  const match = allOutliersList.find((o: any) => o.date.startsWith(clickedYM))
                  if (!match) return
                  setSelectedOutlierDate((prev: string | null) => prev === match.date ? null : match.date)
                }}
              />
            </div>
          )
        })()}
        {selectedOutlierDate && outlierData && (() => {
          const calO = outlierData?.cal ?? outlierData
          const valO = outlierData?.val
          const allO = [...(calO?.outliers ?? []), ...(valO?.outliers ?? [])]
          const outlier = allO.find((o: any) => o.date === selectedOutlierDate)
          if (!outlier) return null
          return <OutlierDetailPanel outlier={outlier} onClose={() => setSelectedOutlierDate(null)} />
        })()}
        {diagnosticsData && (() => {
          const dd = diagnosticsData as any
          const calDiag = dd.cal ?? dd
          const valDiag = dd.val
          return (
            <div className="space-y-3">
              {calDiag && (
                <div>
                  {valDiag && <div className="text-[10px] font-semibold uppercase tracking-wide text-accent-cyan mb-1">Train diagnostics</div>}
                  <DiagnosticsPanel diagnostics={calDiag} />
                </div>
              )}
              {valDiag && (
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-orange-400 mb-1 mt-3">Test diagnostics</div>
                  <DiagnosticsPanel diagnostics={valDiag} />
                </div>
              )}
            </div>
          )
        })()}
        {!diagnosticsData && <p className="text-xs text-text-muted">Loading diagnostics...</p>}
      </Section>

      {/* 6. Parameters */}
      {parameters.length > 0 && (
        <Section title="Model Parameters" defaultOpen={false}>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-text-muted border-b border-white/5">
                  <th className="text-left px-3 py-2">Name</th>
                  <th className="text-right px-3 py-2">Optimal</th>
                  <th className="text-right px-3 py-2">Std err</th>
                  <th className="text-right px-3 py-2">Initial</th>
                </tr>
              </thead>
              <tbody>
                {parameters.map(p => (
                  <tr key={p.name} className="border-b border-white/5 hover:bg-bg-hover">
                    <td className="px-3 py-2 text-text-primary font-mono">{p.name}</td>
                    <td className="px-3 py-2 text-right text-accent-cyan">{p.optimal.toFixed(6)}</td>
                    <td className="px-3 py-2 text-right text-text-secondary">{p.stderr !== null ? p.stderr.toFixed(6) : '—'}</td>
                    <td className="px-3 py-2 text-right text-text-muted">{p.initial.toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* 7. Signatures */}
      <Section title="Hydrological Signatures" defaultOpen={false}>
        {signaturesData ? (
          <SignaturesPanel signatures={signaturesData} />
        ) : (
          <p className="text-xs text-text-muted">Loading signatures...</p>
        )}
      </Section>

      {/* 8. Recession Analysis */}
      <Section title="Recession Analysis" defaultOpen={false}>
        {recessionData ? <RecessionPanel data={recessionData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      {/* 9. Baseflow Separation */}
      <Section title="Baseflow Separation" defaultOpen={false}>
        {baseflowData ? <BaseflowPanel data={baseflowData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      {/* 10. Spectral Analysis */}
      <Section title="Spectral Analysis" defaultOpen={false}>
        {spectralData ? <SpectralPanel data={spectralData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      {/* 11. Signal Decomposition */}
      <Section title="Signal Decomposition (STL)" defaultOpen={false}>
        {decompositionData ? <DecompositionPanel data={decompositionData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      {/* 12. Cross-Correlation */}
      <Section title="Cross-Correlation (Precip → Piezo)" defaultOpen={false}>
        {crossCorrData ? <CrossCorrelationPanel data={crossCorrData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      {/* 13. Regional Residual Comparison */}
      <Section title="Regional Residual Comparison" defaultOpen={false}>
        {regionalData ? <RegionalResidualsPanel data={regionalData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>
    </div>
  )
}
