import { useState, useMemo } from 'react'
import { ChevronDown, Brain } from 'lucide-react'
import type { PastasFitResponse } from '@/lib/types'
import { usePastasDiagnostics, usePastasSignatures, usePastasOutlierDiagnostics, usePastasRecession, usePastasBaseflow, usePastasSpectral, usePastasDecomposition, usePastasCrossCorrelation, usePastasRegionalResiduals, usePastasInputQuality } from '@/hooks/usePastas'
import { useModels } from '@/hooks/useModels'
import { DiagnosticsPanel } from './DiagnosticsPanel'
import { ResponsePanel } from './ResponsePanel'
import { SignaturesPanel } from './SignaturesPanel'
import { OutlierDetailPanel } from './OutlierDetailPanel'
import { UnifiedAnalysisChart } from './UnifiedAnalysisChart'
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
  nse: { label: 'NSE', tooltip: 'Nash-Sutcliffe Efficiency. 1 = perfect, 0 = as good as mean, <0 = worse.', format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor' },
  kge: { label: 'KGE', tooltip: 'Kling-Gupta Efficiency. Combines correlation, bias and variability. >0.7 = good.', format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor' },
  evp: { label: 'EVP (%)', tooltip: 'Explained Variance Percentage. 100% = fully explained variance.', format: v => v.toFixed(1), quality: v => v > 70 ? 'good' : v > 40 ? 'ok' : 'poor' },
  rmse: { label: 'RMSE', tooltip: 'Root Mean Square Error (m). Lower is better.', format: v => v.toFixed(4), quality: () => 'ok' },
  rsq: { label: 'R²', tooltip: 'Coefficient of determination. 1 = perfect correlation.', format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor' },
  mae: { label: 'MAE', tooltip: 'Mean Absolute Error (m). Average absolute error.', format: v => v.toFixed(4), quality: () => 'ok' },
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
      <div className="text-[10px] text-text-muted mb-0.5 flex items-center gap-1">{def.label}<span className="cursor-help opacity-50">ⓘ</span></div>
      <div className={`text-base font-semibold ${hasValue ? Q_COLORS[quality] : 'text-text-muted'}`}>{hasValue ? def.format(value!) : '—'}</div>
    </div>
  )
}

function MetricGrid({ metrics, title, period, borderColor }: { metrics: Record<string, number>; title?: string; period?: string[] | null; borderColor?: string }) {
  return (
    <div className={`rounded-lg border ${borderColor ?? 'border-white/5'} p-3`}>
      {title && (
        <div className={`text-xs font-semibold uppercase tracking-wide mb-2 flex items-center gap-2 ${borderColor?.includes('orange') ? 'text-orange-400' : 'text-accent-cyan'}`}>
          {title}
          {period && <span className="text-text-muted font-normal normal-case text-[10px]">{period[0]} → {period[1]}</span>}
        </div>
      )}
      <div className="grid grid-cols-3 gap-2">
        {['nse', 'kge', 'evp', 'rmse', 'rsq', 'mae'].map(k => <MetricCard key={k} metricKey={k} value={metrics[k]} />)}
      </div>
    </div>
  )
}

type ViewPeriod = 'cal' | 'val' | 'full'

// --- Main component ---

interface FitResultsPanelProps { result: PastasFitResponse; codeBss?: string }

export function FitResultsPanel({ result, codeBss }: FitResultsPanelProps) {
  const { data: diagnosticsData } = usePastasDiagnostics(result.run_id)
  const { data: signaturesData } = usePastasSignatures(result.run_id)
  const { data: outlierData } = usePastasOutlierDiagnostics(result.run_id) as { data: any }
  const { data: recessionData } = usePastasRecession(result.run_id) as { data: any }
  const { data: baseflowData } = usePastasBaseflow(result.run_id) as { data: any }
  const { data: spectralData } = usePastasSpectral(result.run_id) as { data: any }
  const { data: decompositionData } = usePastasDecomposition(result.run_id) as { data: any }
  const { data: crossCorrData } = usePastasCrossCorrelation(result.run_id) as { data: any }
  const { data: regionalData } = usePastasRegionalResiduals(result.run_id) as { data: any }
  const { data: inputQualityData } = usePastasInputQuality(result.run_id) as { data: any }
  const [selectedOutlierDate, setSelectedOutlierDate] = useState<string | null>(null)
  const [viewPeriod, setViewPeriod] = useState<ViewPeriod>('full')
  const { data: aiModels } = useModels()
  const aiModel = aiModels?.find(m => m.stations?.includes(codeBss ?? '') || m.primary_station === codeBss)

  const { metrics, parameters, observed, simulated, residuals, contributions, step_response, block_response, warnings, validation_metrics, cal_period, val_period } = result
  const hasValidation = !!val_period
  const hasStepResponse = step_response?.index?.length > 0 && step_response?.values?.length > 0
  const splitDate = val_period?.[0]

  // --- Slice data by period ---
  const sliceByPeriod = useMemo(() => {
    if (!observed?.index?.length) return { obsX: [], obsY: [], simX: [], simY: [], resIdx: [], resVals: [] }

    let startDate: string | undefined
    let endDate: string | undefined
    if (viewPeriod === 'cal' && cal_period) { startDate = cal_period[0]; endDate = cal_period[1] }
    else if (viewPeriod === 'val' && val_period) { startDate = val_period[0]; endDate = val_period[1] }

    const inRange = (d: string) => {
      if (startDate && d < startDate) return false
      if (endDate && d > endDate) return false
      return true
    }

    const obsX: string[] = [], obsY: number[] = []
    observed.index.forEach((d, i) => { if (inRange(d)) { obsX.push(d); obsY.push(observed.values[i]) } })

    const simX: string[] = [], simY: number[] = []
    simulated.index.forEach((d, i) => { if (inRange(d)) { simX.push(d); simY.push(simulated.values[i]) } })

    // Compute residuals from obs - sim (aligned by date) so they cover val period too
    const resIdx: string[] = [], resVals: number[] = []
    const simMap = new Map<string, number>()
    simulated.index.forEach((d, i) => simMap.set(d, simulated.values[i]))
    observed.index.forEach((d, i) => {
      if (!inRange(d)) return
      const sv = simMap.get(d)
      if (sv !== undefined && Number.isFinite(observed.values[i]) && Number.isFinite(sv)) {
        resIdx.push(d)
        resVals.push(observed.values[i] - sv)
      }
    })

    return { obsX, obsY, simX, simY, resIdx, resVals }
  }, [observed, simulated, residuals, viewPeriod, cal_period, val_period])

  // --- Get diagnostics for current period ---
  const currentDiagnostics = useMemo(() => {
    if (!diagnosticsData) return null
    const dd = diagnosticsData as any
    if (viewPeriod === 'val') return dd.val ?? null
    if (viewPeriod === 'cal') return dd.cal ?? dd
    return dd.cal ?? dd
  }, [diagnosticsData, viewPeriod])

  const currentOutliers = useMemo(() => {
    if (!outlierData) return null
    if (viewPeriod === 'val') return outlierData.val
    if (viewPeriod === 'cal') return outlierData.cal ?? outlierData
    const cal = outlierData.cal ?? outlierData
    const val = outlierData.val
    return {
      ...cal,
      n_outliers: (cal?.n_outliers ?? 0) + (val?.n_outliers ?? 0),
      outliers: [...(cal?.outliers ?? []), ...(val?.outliers ?? [])],
    }
  }, [outlierData, viewPeriod])

  const currentSpectral = useMemo(() => {
    if (!spectralData) return null
    if (viewPeriod === 'val') return spectralData.val
    if (viewPeriod === 'cal') return spectralData.cal ?? spectralData
    return spectralData.cal ?? spectralData
  }, [spectralData, viewPeriod])

  const currentRegional = useMemo(() => {
    if (!regionalData) return null
    if (viewPeriod === 'val') return regionalData.val
    if (viewPeriod === 'cal') return regionalData.cal ?? regionalData
    return regionalData.cal ?? regionalData
  }, [regionalData, viewPeriod])

  // --- Period color ---
  const periodColor = viewPeriod === 'val' ? '#f97316' : '#22d3ee'
  const periodLabel = viewPeriod === 'cal' ? 'Train' : viewPeriod === 'val' ? 'Test' : 'Full period'

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
            <span className="text-xs text-purple-300">AI model: <span className="font-medium text-purple-200">{aiModel.model_name}</span> ({aiModel.model_type}){aiModel.metrics?.NSE != null && <span className="text-text-muted"> — NSE {aiModel.metrics.NSE.toFixed(3)}</span>}</span>
          </div>
          <a href="/ai/forecasting" className="text-[10px] text-purple-400 hover:underline shrink-0">View forecast →</a>
        </div>
      )}

      {/* ═══════════ METRICS ═══════════ */}
      <Section title="Performance Metrics">
        {validation_metrics ? (
          <div className="grid grid-cols-2 gap-3">
            <MetricGrid metrics={metrics} title="Train" period={cal_period} borderColor="border-accent-cyan/20" />
            <MetricGrid metrics={validation_metrics} title="Test (unseen data)" period={val_period} borderColor="border-orange-500/20" />
          </div>
        ) : (
          <MetricGrid metrics={metrics} />
        )}
      </Section>

      {/* ═══════════ PERIOD TOGGLE ═══════════ */}
      {hasValidation && (
        <div className="flex items-center gap-1 bg-bg-primary rounded-lg border border-white/5 p-1">
          {(['cal', 'val', 'full'] as ViewPeriod[]).map(p => {
            const labels: Record<ViewPeriod, string> = { cal: 'Train', val: 'Test', full: 'Full period' }
            const colors: Record<ViewPeriod, string> = { cal: 'text-accent-cyan', val: 'text-orange-400', full: 'text-text-primary' }
            return (
              <button
                key={p}
                onClick={() => { setViewPeriod(p); setSelectedOutlierDate(null) }}
                className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${viewPeriod === p ? `bg-bg-hover ${colors[p]}` : 'text-text-muted hover:text-text-secondary'}`}
              >
                {labels[p]}
                {p === 'cal' && cal_period && <span className="text-[9px] text-text-muted ml-1.5 font-normal">{cal_period[0].slice(0, 4)}–{cal_period[1].slice(0, 4)}</span>}
                {p === 'val' && val_period && <span className="text-[9px] text-text-muted ml-1.5 font-normal">{val_period[0].slice(0, 4)}–{val_period[1].slice(0, 4)}</span>}
              </button>
            )
          })}
        </div>
      )}

      {/* ═══════════ UNIFIED MODEL ANALYSIS ═══════════ */}

      <Section title={`Model Analysis — ${periodLabel}`}>
        {selectedOutlierDate && (
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[10px] text-orange-400 font-medium">Zoomed to: {new Date(selectedOutlierDate).toLocaleDateString('en-GB', { year: 'numeric', month: 'short' })}</span>
            <button onClick={() => setSelectedOutlierDate(null)} className="text-[10px] text-accent-cyan hover:underline">Reset zoom</button>
          </div>
        )}
        {currentOutliers && currentOutliers.n_outliers > 0 && (
          <div className="flex items-center gap-2 mb-1 text-xs text-text-muted">
            <span className="font-medium text-text-secondary">{currentOutliers.n_outliers} outlier months</span>
            <span className="text-[10px]">— click a red bar in the error panel to investigate</span>
          </div>
        )}
        {sliceByPeriod.obsX.length > 0 && (() => {
          const { obsX, obsY, simX, simY, resIdx, resVals } = sliceByPeriod

          // Aggregate residuals to monthly
          const monthlyMap = new Map<string, number[]>()
          resIdx.forEach((d, i) => {
            const ym = d.slice(0, 7)
            if (!monthlyMap.has(ym)) monthlyMap.set(ym, [])
            if (Number.isFinite(resVals[i])) monthlyMap.get(ym)!.push(resVals[i])
          })
          const monthlyDates: string[] = []
          const monthlyVals: number[] = []
          for (const [ym, vals] of monthlyMap) {
            monthlyDates.push(ym + '-15')
            monthlyVals.push(vals.reduce((a, b) => a + b, 0) / vals.length)
          }
          const finiteVals = monthlyVals.filter(v => Number.isFinite(v))
          const mean = finiteVals.length > 0 ? finiteVals.reduce((a, b) => a + b, 0) / finiteVals.length : 0
          const std = finiteVals.length > 1 ? Math.sqrt(finiteVals.reduce((a, b) => a + (b - mean) ** 2, 0) / finiteVals.length) : 1
          const threshold = 2 * std

          const allOutliersList = currentOutliers?.outliers ?? []
          const outlierMonths = new Set<string>(allOutliersList.map((o: any) => o.date.slice(0, 7)))

          // Outlier highlight
          const outlierHighlight = selectedOutlierDate ? (() => {
            const d = new Date(selectedOutlierDate)
            const zoomStart = new Date(d); zoomStart.setMonth(zoomStart.getMonth() - 6)
            const zoomEnd = new Date(d); zoomEnd.setMonth(zoomEnd.getMonth() + 7)
            const monthEnd = new Date(d); monthEnd.setMonth(monthEnd.getMonth() + 1)
            return {
              range: [zoomStart.toISOString().slice(0, 10), zoomEnd.toISOString().slice(0, 10)] as [string, string],
              shape: { type: 'rect' as const, x0: selectedOutlierDate, x1: monthEnd.toISOString().slice(0, 10), y0: 0, y1: 1, yref: 'paper' as const, fillcolor: 'rgba(239,68,68,0.08)', line: { color: 'rgba(239,68,68,0.3)', width: 1, dash: 'dot' as const } },
            }
          })() : null

          return (
            <UnifiedAnalysisChart
              obsX={obsX} obsY={obsY} simX={simX} simY={simY}
              contributions={contributions}
              monthlyDates={monthlyDates} monthlyResiduals={monthlyVals} threshold={threshold}
              outlierMonths={outlierMonths}
              selectedOutlierDate={selectedOutlierDate}
              onOutlierClick={(clickedYM) => {
                const match = allOutliersList.find((o: any) => o.date.startsWith(clickedYM))
                if (match) setSelectedOutlierDate(prev => prev === match.date ? null : match.date)
              }}
              periodColor={periodColor}
              splitDate={splitDate}
              viewPeriod={viewPeriod}
              selectedOutlierHighlight={outlierHighlight}
            />
          )
        })()}
        {selectedOutlierDate && currentOutliers && (() => {
          const outlier = currentOutliers.outliers?.find((o: any) => o.date === selectedOutlierDate)
          if (!outlier) return null
          return <OutlierDetailPanel outlier={outlier} onClose={() => setSelectedOutlierDate(null)} />
        })()}
      </Section>

      {/* Statistical Diagnostics */}
      <Section title={`Statistical Diagnostics — ${periodLabel}`}>
        {currentDiagnostics ? <DiagnosticsPanel diagnostics={currentDiagnostics} /> : <p className="text-xs text-text-muted">Loading diagnostics...</p>}
      </Section>

      {/* Spectral Analysis */}
      <Section title={`Spectral Analysis — ${periodLabel}`} defaultOpen={false}>
        {currentSpectral ? <SpectralPanel data={currentSpectral} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      {/* Regional Comparison */}
      <Section title={`Regional Residual Comparison — ${periodLabel}`} defaultOpen={false}>
        {currentRegional ? <RegionalResidualsPanel data={currentRegional} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      {/* ═══════════ ALWAYS FULL PERIOD ═══════════ */}

      {(hasStepResponse || block_response?.values?.length > 0) && (
        <Section title="Response Function" defaultOpen={false}>
          <p className="text-[10px] text-text-muted mb-2">How the aquifer responds to a unit input of recharge. The shape reveals the aquifer's memory: a steep curve = fast response (alluvial), a slow curve = high inertia (deep sedimentary). The time to reach 95% of the response (T95) is the key indicator.</p>
          <ResponsePanel stepResponse={step_response} blockResponse={block_response} parameters={parameters} responseType="" />
        </Section>
      )}

      <Section title="Recession Analysis" defaultOpen={false}>
        <p className="text-[10px] text-text-muted mb-2">Identifies falling water table periods and fits exponential decay h(t) = h0·e^(-t/T). The time constant T (in days) measures how fast the aquifer drains — small T = fast drainage (alluvial), large T = slow drainage (confined). The Master Recession Curve (MRC) is the average of all normalized recessions.</p>
        {recessionData ? <RecessionPanel data={recessionData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      <Section title="Baseflow Separation" defaultOpen={false}>
        <p className="text-[10px] text-text-muted mb-2">Separates the water table signal into a slow component (baseflow — long-term storage response) and a fast component (quickflow — event-driven recharge pulses). BFI close to 1 = the aquifer is dominated by slow, steady flow. BFI close to 0 = dominated by rapid infiltration events.</p>
        {baseflowData ? <BaseflowPanel data={baseflowData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      <Section title="Signal Decomposition (STL)" defaultOpen={false}>
        <p className="text-[10px] text-text-muted mb-2">Breaks the observed water level into three components: Trend (long-term direction — rising or falling over years), Seasonal (annual cycle — high in winter, low in summer for most aquifers), and Residual (what's left — noise, events, anomalies). Strength values near 1.0 mean that component strongly drives the signal.</p>
        {decompositionData ? <DecompositionPanel data={decompositionData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      <Section title="Cross-Correlation (Precip → Piezo)" defaultOpen={false}>
        <p className="text-[10px] text-text-muted mb-2">Measures the empirical delay between rainfall and water table response. The peak lag (in months) shows how long it takes for rain to reach the aquifer. Compare with the model's T95 (from the response function) — if they disagree significantly, the model may have the wrong response timescale.</p>
        {crossCorrData ? <CrossCorrelationPanel data={crossCorrData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>

      {/* Input Data Anomalies */}
      {inputQualityData && (
        <Section title={`Input Data Quality — ${inputQualityData.n_flagged ?? 0} anomalies`} defaultOpen={false}>
          <p className="text-[10px] text-text-muted mb-2">AI-based scan (Isolation Forest) of the input data to detect months where climate or piezometric values are statistically unusual. Flagged months may indicate sensor errors, data gaps, or extreme events that could affect model calibration. Click a month to zoom the time series.</p>
          {inputQualityData.n_flagged > 0 ? (
            <div className="space-y-2">
              {inputQualityData.flagged?.map((f: any) => (
                <button
                  key={f.month}
                  onClick={() => setSelectedOutlierDate(f.month)}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-lg border border-orange-500/20 bg-orange-500/5 hover:bg-orange-500/10 transition-colors text-left"
                >
                  <div>
                    <span className="text-xs font-medium text-orange-400">
                      {new Date(f.month).toLocaleDateString('en-GB', { year: 'numeric', month: 'long' })}
                    </span>
                    <p className="text-[10px] text-text-muted mt-0.5">{f.reason}</p>
                  </div>
                  <span className="text-[10px] font-mono text-text-muted shrink-0 ml-2">score: {f.score}</span>
                </button>
              ))}
            </div>
          ) : (
            <p className="text-xs text-green-400">No anomalies detected in input data</p>
          )}
        </Section>
      )}

      {/* ═══════════ MODEL INFO ═══════════ */}

      {parameters.length > 0 && (
        <Section title="Model Parameters" defaultOpen={false}>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead><tr className="text-text-muted border-b border-white/5">
                <th className="text-left px-3 py-2">Name</th><th className="text-right px-3 py-2">Optimal</th>
                <th className="text-right px-3 py-2">Std err</th><th className="text-right px-3 py-2">Initial</th>
              </tr></thead>
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

      <Section title="Hydrological Signatures" defaultOpen={false}>
        {signaturesData ? <SignaturesPanel signatures={signaturesData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </Section>
    </div>
  )
}
