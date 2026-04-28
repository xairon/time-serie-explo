import { useState, useMemo } from 'react'
import { ChevronDown, Brain, AlertTriangle, Droplets, Clock, Activity } from 'lucide-react'
import Plot from 'react-plotly.js'
import type { PastasFitResponse } from '@/lib/types'
import { ExportCsvButton } from './ExportCsvButton'
import type { CsvColumn } from '@/lib/csv-export'
import { usePastasDiagnostics, usePastasSignatures, usePastasOutlierDiagnostics, usePastasRecession, usePastasBaseflow, usePastasDecomposition, usePastasInputQuality } from '@/hooks/usePastas'
import { useModels } from '@/hooks/useModels'
import { DiagnosticsPanel } from './DiagnosticsPanel'
import { ResponsePanel } from './ResponsePanel'
import { SignaturesPanel } from './SignaturesPanel'
import { OutlierDetailPanel } from './OutlierDetailPanel'
import { UnifiedAnalysisChart } from './UnifiedAnalysisChart'

// --- Accordion ---

function Section({ title, defaultOpen = true, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="bg-bg-primary rounded-lg border border-white/5 overflow-hidden">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-bg-hover transition-colors">
        <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">{title}</span>
        <ChevronDown className={`w-4 h-4 text-text-muted transition-transform ${open ? '' : '-rotate-90'}`} />
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}

function ControlledSection({ title, open, onToggle, children }: { title: string; open: boolean; onToggle: () => void; children: React.ReactNode }) {
  return (
    <div className="bg-bg-primary rounded-lg border border-white/5 overflow-hidden">
      <button onClick={onToggle} className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-bg-hover transition-colors">
        <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">{title}</span>
        <ChevronDown className={`w-4 h-4 text-text-muted transition-transform ${open ? '' : '-rotate-90'}`} />
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}

// --- Metrics ---

const METRIC_DEFS: Record<string, { label: string; tooltip: string; format: (v: number) => string; quality: (v: number) => 'good' | 'ok' | 'poor' }> = {
  nse: { label: 'NSE', tooltip: 'Nash-Sutcliffe Efficiency — does the model reproduce observed variability? 1 = perfect reproduction, 0 = no better than the mean, negative = worse than the mean. Aim for > 0.7.', format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor' },
  kge: { label: 'KGE', tooltip: 'Kling-Gupta Efficiency — combines correlation, bias, and variability into a single score. More balanced than NSE (which overweights peaks). > 0.7 = good, > 0.9 = excellent.', format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor' },
  evp: { label: 'EVP %', tooltip: 'Explained Variance (%) — what proportion of piezometric signal variability does the model capture? > 70% = acceptable, > 90% = excellent.', format: v => v.toFixed(1), quality: v => v > 70 ? 'good' : v > 40 ? 'ok' : 'poor' },
  rmse: { label: 'RMSE', tooltip: 'Root Mean Square Error (m) — average prediction error in meters. Lower = better. Depends on the natural variability of the aquifer — compare across models for the same station, not between stations.', format: v => v.toFixed(4), quality: () => 'ok' },
  rsq: { label: 'R²', tooltip: 'Coefficient of determination — proportion of signal variance explained by the model. 1 = perfect, 0 = no linear correlation.', format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor' },
  mae: { label: 'MAE', tooltip: 'Mean Absolute Error (m) — average prediction error. Less sensitive to extreme values than RMSE.', format: v => v.toFixed(4), quality: () => 'ok' },
}
const Q_COLORS = { good: 'text-green-400', ok: 'text-accent-cyan', poor: 'text-red-400' }
const Q_BORDERS = { good: 'border-green-500/20', ok: 'border-white/5', poor: 'border-red-500/20' }

function MetricCard({ metricKey, value }: { metricKey: string; value: number | null | undefined }) {
  const def = METRIC_DEFS[metricKey]; if (!def) return null
  const hasValue = value !== null && value !== undefined && Number.isFinite(value)
  const quality = hasValue ? def.quality(value!) : 'ok'
  return (
    <div className={`bg-bg-card rounded-lg p-2.5 border ${Q_BORDERS[quality]}`} title={def.tooltip}>
      <div className="text-[10px] text-text-muted mb-0.5">{def.label}</div>
      <div className={`text-base font-semibold ${hasValue ? Q_COLORS[quality] : 'text-text-muted'}`}>{hasValue ? def.format(value!) : '—'}</div>
    </div>
  )
}

function MetricGrid({ metrics, title, period, borderColor }: { metrics: Record<string, number>; title?: string; period?: string[] | null; borderColor?: string }) {
  return (
    <div className={`rounded-lg border ${borderColor ?? 'border-white/5'} p-3`}>
      {title && (
        <div className={`text-xs font-semibold uppercase tracking-wide mb-2 flex items-center gap-2 ${borderColor?.includes('orange') ? 'text-orange-400' : 'text-accent-cyan'}`}>
          {title}{period && <span className="text-text-muted font-normal normal-case text-[10px]">{period[0]} → {period[1]}</span>}
        </div>
      )}
      <div className="grid grid-cols-3 gap-2">{['nse', 'kge', 'evp', 'rmse', 'rsq', 'mae'].map(k => <MetricCard key={k} metricKey={k} value={metrics[k]} />)}</div>
    </div>
  )
}

// --- Aquifer KPI Card ---

function AquiferKPI({ icon: Icon, label, value, unit, description }: { icon: any; label: string; value: string | null; unit?: string; description: string }) {
  return (
    <div className="bg-bg-card rounded-lg border border-white/5 p-3 flex-1 min-w-0" title={description}>
      <div className="flex items-center gap-2 mb-1.5">
        <Icon className="w-3.5 h-3.5 text-accent-cyan shrink-0" />
        <span className="text-[10px] text-text-muted uppercase tracking-wide truncate">{label}</span>
      </div>
      <div className="text-lg font-semibold text-text-primary font-mono">
        {value ?? '—'}{unit && <span className="text-xs text-text-muted ml-0.5">{unit}</span>}
      </div>
      <p className="text-[9px] text-text-muted mt-1 leading-relaxed">{description}</p>
    </div>
  )
}

// --- Signal Sparkline ---

function SignalSparkline({ data, label, color }: { data: { index: string[]; values: number[] }; label: string; color: string }) {
  if (!data.index.length) return null
  return (
    <div className="flex-1 min-w-0">
      <span className="text-[10px] text-text-muted">{label}</span>
      <Plot
        data={[{ x: data.index, y: data.values, type: 'scatter' as const, mode: 'lines' as const, line: { color, width: 1.2 }, hovertemplate: '%{y:.2f}<extra></extra>' }]}
        layout={{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { size: 8, color: '#6b7280' }, margin: { t: 2, r: 4, b: 14, l: 4 }, height: 50, showlegend: false, xaxis: { showgrid: false, showticklabels: false }, yaxis: { showgrid: false, showticklabels: false } }}
        config={{ displayModeBar: false }} style={{ width: '100%' }}
      />
    </div>
  )
}

type ViewPeriod = 'cal' | 'val' | 'full'

// ═══════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════

interface FitResultsPanelProps { result: PastasFitResponse; codeBss?: string }

export function FitResultsPanel({ result, codeBss }: FitResultsPanelProps) {
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false)
  const [signaturesOpen, setSignaturesOpen] = useState(false)
  const [decompositionOpen, setDecompositionOpen] = useState(false)
  const [paramsOpen, setParamsOpen] = useState(false)

  const { data: diagnosticsData } = usePastasDiagnostics(diagnosticsOpen ? result.run_id : null)
  const { data: signaturesData } = usePastasSignatures(signaturesOpen ? result.run_id : null)
  const { data: outlierData } = usePastasOutlierDiagnostics(result.run_id) as { data: any }
  const { data: recessionData } = usePastasRecession(result.run_id) as { data: any }
  const { data: baseflowData } = usePastasBaseflow(result.run_id) as { data: any }
  const { data: decompositionData } = usePastasDecomposition(decompositionOpen ? result.run_id : null) as { data: any }
  const { data: inputQualityData } = usePastasInputQuality(result.run_id) as { data: any }
  const [selectedOutlierDate, setSelectedOutlierDate] = useState<string | null>(null)
  const [viewPeriod, setViewPeriod] = useState<ViewPeriod>('full')
  const { data: aiModels } = useModels()
  const aiModel = aiModels?.find(m => m.stations?.includes(codeBss ?? '') || m.primary_station === codeBss)

  const { metrics, parameters, observed, simulated, contributions, step_response, block_response, warnings, validation_metrics, cal_period, val_period } = result
  const hasValidation = !!val_period
  const hasStepResponse = step_response?.index?.length > 0 && step_response?.values?.length > 0
  const splitDate = val_period?.[0]

  // --- Slice data by period ---
  const sliceByPeriod = useMemo(() => {
    if (!observed?.index?.length) return { obsX: [], obsY: [], simX: [], simY: [], resIdx: [], resVals: [] }
    let startDate: string | undefined, endDate: string | undefined
    if (viewPeriod === 'cal' && cal_period) { startDate = cal_period[0]; endDate = cal_period[1] }
    else if (viewPeriod === 'val' && val_period) { startDate = val_period[0]; endDate = val_period[1] }
    const inRange = (d: string) => { if (startDate && d < startDate) return false; if (endDate && d > endDate) return false; return true }
    const obsX: string[] = [], obsY: number[] = [], simX: string[] = [], simY: number[] = []
    observed.index.forEach((d, i) => { if (inRange(d)) { obsX.push(d); obsY.push(observed.values[i]) } })
    simulated.index.forEach((d, i) => { if (inRange(d)) { simX.push(d); simY.push(simulated.values[i]) } })
    const resIdx: string[] = [], resVals: number[] = []
    const simMap = new Map<string, number>(); simulated.index.forEach((d, i) => simMap.set(d, simulated.values[i]))
    observed.index.forEach((d, i) => { if (!inRange(d)) return; const sv = simMap.get(d); if (sv !== undefined && Number.isFinite(observed.values[i]) && Number.isFinite(sv)) { resIdx.push(d); resVals.push(observed.values[i] - sv) } })
    return { obsX, obsY, simX, simY, resIdx, resVals }
  }, [observed, simulated, viewPeriod, cal_period, val_period])

  const currentDiagnostics = useMemo(() => { if (!diagnosticsData) return null; const dd = diagnosticsData as any; if (viewPeriod === 'val') return dd.val ?? null; return dd.cal ?? dd }, [diagnosticsData, viewPeriod])

  const currentOutliers = useMemo(() => {
    if (!outlierData) return null
    if (viewPeriod === 'val') return outlierData.val
    if (viewPeriod === 'cal') return outlierData.cal ?? outlierData
    const cal = outlierData.cal ?? outlierData; const val = outlierData.val
    return { ...cal, n_outliers: (cal?.n_outliers ?? 0) + (val?.n_outliers ?? 0), outliers: [...(cal?.outliers ?? []), ...(val?.outliers ?? [])] }
  }, [outlierData, viewPeriod])

  const periodColor = viewPeriod === 'val' ? '#f97316' : '#22d3ee'
  const periodLabel = viewPeriod === 'cal' ? 'Train' : viewPeriod === 'val' ? 'Test' : 'Full period'

  // --- Aquifer KPIs ---
  const t95Days = useMemo(() => {
    if (!step_response?.values?.length) return null
    const vals = step_response.values
    const idx = step_response.index
    const target = 0.95 * vals[vals.length - 1]
    const i = vals.findIndex(v => v >= target)
    if (i < 0 || !idx[i]) return null
    const dayVal = parseFloat(idx[i])
    return Number.isFinite(dayVal) ? Math.round(dayVal) : i
  }, [step_response])

  const monthlyStats = useMemo(() => {
    const { resIdx, resVals } = sliceByPeriod
    if (resIdx.length === 0) return null
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
    return { monthlyDates, monthlyVals, threshold: 2 * std }
  }, [sliceByPeriod])

  const outlierMonths = useMemo(() => {
    const list = currentOutliers?.outliers ?? []
    return new Set<string>(list.map((o: any) => o.date.slice(0, 7)))
  }, [currentOutliers])

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
          <span className="text-xs text-purple-300 flex-1">AI Model: <span className="font-medium text-purple-200">{aiModel.model_name}</span>{aiModel.metrics?.NSE != null && ` — NSE ${aiModel.metrics.NSE.toFixed(3)}`}</span>
          <a href="/ai/forecasting" className="text-[10px] text-purple-400 hover:underline shrink-0">View →</a>
        </div>
      )}

      {/* ═══════════ 1. METRICS ═══════════ */}
      <Section title="Performance">
        {validation_metrics ? (
          <div className="grid grid-cols-2 gap-3">
            <MetricGrid metrics={metrics} title="Train" period={cal_period} borderColor="border-accent-cyan/20" />
            <MetricGrid metrics={validation_metrics} title="Test (unseen)" period={val_period} borderColor="border-orange-500/20" />
          </div>
        ) : <MetricGrid metrics={metrics} />}
      </Section>

      {/* ═══════════ 2. DATA QUALITY BANNER ═══════════ */}
      {inputQualityData && inputQualityData.n_flagged > 0 && (
        <div className="flex items-center gap-2 bg-orange-500/5 border border-orange-500/20 rounded-lg px-3 py-2 overflow-x-auto">
          <AlertTriangle className="w-3.5 h-3.5 text-orange-400 shrink-0" />
          <span className="text-[10px] text-orange-400 font-medium shrink-0">{inputQualityData.n_flagged} suspect months:</span>
          <div className="flex gap-1 flex-nowrap">
            {inputQualityData.flagged?.map((f: any) => (
              <button
                key={f.month}
                onClick={() => setSelectedOutlierDate(f.month)}
                title={f.reason}
                className="px-1.5 py-0.5 rounded text-[9px] font-mono border border-orange-500/20 bg-orange-500/10 text-orange-300 hover:bg-orange-500/20 transition-colors whitespace-nowrap shrink-0"
              >
                {new Date(f.month).toLocaleDateString('en-GB', { year: '2-digit', month: 'short' })}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ═══════════ 3. PERIOD TOGGLE ═══════════ */}
      {hasValidation && (
        <div className="flex items-center gap-1 bg-bg-primary rounded-lg border border-white/5 p-1">
          {(['cal', 'val', 'full'] as ViewPeriod[]).map(p => {
            const labels: Record<ViewPeriod, string> = { cal: 'Train', val: 'Test', full: 'Full period' }
            const colors: Record<ViewPeriod, string> = { cal: 'text-accent-cyan', val: 'text-orange-400', full: 'text-text-primary' }
            return (
              <button key={p} onClick={() => { setViewPeriod(p); setSelectedOutlierDate(null) }}
                className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${viewPeriod === p ? `bg-bg-hover ${colors[p]}` : 'text-text-muted hover:text-text-secondary'}`}>
                {labels[p]}
                {p === 'cal' && cal_period && <span className="text-[9px] text-text-muted ml-1 font-normal">{cal_period[0].slice(0, 4)}–{cal_period[1].slice(0, 4)}</span>}
                {p === 'val' && val_period && <span className="text-[9px] text-text-muted ml-1 font-normal">{val_period[0].slice(0, 4)}–{val_period[1].slice(0, 4)}</span>}
              </button>
            )
          })}
        </div>
      )}

      {/* ═══════════ 4. MODEL ANALYSIS (unified chart) ═══════════ */}
      <Section title={`Model Analysis — ${periodLabel}`}>
        <div className="flex items-center gap-1 mb-2">
          <ExportCsvButton
            filename={`${result.code_bss}_observed_simulated_residuals.csv`}
            title="Export observed, simulated & residuals as CSV"
            label="Time series"
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
            label="Contributions"
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
        {selectedOutlierDate && (
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[10px] text-orange-400 font-medium">Zoomed on: {new Date(selectedOutlierDate).toLocaleDateString('en-GB', { year: 'numeric', month: 'short' })}</span>
            <button onClick={() => setSelectedOutlierDate(null)} className="text-[10px] text-accent-cyan hover:underline">Reset zoom</button>
          </div>
        )}
        {currentOutliers && currentOutliers.n_outliers > 0 && (
          <p className="text-[10px] text-text-muted mb-1">{currentOutliers.n_outliers} outlier months — click a red bar to investigate</p>
        )}
        {monthlyStats && sliceByPeriod.obsX.length > 0 && (
          <>
            <UnifiedAnalysisChart
              obsX={sliceByPeriod.obsX} obsY={sliceByPeriod.obsY}
              simX={sliceByPeriod.simX} simY={sliceByPeriod.simY}
              contributions={contributions}
              monthlyDates={monthlyStats.monthlyDates}
              monthlyResiduals={monthlyStats.monthlyVals}
              threshold={monthlyStats.threshold}
              outlierMonths={outlierMonths}
              selectedOutlierDate={selectedOutlierDate}
              onOutlierClick={(ym) => {
                const allOutliersList = currentOutliers?.outliers ?? []
                const m = allOutliersList.find((o: any) => o.date.startsWith(ym))
                if (m) setSelectedOutlierDate(prev => prev === m.date ? null : m.date)
              }}
              periodColor={periodColor}
              splitDate={splitDate}
              viewPeriod={viewPeriod}
              selectedOutlierHighlight={selectedOutlierDate ? (() => {
                const d = new Date(selectedOutlierDate)
                const zs = new Date(d); zs.setMonth(zs.getMonth() - 6)
                const ze = new Date(d); ze.setMonth(ze.getMonth() + 7)
                const me = new Date(d); me.setMonth(me.getMonth() + 1)
                return {
                  range: [zs.toISOString().slice(0, 10), ze.toISOString().slice(0, 10)] as [string, string],
                  shape: {
                    type: 'rect' as const, x0: selectedOutlierDate, x1: me.toISOString().slice(0, 10),
                    y0: 0, y1: 1, yref: 'paper' as const,
                    fillcolor: 'rgba(239,68,68,0.08)', line: { color: 'rgba(239,68,68,0.3)', width: 1, dash: 'dot' as const }
                  }
                }
              })() : null}
            />
          </>
        )}
        {selectedOutlierDate && currentOutliers && (() => {
          const outlier = currentOutliers.outliers?.find((o: any) => o.date === selectedOutlierDate)
          return outlier ? <OutlierDetailPanel outlier={outlier} onClose={() => setSelectedOutlierDate(null)} /> : null
        })()}
      </Section>

      {/* ═══════════ 5. AQUIFER CHARACTERIZATION ═══════════ */}
      <Section title="Aquifer Characterization">
        <div className="grid grid-cols-3 gap-2 mb-3">
          <AquiferKPI icon={Clock} label="Response Time" value={t95Days != null ? String(t95Days) : null} unit="days"
            description={`T95 — time for the aquifer to reach 95% of its response to a recharge change. ${t95Days != null && t95Days < 100 ? 'Fast response (unconfined alluvial, shallow aquifer).' : t95Days != null && t95Days < 500 ? 'Moderate response (shallow sedimentary).' : t95Days != null ? 'Slow response (deep sedimentary, confined).' : ''}`} />
          <AquiferKPI icon={Droplets} label="Baseflow Index" value={baseflowData?.bfi != null ? baseflowData.bfi.toFixed(2) : null}
            description={`Proportion of the piezometric signal due to baseflow (slow) vs fast rainfall response. ${baseflowData?.bfi > 0.7 ? 'Inertial aquifer, dominated by slow flow.' : baseflowData?.bfi > 0.4 ? 'Mixed response (slow flow + fast response).' : 'Highly reactive aquifer to precipitation.'}`} />
          <AquiferKPI icon={Activity} label="Recession Constant" value={recessionData?.T_median_days != null ? String(recessionData.T_median_days) : null} unit="days"
            description={`Median aquifer drainage duration during recession periods. The longer T, the longer the aquifer retains water. ${recessionData?.n_segments ? `Computed from ${recessionData.n_segments} recession episodes.` : ''}`} />
        </div>
        {hasStepResponse && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-3">
            <p className="text-[10px] text-text-muted mb-1">Response function — how the aquifer transforms an input signal (recharge) into piezometric level variation. Dashed lines mark 50% (t50) and 95% (t95) of the final response.</p>
            <ResponsePanel stepResponse={step_response} blockResponse={block_response} parameters={parameters} responseType="" />
          </div>
        )}
      </Section>

      {/* ═══════════ 6. SIGNAL STRUCTURE ═══════════ */}
      <ControlledSection title="Signal Structure" open={decompositionOpen} onToggle={() => setDecompositionOpen(v => !v)}>
        {decompositionData?.index?.length > 0 ? (
          <>
            <p className="text-[10px] text-text-muted mb-2 leading-relaxed">
              Decomposition of the piezometric signal into independent components (STL method).
              The <span className="text-accent-cyan">trend</span> represents the long-term level evolution (recharge, climate, pumping).
              The <span className="text-purple-400">seasonality</span> represents the annual cycle (winter recharge, summer low water).
              The residual (not shown) is what the model cannot explain with these two components.
            </p>
            <div className="flex items-center gap-4 mb-2">
              <span className="text-[10px] text-text-muted" title="Proportion of signal variance explained by the trend. Close to 1 = dominant trend (long-term decline or rise). Close to 0 = no marked trend.">
                Trend strength: <span className="text-text-primary font-mono font-semibold">{decompositionData.trend_strength}</span>
              </span>
              <span className="text-[10px] text-text-muted" title="Proportion of signal variance explained by the annual cycle. Close to 1 = highly seasonal aquifer (typical unconfined alluvial). Low = high-inertia or confined aquifer.">
                Seasonal strength: <span className="text-text-primary font-mono font-semibold">{decompositionData.seasonal_strength}</span>
              </span>
              <span className="text-[9px] text-text-muted ml-auto">0 = absent component, 1 = dominant component</span>
            </div>
            <div className="flex gap-3">
              <SignalSparkline data={{ index: decompositionData.index, values: decompositionData.trend }} label="Long-term Trend" color="#22d3ee" />
              <SignalSparkline data={{ index: decompositionData.index, values: decompositionData.seasonal }} label="Seasonal Cycle" color="#a78bfa" />
            </div>
          </>
        ) : <p className="text-xs text-text-muted">Loading...</p>}
      </ControlledSection>

      {/* ═══════════ 7. STATISTICAL DIAGNOSTICS ═══════════ */}
      <ControlledSection title="Statistical Diagnostics" open={diagnosticsOpen} onToggle={() => setDiagnosticsOpen(v => !v)}>
        {currentDiagnostics ? <DiagnosticsPanel diagnostics={currentDiagnostics} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </ControlledSection>

      {/* ═══════════ 8. MODEL INFO ═══════════ */}
      {parameters.length > 0 && (
        <ControlledSection title="Model Parameters" open={paramsOpen} onToggle={() => setParamsOpen(v => !v)}>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead><tr className="text-text-muted border-b border-white/5">
                <th className="text-left px-3 py-2">Name</th><th className="text-right px-3 py-2">Optimal</th>
                <th className="text-right px-3 py-2">Std. err</th><th className="text-right px-3 py-2">Initial</th>
              </tr></thead>
              <tbody>{parameters.map(p => (
                <tr key={p.name} className="border-b border-white/5 hover:bg-bg-hover">
                  <td className="px-3 py-2 text-text-primary font-mono">{p.name}</td>
                  <td className="px-3 py-2 text-right text-accent-cyan">{p.optimal.toFixed(6)}</td>
                  <td className="px-3 py-2 text-right text-text-secondary">{p.stderr !== null ? p.stderr.toFixed(6) : '—'}</td>
                  <td className="px-3 py-2 text-right text-text-muted">{p.initial.toFixed(6)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </ControlledSection>
      )}

      <ControlledSection title="Hydrological Signatures" open={signaturesOpen} onToggle={() => setSignaturesOpen(v => !v)}>
        {signaturesData ? <SignaturesPanel signatures={signaturesData} /> : <p className="text-xs text-text-muted">Loading...</p>}
      </ControlledSection>
    </div>
  )
}
