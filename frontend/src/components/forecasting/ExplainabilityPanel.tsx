import { useState } from 'react'
import {
  TrendingUp,
  Activity,
  Clock,
  Sun,
  Beaker,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
} from 'lucide-react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import {
  useFeatureImportance,
  usePermutationImportance,
  useShapAnalysis,
  useGradientAnalysis,
  useLagImportance,
  useResidualAnalysis,
  useSeasonalityAnalysis,
} from '@/hooks/useForecasting'
import type { ExplainResult } from '@/lib/types'
import { InfoTip } from '@/components/pastas/InfoTip'

interface Props {
  modelId: string
  className?: string
}

type Section = 'drivers' | 'quality' | 'behavior' | 'expert'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function LoadingSkeleton() {
  return <div className="h-[200px] bg-bg-hover rounded-lg animate-pulse" />
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="text-center py-6">
      <p className="text-xs text-red-400 mb-2">{message}</p>
      <button onClick={onRetry} className="text-xs text-accent-cyan hover:underline">Retry</button>
    </div>
  )
}

function extractImportance(data: ExplainResult): { features: string[]; values: number[] } | null {
  if (!data.feature_importance) return null
  const entries = Object.entries(data.feature_importance)
    .filter(([, v]) => v != null)
    .map(([k, v]) => [k, v as number] as const)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
  if (entries.length === 0) return null
  return { features: entries.map(([k]) => k), values: entries.map(([, v]) => v) }
}

const HYDRO_LABELS: Record<string, string> = {
  total_precipitation: 'Precipitation',
  potential_evaporation: 'Evapotranspiration (PET)',
  temperature_2m: 'Temperature',
  niveau_nappe_eau: 'Water level (autoregressive)',
}

function hydroLabel(name: string): string {
  return HYDRO_LABELS[name] ?? name.replace(/_/g, ' ')
}

function influenceColor(pct: number): string {
  if (pct > 40) return '#06b6d4'
  if (pct > 15) return '#eab308'
  return '#6b7280'
}

function SectionHeader({ icon: Icon, title, tip, open, onToggle }: {
  icon: React.ElementType; title: string; tip: string; open: boolean; onToggle: () => void
}) {
  return (
    <button onClick={onToggle} className="w-full flex items-center gap-2 py-2 group">
      <Icon className="w-4 h-4 text-accent-cyan shrink-0" />
      <span className="text-sm font-semibold text-text-primary">{title}</span>
      <InfoTip text={tip} />
      <ChevronDown className={`w-3.5 h-3.5 text-text-muted ml-auto transition-transform ${open ? '' : '-rotate-90'}`} />
    </button>
  )
}

// ---------------------------------------------------------------------------
// Section 1: What drives predictions
// ---------------------------------------------------------------------------

function DriversSection({ modelId }: { modelId: string }) {
  const mutation = useFeatureImportance()
  if (!mutation.data && !mutation.isPending && !mutation.isError) mutation.mutate(modelId)
  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError) return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const importance = extractImportance(mutation.data)
  if (!importance) return <p className="text-xs text-text-muted py-4">No feature importance data available.</p>

  const total = importance.values.reduce((s, v) => s + Math.abs(v), 0)
  const features = importance.features.map(hydroLabel)
  const pcts = importance.values.map(v => total > 0 ? Math.abs(v) / total * 100 : 0)
  const colors = pcts.map(influenceColor)

  return (
    <div className="space-y-3">
      <div className="h-[220px]">
        <Plot
          data={[{
            type: 'bar', orientation: 'h' as const,
            y: features, x: importance.values.map(Math.abs),
            marker: { color: colors },
            hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
          }]}
          layout={{
            ...darkLayout,
            xaxis: { ...darkLayout.xaxis, title: { text: 'Correlation strength' } },
            yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const },
            margin: { t: 5, r: 20, b: 35, l: 160 },
          }}
          config={plotlyConfig}
          useResizeHandler
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      <div className="bg-bg-hover rounded-lg p-3 space-y-1.5">
        {importance.features.slice(0, 4).map((feat, i) => {
          const pct = pcts[i]
          const label = hydroLabel(feat)
          const isTarget = feat === 'niveau_nappe_eau'
          return (
            <p key={feat} className="text-xs text-text-secondary">
              <span className="font-medium" style={{ color: colors[i] }}>{label}</span>
              {' '}accounts for <span className="text-text-primary font-mono">{pct.toFixed(0)}%</span> of the signal.
              {isTarget && ' The water level is strongly autocorrelated — recent levels predict future levels.'}
              {feat === 'total_precipitation' && pct > 30 && ' Precipitation is a major driver — typical for shallow or alluvial aquifers.'}
              {feat === 'potential_evaporation' && pct > 20 && ' Evapotranspiration significantly depletes recharge during warm months.'}
              {feat === 'temperature_2m' && pct > 30 && ' Strong temperature influence — may indicate thermal effects or correlation with evapotranspiration.'}
            </p>
          )
        })}
      </div>

      <div className="flex gap-4 text-[10px] text-text-muted">
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded" style={{ backgroundColor: '#06b6d4' }} /> Strong (&gt;40%)</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded" style={{ backgroundColor: '#eab308' }} /> Moderate (15-40%)</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded" style={{ backgroundColor: '#6b7280' }} /> Weak (&lt;15%)</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 2: Model quality
// ---------------------------------------------------------------------------

function QualitySection({ modelId }: { modelId: string }) {
  const mutation = useResidualAnalysis()
  if (!mutation.data && !mutation.isPending && !mutation.isError) mutation.mutate(modelId)
  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError) return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const d = mutation.data
  const balanced = d.bias_status === 'balanced' || d.bias_status === 'equilibre'
  const normal = d.normality_pvalue != null ? d.normality_pvalue >= 0.05 : null
  const acfOk = d.acf_lag1 != null ? Math.abs(d.acf_lag1) < 0.3 : null
  const direction = d.mean_error < 0 ? 'overestimates' : 'underestimates'

  return (
    <div className="space-y-3">
      <div className={`flex items-center gap-3 p-3 rounded-lg border ${balanced ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-amber-500/10 border-amber-500/30'}`}>
        {balanced ? <CheckCircle className="w-6 h-6 text-emerald-400 shrink-0" /> : <AlertTriangle className="w-6 h-6 text-amber-400 shrink-0" />}
        <div>
          <p className={`text-sm font-semibold ${balanced ? 'text-emerald-400' : 'text-amber-400'}`}>
            {balanced ? 'Unbiased predictions' : 'Systematic bias detected'}
          </p>
          <p className="text-xs text-text-muted">
            {balanced
              ? 'Prediction errors are centered around zero — no systematic over/under-estimation.'
              : `The model ${direction} the water level by ${Math.abs(d.mean_error).toFixed(3)} m on average.`}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">Mean error</span>
            <InfoTip text="Average prediction error (bias). A value close to zero means no systematic over- or under-estimation. Positive = model underestimates levels." iconSize={10} />
          </div>
          <p className="text-base font-bold font-mono text-text-primary">{d.mean_error.toFixed(3)} <span className="text-xs text-text-muted font-normal">m</span></p>
        </div>
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">Typical error</span>
            <InfoTip text="Standard deviation of prediction errors — the typical magnitude of mistakes. 95% of errors fall within ±2σ. Compare with the natural amplitude of the water level to assess significance." iconSize={10} />
          </div>
          <p className="text-base font-bold font-mono text-text-primary">±{d.std_error.toFixed(3)} <span className="text-xs text-text-muted font-normal">m</span></p>
        </div>
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">Normal errors</span>
            <InfoTip text="D'Agostino-Pearson normality test on residuals. 'Yes' means errors are random and well-behaved. 'No' means the model misses some patterns — look for clusters in the error chart below." iconSize={10} />
          </div>
          <p className="text-base font-bold">
            {normal === null ? <span className="text-text-muted">?</span> : normal ? <span className="text-emerald-400">Yes</span> : <span className="text-amber-400">No</span>}
          </p>
        </div>
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">Independent</span>
            <InfoTip text="Autocorrelation at lag 1 (ACF₁). Low (< 0.3) means successive errors are independent — good. High means errors are correlated: if the model is wrong today, it will likely be wrong tomorrow too, suggesting missing dynamics." iconSize={10} />
          </div>
          <p className="text-base font-bold">
            {acfOk === null ? <span className="text-text-muted">?</span> : acfOk ? <span className="text-emerald-400">Yes</span> : <span className="text-amber-400">No</span>}
          </p>
          {d.acf_lag1 != null && <p className="text-[10px] text-text-muted mt-0.5">ACF₁ = {d.acf_lag1.toFixed(3)}</p>}
        </div>
      </div>

      {d.residuals && d.dates && (
        <div>
          <p className="text-xs text-text-muted mb-1">Error over time — ideally scattered evenly around zero</p>
          <div className="h-[200px]">
            <Plot
              data={[{
                type: 'scatter', mode: 'markers',
                x: d.dates, y: d.residuals,
                marker: { color: '#f43f5e', size: 3, opacity: 0.6 },
                hovertemplate: '%{x|%d/%m/%Y}<br>Error: %{y:.4f} m<extra></extra>',
              }]}
              layout={{
                ...darkLayout,
                margin: { t: 5, r: 20, b: 30, l: 50 },
                height: 200,
                xaxis: { ...darkLayout.xaxis },
                yaxis: { ...darkLayout.yaxis, title: { text: 'Error (m)' } },
                shapes: [{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: 0, line: { color: 'rgba(255,255,255,0.15)', dash: 'dash', width: 1 } }],
              }}
              config={plotlyConfig}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 3: Aquifer behavior (temporal memory + seasonality)
// ---------------------------------------------------------------------------

function BehaviorSection({ modelId }: { modelId: string }) {
  const lagMutation = useLagImportance()
  const seasonMutation = useSeasonalityAnalysis()

  if (!lagMutation.data && !lagMutation.isPending && !lagMutation.isError) lagMutation.mutate(modelId)
  if (!seasonMutation.data && !seasonMutation.isPending && !seasonMutation.isError) seasonMutation.mutate(modelId)

  const lagLoading = lagMutation.isPending
  const seasonLoading = seasonMutation.isPending

  const lag = lagMutation.data
  const season = seasonMutation.data

  const memoryHorizon = lag?.significant_lags?.length
    ? Math.max(...lag.significant_lags)
    : null

  const periodLabels: Record<number, string> = { 7: 'Weekly', 30: 'Monthly', 90: 'Quarterly', 365: 'Annual' }

  return (
    <div className="space-y-4">
      {/* Response time KPI */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="bg-bg-hover rounded-lg p-3 border border-accent-cyan/20">
          <div className="flex items-center gap-1 mb-1">
            <Clock className="w-3.5 h-3.5 text-accent-cyan" />
            <span className="text-[10px] text-text-muted uppercase">Aquifer memory</span>
            <InfoTip text="How many days of past data significantly influence today's water level. Longer memory = slower, deeper aquifer. Short memory = fast-responding (alluvial, shallow). Derived from autocorrelation analysis (ACF)." iconSize={10} />
          </div>
          {lagLoading ? (
            <div className="h-6 bg-white/5 rounded animate-pulse mt-1" />
          ) : memoryHorizon != null ? (
            <p className="text-lg font-bold text-accent-cyan">{memoryHorizon} <span className="text-xs text-text-muted font-normal">days</span></p>
          ) : (
            <p className="text-sm text-text-muted">—</p>
          )}
          {memoryHorizon != null && (
            <p className="text-[10px] text-text-muted mt-0.5">
              {memoryHorizon > 180 ? 'Deep or confined aquifer with very long response time.' :
               memoryHorizon > 60 ? 'Moderate response — typical for sedimentary or semi-confined aquifers.' :
               'Fast response — typical for alluvial or shallow aquifers.'}
            </p>
          )}
        </div>

        <div className="bg-bg-hover rounded-lg p-3 border border-amber-500/20">
          <div className="flex items-center gap-1 mb-1">
            <Sun className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[10px] text-text-muted uppercase">Detected cycles</span>
            <InfoTip text="Significant periodic patterns in the water level signal, detected via FFT spectral analysis. Annual cycles are expected for most aquifers. Monthly cycles may indicate tidal or pumping influences." iconSize={10} />
          </div>
          {seasonLoading ? (
            <div className="h-6 bg-white/5 rounded animate-pulse mt-1" />
          ) : season?.detected_periods?.length ? (
            <div className="flex flex-wrap gap-1.5 mt-1">
              {season.detected_periods.map(p => (
                <span key={p} className="px-2 py-0.5 bg-amber-500/15 text-amber-400 text-xs rounded-md font-medium">
                  {periodLabels[p] ?? `${p}d`}
                  {season.period_strengths?.[String(p)] != null && (
                    <span className="text-amber-400/60 ml-1">({season.period_strengths[String(p)].toFixed(0)}x)</span>
                  )}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-sm text-text-muted mt-1">None detected</p>
          )}
        </div>
      </div>

      {/* ACF chart */}
      {lag && (
        <div className="h-[200px]">
          <Plot
            data={[{
              type: 'bar',
              x: lag.lags, y: lag.autocorrelations,
              marker: { color: lag.lags.map(l => lag.significant_lags?.includes(l) ? '#06b6d4' : 'rgba(6,182,212,0.15)') },
              hovertemplate: 'Day -%{x}<br>Autocorrelation: %{y:.3f}<extra></extra>',
            }]}
            layout={{
              ...darkLayout,
              height: 200,
              xaxis: { ...darkLayout.xaxis, title: { text: 'Lag (days)' } },
              yaxis: { ...darkLayout.yaxis, title: { text: 'ACF' } },
              margin: { t: 5, r: 20, b: 35, l: 50 },
            }}
            config={plotlyConfig}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}

      {/* Variance decomposition */}
      {season?.variance_trend != null && season.variance_seasonal != null && season.variance_residual != null && (
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-2">
            <span className="text-xs font-medium text-text-secondary">Signal decomposition</span>
            <InfoTip text="STL decomposition breaks the water level into 3 components: long-term trend (climate change, land use), seasonal cycle (annual recharge/discharge), and residual noise (unpredictable events). A well-modeled aquifer has low residual noise." iconSize={10} />
          </div>
          <div className="h-6 rounded-lg overflow-hidden flex">
            {[
              { pct: season.variance_trend!, color: '#06b6d4', label: 'Trend' },
              { pct: season.variance_seasonal!, color: '#8b5cf6', label: 'Seasonal' },
              { pct: season.variance_residual!, color: '#f43f5e', label: 'Noise' },
            ].map(({ pct, color, label }) => (
              <div key={label} className="flex items-center justify-center text-[9px] font-semibold"
                style={{ width: `${pct}%`, backgroundColor: color, color: pct > 10 ? '#0f172a' : 'transparent' }}>
                {pct > 10 ? `${label} ${pct.toFixed(0)}%` : ''}
              </div>
            ))}
          </div>
          <div className="flex gap-3 mt-1.5 text-[10px] text-text-muted">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded" style={{ backgroundColor: '#06b6d4' }} />Trend {season.variance_trend!.toFixed(0)}%</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded" style={{ backgroundColor: '#8b5cf6' }} />Seasonal {season.variance_seasonal!.toFixed(0)}%</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded" style={{ backgroundColor: '#f43f5e' }} />Noise {season.variance_residual!.toFixed(0)}%</span>
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 4: Expert tools (collapsed by default)
// ---------------------------------------------------------------------------

type ExpertMethod = 'permutation' | 'shap' | 'gradients'

function ExpertSection({ modelId }: { modelId: string }) {
  const [method, setMethod] = useState<ExpertMethod>('permutation')

  const METHODS: { key: ExpertMethod; label: string; tip: string }[] = [
    { key: 'permutation', label: 'Permutation', tip: 'Measures prediction degradation when each variable is randomly shuffled. Model-agnostic, reliable but slow.' },
    { key: 'shap', label: 'SHAP', tip: 'Shapley values from game theory — fairly distributes the prediction among all input variables. Accounts for interactions between features.' },
    { key: 'gradients', label: 'Integrated Gradients', tip: 'Traces the gradient path from a baseline (zero input) to the actual input. Shows which timesteps and features the neural network is most sensitive to.' },
  ]

  return (
    <div className="space-y-3">
      <div className="flex gap-1">
        {METHODS.map(m => (
          <button key={m.key} onClick={() => setMethod(m.key)} title={m.tip}
            className={`px-2.5 py-1 text-[11px] font-medium rounded transition-colors ${method === m.key ? 'bg-white/10 text-text-primary' : 'text-text-muted hover:text-text-primary hover:bg-white/5'}`}>
            {m.label}
          </button>
        ))}
      </div>
      {method === 'permutation' && <PermutationView modelId={modelId} />}
      {method === 'shap' && <ShapView modelId={modelId} />}
      {method === 'gradients' && <GradientsView modelId={modelId} />}
    </div>
  )
}

function PermutationView({ modelId }: { modelId: string }) {
  const m = usePermutationImportance()
  if (!m.data && !m.isPending && !m.isError) m.mutate({ model_id: modelId, n_permutations: 3 })
  if (m.isPending) return <LoadingSkeleton />
  if (m.isError) return <ErrorState message={(m.error as Error).message} onRetry={() => m.mutate({ model_id: modelId })} />
  const imp = m.data ? extractImportance(m.data) : null
  if (!imp) return <p className="text-xs text-text-muted py-4">No data.</p>
  return (
    <div className="space-y-2">
      <div className="h-[220px]">
        <Plot
          data={[{ type: 'bar', orientation: 'h' as const, y: imp.features.map(hydroLabel), x: imp.values, marker: { color: '#f59e0b' } }]}
          layout={{ ...darkLayout, margin: { t: 5, r: 20, b: 35, l: 160 }, xaxis: { ...darkLayout.xaxis, title: { text: 'Importance' } }, yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const } }}
          config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
        />
      </div>
      <p className="text-[10px] text-text-muted">Higher = the model relies more on this variable. If shuffling a variable barely changes predictions, the model doesn't need it.</p>
    </div>
  )
}

function ShapView({ modelId }: { modelId: string }) {
  const m = useShapAnalysis()
  if (!m.data && !m.isPending && !m.isError) m.mutate({ model_id: modelId })
  if (m.isPending) return <LoadingSkeleton />
  if (m.isError) return <ErrorState message={(m.error as Error).message} onRetry={() => m.mutate({ model_id: modelId })} />
  const imp = m.data ? extractImportance(m.data) : null
  if (!imp) return <p className="text-xs text-text-muted py-4">No data.</p>
  return (
    <div className="space-y-2">
      <div className="h-[220px]">
        <Plot
          data={[{ type: 'bar', orientation: 'h' as const, y: imp.features.map(hydroLabel), x: imp.values, marker: { color: imp.values.map(v => v >= 0 ? '#8b5cf6' : '#f43f5e') } }]}
          layout={{ ...darkLayout, margin: { t: 5, r: 20, b: 35, l: 160 }, xaxis: { ...darkLayout.xaxis, title: { text: 'Mean SHAP value' } }, yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const } }}
          config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
        />
      </div>
      <p className="text-[10px] text-text-muted">Positive SHAP = pushes prediction up, negative = pushes down. Sign and magnitude show the direction and strength of each variable's influence.</p>
    </div>
  )
}

function GradientsView({ modelId }: { modelId: string }) {
  const m = useGradientAnalysis()
  if (!m.data && !m.isPending && !m.isError) m.mutate({ model_id: modelId, method: 'integrated_gradients' })
  if (m.isPending) return <LoadingSkeleton />
  if (m.isError) return <ErrorState message={(m.error as Error).message} onRetry={() => m.mutate({ model_id: modelId, method: 'integrated_gradients' })} />
  if (!m.data) return null

  const hasTemporal = m.data.temporal_importance && m.data.temporal_importance.length > 0
  const imp = extractImportance(m.data)

  return (
    <div className="space-y-3">
      {hasTemporal && (
        <div className="h-[200px]">
          <Plot
            data={[{
              type: 'scatter', mode: 'lines',
              x: m.data.temporal_importance!.map((_, i) => i - m.data.temporal_importance!.length),
              y: m.data.temporal_importance!,
              line: { color: '#10b981', width: 1.5 },
              fill: 'tozeroy' as const,
              fillcolor: 'rgba(16,185,129,0.1)',
              hovertemplate: 'Day %{x}<br>Attribution: %{y:.4f}<extra></extra>',
            }]}
            layout={{
              ...darkLayout, height: 200,
              margin: { t: 5, r: 20, b: 35, l: 50 },
              xaxis: { ...darkLayout.xaxis, title: { text: 'Days before prediction' } },
              yaxis: { ...darkLayout.yaxis, title: { text: 'Sensitivity' } },
            }}
            config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}
      {imp && (
        <div className="h-[180px]">
          <Plot
            data={[{ type: 'bar', orientation: 'h' as const, y: imp.features.map(hydroLabel), x: imp.values, marker: { color: '#10b981' } }]}
            layout={{ ...darkLayout, margin: { t: 5, r: 20, b: 35, l: 160 }, xaxis: { ...darkLayout.xaxis, title: { text: 'Attribution' } }, yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const } }}
            config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}
      <p className="text-[10px] text-text-muted">Temporal chart: which past days the model is most sensitive to. Feature chart: which variables contribute most to the gradient signal.</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function ExplainabilityPanel({ modelId, className = '' }: Props) {
  const [openSections, setOpenSections] = useState<Set<Section>>(new Set(['drivers', 'quality', 'behavior']))

  const toggle = (s: Section) => setOpenSections(prev => {
    const next = new Set(prev)
    next.has(s) ? next.delete(s) : next.add(s)
    return next
  })

  const sections: { key: Section; icon: React.ElementType; title: string; tip: string }[] = [
    { key: 'drivers', icon: TrendingUp, title: 'What drives the water level', tip: 'Correlation-based feature importance — measures how strongly each meteorological variable (precipitation, temperature, PET) correlates with the water level. Higher = the model relies more on this variable to make predictions.' },
    { key: 'quality', icon: Activity, title: 'Prediction quality', tip: 'Residual analysis — checks whether prediction errors are random (good) or systematic (bad). Looks at bias, error magnitude, normality, and temporal independence of residuals.' },
    { key: 'behavior', icon: Clock, title: 'Aquifer response', tip: 'Combines temporal memory analysis (ACF — how far back in time the water level depends on itself) and seasonality detection (FFT — which periodic cycles exist in the signal).' },
    { key: 'expert', icon: Beaker, title: 'Expert methods', tip: 'Advanced ML interpretability methods: Permutation Importance (model-agnostic), SHAP values (game theory), Integrated Gradients (neural network sensitivity). These provide alternative perspectives on feature importance.' },
  ]

  return (
    <div className={`bg-bg-card rounded-xl border border-white/5 p-4 space-y-1 ${className}`}>
      <h3 className="text-sm font-semibold text-text-primary mb-2">Model Explainability</h3>

      {sections.map(({ key, icon, title, tip }) => (
        <div key={key} className="border-t border-white/5 pt-1">
          <SectionHeader icon={icon} title={title} tip={tip} open={openSections.has(key)} onToggle={() => toggle(key)} />
          {openSections.has(key) && (
            <div className="pb-3 pt-1">
              {key === 'drivers' && <DriversSection modelId={modelId} />}
              {key === 'quality' && <QualitySection modelId={modelId} />}
              {key === 'behavior' && <BehaviorSection modelId={modelId} />}
              {key === 'expert' && <ExpertSection modelId={modelId} />}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
