import { useState } from 'react'
import { useTranslation } from 'react-i18next'
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
  const { t } = useTranslation()
  return (
    <div className="text-center py-6">
      <p className="text-xs text-red-400 mb-2">{message}</p>
      <button onClick={onRetry} className="text-xs text-accent-cyan hover:underline">{t('sharedComponents.explainability.retry')}</button>
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

function useHydroLabel() {
  const { t } = useTranslation()
  return (name: string): string => {
    const map: Record<string, string> = {
      total_precipitation: t('sharedComponents.explainability.varPrecipitation'),
      potential_evaporation: t('sharedComponents.explainability.varEvapotranspiration'),
      temperature_2m: t('sharedComponents.explainability.varTemperature'),
      niveau_nappe_eau: t('sharedComponents.explainability.varWaterLevel'),
    }
    return map[name] ?? name.replace(/_/g, ' ')
  }
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
  const { t } = useTranslation()
  const hydroLabel = useHydroLabel()
  const mutation = useFeatureImportance()
  if (!mutation.data && !mutation.isPending && !mutation.isError) mutation.mutate(modelId)
  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError) return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const importance = extractImportance(mutation.data)
  if (!importance) return <p className="text-xs text-text-muted py-4">{t('sharedComponents.explainability.noImportanceData')}</p>

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
            xaxis: { ...darkLayout.xaxis, title: { text: t('sharedComponents.explainability.correlationStrength') } },
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
              {' '}{t('sharedComponents.explainability.representsSignal')} <span className="text-text-primary font-mono">{pct.toFixed(0)}%</span> {t('sharedComponents.explainability.ofSignal')}
              {isTarget && t('sharedComponents.explainability.autocorrelated')}
              {feat === 'total_precipitation' && pct > 30 && t('sharedComponents.explainability.precipMajor')}
              {feat === 'potential_evaporation' && pct > 20 && t('sharedComponents.explainability.etpReducesRecharge')}
              {feat === 'temperature_2m' && pct > 30 && t('sharedComponents.explainability.tempStrong')}
            </p>
          )
        })}
      </div>

      <div className="flex gap-4 text-[10px] text-text-muted">
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded" style={{ backgroundColor: '#06b6d4' }} /> {t('sharedComponents.explainability.strong')}</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded" style={{ backgroundColor: '#eab308' }} /> {t('sharedComponents.explainability.moderate')}</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded" style={{ backgroundColor: '#6b7280' }} /> {t('sharedComponents.explainability.weak')}</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 2: Model quality
// ---------------------------------------------------------------------------

function QualitySection({ modelId }: { modelId: string }) {
  const { t } = useTranslation()
  const mutation = useResidualAnalysis()
  if (!mutation.data && !mutation.isPending && !mutation.isError) mutation.mutate(modelId)
  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError) return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const d = mutation.data
  const balanced = d.bias_status === 'balanced' || d.bias_status === 'equilibre'
  const normal = d.normality_pvalue != null ? d.normality_pvalue >= 0.05 : null
  const acfOk = d.acf_lag1 != null ? Math.abs(d.acf_lag1) < 0.3 : null
  const direction = d.mean_error < 0 ? t('sharedComponents.explainability.overestimates') : t('sharedComponents.explainability.underestimates')

  return (
    <div className="space-y-3">
      <div className={`flex items-center gap-3 p-3 rounded-lg border ${balanced ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-amber-500/10 border-amber-500/30'}`}>
        {balanced ? <CheckCircle className="w-6 h-6 text-emerald-400 shrink-0" /> : <AlertTriangle className="w-6 h-6 text-amber-400 shrink-0" />}
        <div>
          <p className={`text-sm font-semibold ${balanced ? 'text-emerald-400' : 'text-amber-400'}`}>
            {balanced ? t('sharedComponents.explainability.predictionsUnbiased') : t('sharedComponents.explainability.systematicBias')}
          </p>
          <p className="text-xs text-text-muted">
            {balanced
              ? t('sharedComponents.explainability.balancedDesc')
              : t('sharedComponents.explainability.biasDesc', { direction, value: Math.abs(d.mean_error).toFixed(3) })}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">{t('sharedComponents.explainability.meanError')}</span>
            <InfoTip text={t('sharedComponents.explainability.meanErrorTip')} iconSize={10} />
          </div>
          <p className="text-base font-bold font-mono text-text-primary">{d.mean_error.toFixed(3)} <span className="text-xs text-text-muted font-normal">m</span></p>
        </div>
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">{t('sharedComponents.explainability.typicalError')}</span>
            <InfoTip text={t('sharedComponents.explainability.typicalErrorTip')} iconSize={10} />
          </div>
          <p className="text-base font-bold font-mono text-text-primary">±{d.std_error.toFixed(3)} <span className="text-xs text-text-muted font-normal">m</span></p>
        </div>
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">{t('sharedComponents.explainability.normalErrors')}</span>
            <InfoTip text={t('sharedComponents.explainability.normalErrorsTip')} iconSize={10} />
          </div>
          <p className="text-base font-bold">
            {normal === null ? <span className="text-text-muted">?</span> : normal ? <span className="text-emerald-400">{t('sharedComponents.explainability.yes')}</span> : <span className="text-amber-400">{t('sharedComponents.explainability.no')}</span>}
          </p>
        </div>
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">{t('sharedComponents.explainability.independent')}</span>
            <InfoTip text={t('sharedComponents.explainability.independentTip')} iconSize={10} />
          </div>
          <p className="text-base font-bold">
            {acfOk === null ? <span className="text-text-muted">?</span> : acfOk ? <span className="text-emerald-400">{t('sharedComponents.explainability.yes')}</span> : <span className="text-amber-400">{t('sharedComponents.explainability.no')}</span>}
          </p>
          {d.acf_lag1 != null && <p className="text-[10px] text-text-muted mt-0.5">ACF₁ = {d.acf_lag1.toFixed(3)}</p>}
        </div>
      </div>

      {d.residuals && d.dates && (
        <div>
          <p className="text-xs text-text-muted mb-1">{t('sharedComponents.explainability.errorOverTime')}</p>
          <div className="h-[200px]">
            <Plot
              data={[{
                type: 'scatter', mode: 'markers',
                x: d.dates, y: d.residuals,
                marker: { color: '#f43f5e', size: 3, opacity: 0.6 },
                hovertemplate: `%{x|%d/%m/%Y}<br>${t('sharedComponents.explainability.errorAtDate')} : %{y:.4f} m<extra></extra>`,
              }]}
              layout={{
                ...darkLayout,
                margin: { t: 5, r: 20, b: 30, l: 50 },
                height: 200,
                xaxis: { ...darkLayout.xaxis },
                yaxis: { ...darkLayout.yaxis, title: { text: t('sharedComponents.explainability.errorM') } },
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
  const { t } = useTranslation()
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

  const periodLabels: Record<number, string> = {
    7: t('sharedComponents.explainability.weekly'),
    30: t('sharedComponents.explainability.monthly'),
    90: t('sharedComponents.explainability.quarterly'),
    365: t('sharedComponents.explainability.annual'),
  }

  return (
    <div className="space-y-4">
      {/* Response time KPI */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="bg-bg-hover rounded-lg p-3 border border-accent-cyan/20">
          <div className="flex items-center gap-1 mb-1">
            <Clock className="w-3.5 h-3.5 text-accent-cyan" />
            <span className="text-[10px] text-text-muted uppercase">{t('sharedComponents.explainability.aquiferMemory')}</span>
            <InfoTip text={t('sharedComponents.explainability.aquiferMemoryTip')} iconSize={10} />
          </div>
          {lagLoading ? (
            <div className="h-6 bg-white/5 rounded animate-pulse mt-1" />
          ) : memoryHorizon != null ? (
            <p className="text-lg font-bold text-accent-cyan">{memoryHorizon} <span className="text-xs text-text-muted font-normal">{t('sharedComponents.explainability.days')}</span></p>
          ) : (
            <p className="text-sm text-text-muted">—</p>
          )}
          {memoryHorizon != null && (
            <p className="text-[10px] text-text-muted mt-0.5">
              {memoryHorizon > 180 ? t('sharedComponents.explainability.deepConfined') :
               memoryHorizon > 60 ? t('sharedComponents.explainability.moderateResponse') :
               t('sharedComponents.explainability.fastResponse')}
            </p>
          )}
        </div>

        <div className="bg-bg-hover rounded-lg p-3 border border-amber-500/20">
          <div className="flex items-center gap-1 mb-1">
            <Sun className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[10px] text-text-muted uppercase">{t('sharedComponents.explainability.cyclesDetected')}</span>
            <InfoTip text={t('sharedComponents.explainability.cyclesTip')} iconSize={10} />
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
            <p className="text-sm text-text-muted mt-1">{t('sharedComponents.explainability.noCycle')}</p>
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
              hovertemplate: `${t('sharedComponents.explainability.lagDay', { lag: '%{x}' })}<br>${t('sharedComponents.explainability.autocorrelation')} : %{y:.3f}<extra></extra>`,
            }]}
            layout={{
              ...darkLayout,
              height: 200,
              xaxis: { ...darkLayout.xaxis, title: { text: t('sharedComponents.explainability.lagDays') } },
              yaxis: { ...darkLayout.yaxis, title: { text: t('sharedComponents.explainability.acf') } },
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
            <span className="text-xs font-medium text-text-secondary">{t('sharedComponents.explainability.signalDecomposition')}</span>
            <InfoTip text={t('sharedComponents.explainability.signalDecompositionTip')} iconSize={10} />
          </div>
          <div className="h-6 rounded-lg overflow-hidden flex">
            {[
              { pct: season.variance_trend!, color: '#06b6d4', label: t('sharedComponents.explainability.trend') },
              { pct: season.variance_seasonal!, color: '#8b5cf6', label: t('sharedComponents.explainability.seasonal') },
              { pct: season.variance_residual!, color: '#f43f5e', label: t('sharedComponents.explainability.noise') },
            ].map(({ pct, color, label }) => (
              <div key={label} className="flex items-center justify-center text-[9px] font-semibold"
                style={{ width: `${pct}%`, backgroundColor: color, color: pct > 10 ? '#0f172a' : 'transparent' }}>
                {pct > 10 ? `${label} ${pct.toFixed(0)}%` : ''}
              </div>
            ))}
          </div>
          <div className="flex gap-3 mt-1.5 text-[10px] text-text-muted">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded" style={{ backgroundColor: '#06b6d4' }} />{t('sharedComponents.explainability.trend')} {season.variance_trend!.toFixed(0)}%</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded" style={{ backgroundColor: '#8b5cf6' }} />{t('sharedComponents.explainability.seasonal')} {season.variance_seasonal!.toFixed(0)}%</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded" style={{ backgroundColor: '#f43f5e' }} />{t('sharedComponents.explainability.noise')} {season.variance_residual!.toFixed(0)}%</span>
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
  const { t } = useTranslation()
  const [method, setMethod] = useState<ExpertMethod>('permutation')

  const METHODS: { key: ExpertMethod; label: string; tip: string }[] = [
    { key: 'permutation', label: t('sharedComponents.explainability.permutationLabel'), tip: t('sharedComponents.explainability.permutationTip') },
    { key: 'shap', label: 'SHAP', tip: t('sharedComponents.explainability.shapTip') },
    { key: 'gradients', label: t('sharedComponents.explainability.gradientsLabel'), tip: t('sharedComponents.explainability.gradientsTip') },
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
  const { t } = useTranslation()
  const hydroLabel = useHydroLabel()
  const m = usePermutationImportance()
  if (!m.data && !m.isPending && !m.isError) m.mutate({ model_id: modelId, n_permutations: 3 })
  if (m.isPending) return <LoadingSkeleton />
  if (m.isError) return <ErrorState message={(m.error as Error).message} onRetry={() => m.mutate({ model_id: modelId })} />
  const imp = m.data ? extractImportance(m.data) : null
  if (!imp) return <p className="text-xs text-text-muted py-4">{t('sharedComponents.explainability.noData')}</p>
  return (
    <div className="space-y-2">
      <div className="h-[220px]">
        <Plot
          data={[{ type: 'bar', orientation: 'h' as const, y: imp.features.map(hydroLabel), x: imp.values, marker: { color: '#f59e0b' } }]}
          layout={{ ...darkLayout, margin: { t: 5, r: 20, b: 35, l: 160 }, xaxis: { ...darkLayout.xaxis, title: { text: t('sharedComponents.explainability.importance') } }, yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const } }}
          config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
        />
      </div>
      <p className="text-[10px] text-text-muted">{t('sharedComponents.explainability.permutationDesc')}</p>
    </div>
  )
}

function ShapView({ modelId }: { modelId: string }) {
  const { t } = useTranslation()
  const hydroLabel = useHydroLabel()
  const m = useShapAnalysis()
  if (!m.data && !m.isPending && !m.isError) m.mutate({ model_id: modelId })
  if (m.isPending) return <LoadingSkeleton />
  if (m.isError) return <ErrorState message={(m.error as Error).message} onRetry={() => m.mutate({ model_id: modelId })} />
  const imp = m.data ? extractImportance(m.data) : null
  if (!imp) return <p className="text-xs text-text-muted py-4">{t('sharedComponents.explainability.noData')}</p>
  return (
    <div className="space-y-2">
      <div className="h-[220px]">
        <Plot
          data={[{ type: 'bar', orientation: 'h' as const, y: imp.features.map(hydroLabel), x: imp.values, marker: { color: imp.values.map(v => v >= 0 ? '#8b5cf6' : '#f43f5e') } }]}
          layout={{ ...darkLayout, margin: { t: 5, r: 20, b: 35, l: 160 }, xaxis: { ...darkLayout.xaxis, title: { text: t('sharedComponents.explainability.shapValue') } }, yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const } }}
          config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
        />
      </div>
      <p className="text-[10px] text-text-muted">{t('sharedComponents.explainability.shapDesc')}</p>
    </div>
  )
}

function GradientsView({ modelId }: { modelId: string }) {
  const { t } = useTranslation()
  const hydroLabel = useHydroLabel()
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
              hovertemplate: `Jour %{x}<br>${t('sharedComponents.explainability.attribution')} : %{y:.4f}<extra></extra>`,
            }]}
            layout={{
              ...darkLayout, height: 200,
              margin: { t: 5, r: 20, b: 35, l: 50 },
              xaxis: { ...darkLayout.xaxis, title: { text: t('sharedComponents.explainability.dayBeforePrediction') } },
              yaxis: { ...darkLayout.yaxis, title: { text: t('sharedComponents.explainability.sensitivity') } },
            }}
            config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}
      {imp && (
        <div className="h-[180px]">
          <Plot
            data={[{ type: 'bar', orientation: 'h' as const, y: imp.features.map(hydroLabel), x: imp.values, marker: { color: '#10b981' } }]}
            layout={{ ...darkLayout, margin: { t: 5, r: 20, b: 35, l: 160 }, xaxis: { ...darkLayout.xaxis, title: { text: t('sharedComponents.explainability.attribution') } }, yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const } }}
            config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}
      <p className="text-[10px] text-text-muted">{t('sharedComponents.explainability.gradientsDesc')}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function ExplainabilityPanel({ modelId, className = '' }: Props) {
  const { t } = useTranslation()
  const [openSections, setOpenSections] = useState<Set<Section>>(new Set())

  const toggle = (s: Section) => setOpenSections(prev => {
    const next = new Set(prev)
    next.has(s) ? next.delete(s) : next.add(s)
    return next
  })

  const sections: { key: Section; icon: React.ElementType; title: string; tip: string }[] = [
    { key: 'drivers', icon: TrendingUp, title: t('sharedComponents.explainability.drivers'), tip: t('sharedComponents.explainability.driversTip') },
    { key: 'quality', icon: Activity, title: t('sharedComponents.explainability.quality'), tip: t('sharedComponents.explainability.qualityTip') },
    { key: 'behavior', icon: Clock, title: t('sharedComponents.explainability.behavior'), tip: t('sharedComponents.explainability.behaviorTip') },
    { key: 'expert', icon: Beaker, title: t('sharedComponents.explainability.expert'), tip: t('sharedComponents.explainability.expertTip') },
  ]

  return (
    <div className={`bg-bg-card rounded-xl border border-white/5 p-4 space-y-1 ${className}`}>
      <h3 className="text-sm font-semibold text-text-primary mb-2">{t('sharedComponents.explainability.title')}</h3>

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
