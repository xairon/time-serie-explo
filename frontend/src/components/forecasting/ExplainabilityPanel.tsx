import { useState } from 'react'
import {
  ChevronDown,
  ChevronRight,
  TrendingUp,
  Activity,
  Clock,
  Sun,
  Wrench,
  CheckCircle,
  AlertTriangle,
} from 'lucide-react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import {
  useFeatureImportance,
  usePermutationImportance,
  useShapAnalysis,
  useAttentionAnalysis,
  useGradientAnalysis,
  useLagImportance,
  useResidualAnalysis,
  useSeasonalityAnalysis,
} from '@/hooks/useForecasting'
import type { Layout } from 'plotly.js-dist-min'
import type { ExplainResult } from '@/lib/types'

interface ExplainabilityPanelProps {
  modelId: string
  className?: string
}

type TabKey = 'influence' | 'fiabilite' | 'memoire' | 'cycles' | 'avancee'
type AdvancedSubTab = 'permutation' | 'shap' | 'attention' | 'gradients'

const TABS: { key: TabKey; label: string; icon: React.ElementType; description: string }[] = [
  {
    key: 'influence',
    label: 'Feature influence',
    icon: TrendingUp,
    description: 'Which features most influence predictions',
  },
  {
    key: 'fiabilite',
    label: 'Model reliability',
    icon: Activity,
    description: 'Is the model reliable and unbiased',
  },
  {
    key: 'memoire',
    label: 'Temporal memory',
    icon: Clock,
    description: 'How far back the model looks in the past',
  },
  {
    key: 'cycles',
    label: 'Seasonal cycles',
    icon: Sun,
    description: 'Which seasonal cycles are detected',
  },
  {
    key: 'avancee',
    label: 'Advanced analysis',
    icon: Wrench,
    description: 'Advanced analysis tools for expert users',
  },
]

// ---------------------------------------------------------------------------
// Shared sub-components
// ---------------------------------------------------------------------------

function LoadingSkeleton() {
  return <div className="h-[200px] bg-bg-hover rounded-lg animate-pulse" />
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="text-center py-8">
      <p className="text-xs text-accent-red mb-2">Error: {message}</p>
      <button onClick={onRetry} className="text-xs text-accent-cyan hover:underline">
        Retry
      </button>
    </div>
  )
}

function NoDataState({ label }: { label: string }) {
  return (
    <div className="text-center py-8">
      <p className="text-xs text-text-secondary">
        No {label} data available for this model.
      </p>
    </div>
  )
}

function TabDescription({ text }: { text: string }) {
  return (
    <p className="text-xs text-text-secondary mb-4 italic">
      {text}
    </p>
  )
}

// ---------------------------------------------------------------------------
// Color helpers for influence tab
// ---------------------------------------------------------------------------

function getInfluenceColor(value: number): string {
  if (value > 0.4) return '#06b6d4' // cyan - high
  if (value > 0.15) return '#eab308' // yellow - medium
  return '#6b7280' // gray - low
}

function getInfluenceLabel(value: number): string {
  if (value > 0.4) return 'strong'
  if (value > 0.15) return 'moderate'
  return 'weak'
}

// ---------------------------------------------------------------------------
// Extract features/values from feature_importance record
// ---------------------------------------------------------------------------

function extractImportance(data: ExplainResult): { features: string[]; values: number[] } | null {
  if (!data.feature_importance) return null
  const entries = Object.entries(data.feature_importance)
    .filter(([, v]) => v != null)
    .map(([k, v]) => [k, v as number] as const)
    .sort(([, a], [, b]) => b - a)
  if (entries.length === 0) return null
  return {
    features: entries.map(([k]) => k),
    values: entries.map(([, v]) => v),
  }
}

// ---------------------------------------------------------------------------
// Horizontal bar chart helper
// ---------------------------------------------------------------------------

function HorizontalBarChart({
  features,
  values,
  color = '#06b6d4',
  colors,
  title,
}: {
  features: string[]
  values: number[]
  color?: string
  colors?: string[]
  title?: string
}) {
  const layout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Importance' } },
    yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const },
    margin: { t: title ? 30 : 10, r: 20, b: 40, l: 140 },
    ...(title ? { title: { text: title, font: { size: 12, color: '#94a3b8' } } } : {}),
  }

  return (
    <div className="h-[300px]">
      <Plot
        data={[
          {
            type: 'bar',
            orientation: 'h',
            y: features,
            x: values,
            marker: { color: colors ?? color },
          },
        ]}
        layout={layout}
        config={plotlyConfig}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab 1: Influence des variables
// ---------------------------------------------------------------------------

function InfluenceTab({ modelId }: { modelId: string }) {
  const mutation = useFeatureImportance()

  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate(modelId)
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const importance = extractImportance(mutation.data)
  if (!importance) return <NoDataState label="d'influence" />

  // Compute total for percentage
  const total = importance.values.reduce((sum, v) => sum + Math.abs(v), 0)
  const barColors = importance.values.map((v) => getInfluenceColor(total > 0 ? Math.abs(v) / total : 0))

  return (
    <div className="space-y-4">
      <TabDescription text="This chart shows which meteorological variables most influence the model's predictions. The longer the bar, the greater the variable's impact." />

      <HorizontalBarChart
        features={importance.features}
        values={importance.values}
        colors={barColors}
        title="Influence des variables sur la prediction"
      />

      {/* Plain-language interpretation */}
      <div className="bg-bg-hover rounded-lg p-4 space-y-2">
        <h4 className="text-xs font-semibold text-text-primary mb-2">Interpretation</h4>
        {importance.features.slice(0, 5).map((feat, i) => {
          const pct = total > 0 ? Math.round((Math.abs(importance.values[i]) / total) * 100) : 0
          const level = getInfluenceLabel(total > 0 ? Math.abs(importance.values[i]) / total : 0)
          const color = getInfluenceColor(total > 0 ? Math.abs(importance.values[i]) / total : 0)
          return (
            <p key={feat} className="text-xs" style={{ color }}>
              <span className="font-semibold">{feat}</span>
              {' '}has a {level} influence ({pct}%) on groundwater level predictions.
              {level === 'weak' && ' This may indicate a temporal lag or a low direct correlation.'}
            </p>
          )
        })}
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-[10px] text-text-secondary">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded" style={{ backgroundColor: '#06b6d4' }} /> Strong (&gt;40%)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded" style={{ backgroundColor: '#eab308' }} /> Moderate (15-40%)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded" style={{ backgroundColor: '#6b7280' }} /> Weak (&lt;15%)
        </span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab 2: Fiabilite du modele
// ---------------------------------------------------------------------------

function FiabiliteTab({ modelId }: { modelId: string }) {
  const mutation = useResidualAnalysis()

  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate(modelId)
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const data = mutation.data
  const isReliable = data.bias_status === 'equilibre'
  const isNormal = data.normality_pvalue != null ? data.normality_pvalue >= 0.05 : null
  const acfOk = data.acf_lag1 != null ? Math.abs(data.acf_lag1) < 0.3 : null
  const biasDirection = data.mean_error < 0 ? 'surestime' : 'sous-estime'

  const residualLayout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Date' } },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Residu (m)' } },
    margin: { t: 20, r: 20, b: 40, l: 60 },
    shapes: [{
      type: 'line',
      x0: 0,
      x1: 1,
      xref: 'paper',
      y0: 0,
      y1: 0,
      line: { color: 'rgba(255,255,255,0.2)', dash: 'dash', width: 1 },
    }],
  }

  return (
    <div className="space-y-4">
      <TabDescription text="This section evaluates the quality and reliability of the model's predictions by analyzing its errors." />

      {/* Big status indicator */}
      <div className={`flex items-center gap-3 p-4 rounded-lg border ${
        isReliable
          ? 'bg-emerald-500/10 border-emerald-500/30'
          : 'bg-amber-500/10 border-amber-500/30'
      }`}>
        {isReliable ? (
          <CheckCircle className="w-8 h-8 text-emerald-400 flex-shrink-0" />
        ) : (
          <AlertTriangle className="w-8 h-8 text-amber-400 flex-shrink-0" />
        )}
        <div>
          <p className={`text-sm font-bold ${isReliable ? 'text-emerald-400' : 'text-amber-400'}`}>
            {isReliable ? 'Reliable model' : 'Warning: bias detected'}
          </p>
          <p className="text-xs text-text-secondary mt-0.5">
            {isReliable
              ? 'Prediction errors are balanced and show no systematic trend.'
              : `The model systematically ${biasDirection === 'surestime' ? 'overestimates' : 'underestimates'} the groundwater level slightly.`
            }
          </p>
        </div>
      </div>

      {/* Metric cards with explanations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="bg-bg-hover rounded-lg p-4">
          <p className="text-[10px] text-text-secondary uppercase mb-1">Mean error</p>
          <p className="text-lg font-bold text-text-primary">
            {data.mean_error.toFixed(3)} <span className="text-xs text-text-secondary">m</span>
          </p>
          <p className="text-[11px] text-text-secondary mt-1">
            {Math.abs(data.mean_error) < 0.01
              ? 'The model has virtually no systematic bias.'
              : `The model slightly ${biasDirection === 'surestime' ? 'overestimates' : 'underestimates'} the level.`
            }
          </p>
        </div>

        <div className="bg-bg-hover rounded-lg p-4">
          <p className="text-[10px] text-text-secondary uppercase mb-1">Error std deviation</p>
          <p className="text-lg font-bold text-text-primary">
            {data.std_error.toFixed(3)} <span className="text-xs text-text-secondary">m</span>
          </p>
          <p className="text-[11px] text-text-secondary mt-1">
            Typical prediction accuracy: most errors are below {(data.std_error * 2).toFixed(2)} m.
          </p>
        </div>

        <div className="bg-bg-hover rounded-lg p-4">
          <p className="text-[10px] text-text-secondary uppercase mb-1">Error normality</p>
          <p className="text-lg font-bold text-text-primary">
            {isNormal === null ? '?' : isNormal ? (
              <span className="text-emerald-400">Oui</span>
            ) : (
              <span className="text-amber-400">Non</span>
            )}
          </p>
          <p className="text-[11px] text-text-secondary mt-1">
            {isNormal === null
              ? 'Data not available.'
              : isNormal
                ? 'Errors follow a normal distribution: the model captures the main patterns well.'
                : 'Errors are not normal: the model misses some patterns or there are extreme values.'
            }
          </p>
        </div>

        <div className="bg-bg-hover rounded-lg p-4">
          <p className="text-[10px] text-text-secondary uppercase mb-1">Error autocorrelation</p>
          <p className="text-lg font-bold text-text-primary">
            {data.acf_lag1 != null ? data.acf_lag1.toFixed(3) : '?'}
          </p>
          <p className="text-[11px] text-text-secondary mt-1">
            {acfOk === null
              ? 'Data not available.'
              : acfOk
                ? 'Low autocorrelation: errors are independent (good sign).'
                : 'High autocorrelation: successive errors are correlated, the model could be improved.'
            }
          </p>
        </div>
      </div>

      {/* Residual scatter plot */}
      {data.residuals && data.dates && (
        <div>
          <h4 className="text-xs font-semibold text-text-secondary mb-2">Error distribution over time</h4>
          <div className="h-[300px]">
            <Plot
              data={[{
                type: 'scatter',
                mode: 'markers',
                x: data.dates,
                y: data.residuals,
                marker: { color: '#f43f5e', size: 3, opacity: 0.6 },
                hovertemplate: '%{x|%d/%m/%Y}<br>Error: %{y:.4f} m<extra></extra>',
              }]}
              layout={residualLayout}
              config={plotlyConfig}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </div>
          <p className="text-[10px] text-text-secondary mt-1">
            Each point represents the model error at a given date. Ideally, points should be evenly distributed around zero.
          </p>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab 3: Memoire temporelle
// ---------------------------------------------------------------------------

function MemoireTab({ modelId }: { modelId: string }) {
  const mutation = useLagImportance()

  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate(modelId)
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const { lags, autocorrelations, partial_autocorrelations, significant_lags } = mutation.data

  // Find the first lag where autocorrelation drops below significance threshold
  const memoryHorizon = significant_lags && significant_lags.length > 0
    ? Math.max(...significant_lags)
    : lags.length > 0 ? lags[Math.max(0, autocorrelations.findIndex((v) => Math.abs(v) < 0.1) - 1)] : null

  // Detect periodicities from significant lags
  const periodicities: { lag: number; label: string }[] = []
  if (significant_lags) {
    for (const lag of significant_lags) {
      if (lag >= 6 && lag <= 8) periodicities.push({ lag, label: 'weekly' })
      else if (lag >= 28 && lag <= 32) periodicities.push({ lag, label: 'monthly' })
      else if (lag >= 85 && lag <= 95) periodicities.push({ lag, label: 'quarterly' })
      else if (lag >= 360 && lag <= 370) periodicities.push({ lag, label: 'annual' })
    }
  }

  const acfLayout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Time lag (days)' } },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Autocorrelation' } },
    margin: { t: 30, r: 20, b: 50, l: 60 },
    title: { text: 'Strength of link with the past', font: { size: 12, color: '#94a3b8' } },
  }

  const pacfLayout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Time lag (days)' } },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Partial correlation' } },
    margin: { t: 30, r: 20, b: 50, l: 60 },
    title: { text: 'Direct contribution of each lag', font: { size: 12, color: '#94a3b8' } },
  }

  return (
    <div className="space-y-4">
      <TabDescription text="This analysis shows how many days back the model looks to make predictions, and which temporal lags are most important." />

      {/* Key insight card */}
      {memoryHorizon != null && (
        <div className="bg-bg-hover rounded-lg p-4 border border-accent-cyan/20">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-5 h-5 text-accent-cyan" />
            <p className="text-sm font-bold text-text-primary">
              The model primarily uses the last <span className="text-accent-cyan">{memoryHorizon} days</span> to predict
            </p>
          </div>
          {periodicities.length > 0 && (
            <div className="mt-2">
              <p className="text-xs text-text-secondary mb-1">Detected periodicities:</p>
              <div className="flex flex-wrap gap-2">
                {periodicities.map(({ lag, label }) => (
                  <span key={lag} className="px-2 py-1 bg-accent-cyan/20 text-accent-cyan text-xs rounded-md">
                    {lag}j ({label})
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ACF chart */}
      <div className="h-[250px]">
        <Plot
          data={[{
            type: 'bar',
            x: lags,
            y: autocorrelations,
            marker: {
              color: lags.map((l) =>
                significant_lags?.includes(l) ? '#06b6d4' : 'rgba(6,182,212,0.2)'
              ),
            },
            hovertemplate: 'Day -%{x}<br>Autocorrelation: %{y:.3f}<extra></extra>',
          }]}
          layout={acfLayout}
          config={plotlyConfig}
          useResizeHandler
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      {/* PACF chart */}
      {partial_autocorrelations && (
        <div className="h-[250px]">
          <Plot
            data={[{
              type: 'bar',
              x: lags,
              y: partial_autocorrelations,
              marker: { color: '#8b5cf6' },
              hovertemplate: 'Day -%{x}<br>PACF: %{y:.3f}<extra></extra>',
            }]}
            layout={pacfLayout}
            config={plotlyConfig}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}

      {/* Significant lags summary */}
      {significant_lags && significant_lags.length > 0 && (
        <div className="bg-bg-hover rounded-lg p-3">
          <p className="text-xs text-text-secondary">
            <span className="font-semibold text-text-primary">Significant lags: </span>
            {significant_lags.slice(0, 10).map((l) => `${l}d`).join(', ')}
            {significant_lags.length > 10 && ` (+${significant_lags.length - 10} more)`}
          </p>
          <p className="text-[10px] text-text-secondary mt-1">
            Bright-colored bars show the days where the past has a significant influence on the current prediction.
          </p>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab 4: Cycles saisonniers
// ---------------------------------------------------------------------------

function CyclesTab({ modelId }: { modelId: string }) {
  const mutation = useSeasonalityAnalysis()

  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate(modelId)
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const data = mutation.data

  // Map periods to friendly labels and icons
  const periodLabels: Record<number, string> = {
    7: 'Weekly',
    14: 'Bi-weekly',
    30: 'Monthly',
    90: 'Quarterly',
    180: 'Semi-annual',
    365: 'Annual',
  }

  const getPeriodLabel = (p: number) => {
    if (periodLabels[p]) return periodLabels[p]
    if (p <= 7) return `${p} days`
    if (p <= 31) return `~${Math.round(p / 7)} weeks`
    if (p <= 365) return `~${Math.round(p / 30)} months`
    return `${p} days`
  }

  const hasVariance = data.variance_trend != null && data.variance_seasonal != null && data.variance_residual != null

  return (
    <div className="space-y-4">
      <TabDescription text="Analysis of repetitive cycles in groundwater level data, such as annual or monthly seasonal variations." />

      {/* Detected periods - prominent display */}
      <div className="bg-bg-hover rounded-lg p-4">
        <div className="flex items-center gap-2 mb-3">
          <Sun className="w-5 h-5 text-amber-400" />
          <h4 className="text-sm font-semibold text-text-primary">Detected cycles</h4>
        </div>
        {data.detected_periods.length > 0 ? (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {data.detected_periods.map((p) => {
              const strength = data.period_strengths
                ? data.period_strengths[String(p)]
                : null
              return (
                <div key={p} className="flex items-center gap-3 bg-bg-card rounded-lg p-3 border border-amber-500/20">
                  <div className="text-center">
                    <p className="text-lg font-bold text-amber-400">{p}j</p>
                    <p className="text-[10px] text-text-secondary">{getPeriodLabel(p)}</p>
                  </div>
                  {strength != null && (
                    <div className="ml-auto text-right">
                      <p className="text-xs text-text-secondary">Strength</p>
                      <p className="text-sm font-semibold text-text-primary">{strength.toFixed(1)}x</p>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <p className="text-xs text-text-secondary">No significant periodicity detected in the data.</p>
        )}
      </div>

      {/* Variance decomposition as stacked bar */}
      {hasVariance && (
        <div className="bg-bg-hover rounded-lg p-4">
          <h4 className="text-xs font-semibold text-text-primary mb-3">Variance decomposition</h4>
          <p className="text-xs text-text-secondary mb-3">
            What proportion of the level variation is explained by each component:
          </p>
          {/* Stacked horizontal bar */}
          <div className="h-8 rounded-lg overflow-hidden flex">
            <div
              className="flex items-center justify-center text-[10px] font-semibold"
              style={{
                width: `${data.variance_trend!}%`,
                backgroundColor: '#06b6d4',
                color: data.variance_trend! > 8 ? '#0f172a' : 'transparent',
              }}
            >
              {data.variance_trend! > 8 ? `${data.variance_trend!.toFixed(0)}%` : ''}
            </div>
            <div
              className="flex items-center justify-center text-[10px] font-semibold"
              style={{
                width: `${data.variance_seasonal!}%`,
                backgroundColor: '#8b5cf6',
                color: data.variance_seasonal! > 8 ? '#0f172a' : 'transparent',
              }}
            >
              {data.variance_seasonal! > 8 ? `${data.variance_seasonal!.toFixed(0)}%` : ''}
            </div>
            <div
              className="flex items-center justify-center text-[10px] font-semibold"
              style={{
                width: `${data.variance_residual!}%`,
                backgroundColor: '#f43f5e',
                color: data.variance_residual! > 8 ? '#0f172a' : 'transparent',
              }}
            >
              {data.variance_residual! > 8 ? `${data.variance_residual!.toFixed(0)}%` : ''}
            </div>
          </div>
          <div className="flex gap-4 mt-2 text-[10px] text-text-secondary">
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded" style={{ backgroundColor: '#06b6d4' }} />
              Long-term trend ({data.variance_trend!.toFixed(1)}%)
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded" style={{ backgroundColor: '#8b5cf6' }} />
              Seasonality ({data.variance_seasonal!.toFixed(1)}%)
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded" style={{ backgroundColor: '#f43f5e' }} />
              Residual noise ({data.variance_residual!.toFixed(1)}%)
            </span>
          </div>
          <p className="text-[10px] text-text-secondary mt-2">
            {data.variance_seasonal! > 50
              ? 'Seasonality dominates the variation: levels follow a predictable cycle.'
              : data.variance_trend! > 50
                ? 'The long-term trend dominates: levels are evolving mainly in one direction.'
                : 'Variation is distributed between trend, seasonality and noise.'
            }
          </p>
        </div>
      )}

      {/* STL decomposition plots */}
      {data.decomposition && data.decomposition_dates && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-text-secondary">Temporal decomposition (STL)</h4>
          {([
            { key: 'trend' as const, label: 'Long-term trend', color: '#06b6d4' },
            { key: 'seasonal' as const, label: 'Seasonal cycle', color: '#8b5cf6' },
            { key: 'residual' as const, label: 'Residual noise', color: '#f43f5e' },
          ]).map(({ key, label, color }) => {
            const values = data.decomposition![key]
            return (
              <div key={key} className="h-[150px]">
                <Plot
                  data={[{
                    type: 'scatter',
                    mode: 'lines',
                    x: data.decomposition_dates!,
                    y: values,
                    line: { color, width: 1.5 },
                  }]}
                  layout={{
                    ...darkLayout,
                    height: 140,
                    margin: { t: 25, r: 20, b: 20, l: 50 },
                    title: { text: label, font: { size: 11, color: '#94a3b8' } },
                    xaxis: { ...darkLayout.xaxis },
                    yaxis: { ...darkLayout.yaxis },
                  }}
                  config={plotlyConfig}
                  useResizeHandler
                  style={{ width: '100%', height: '100%' }}
                />
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab 5: Analyse avancee (sub-tabs for power users)
// ---------------------------------------------------------------------------

function PermutationSubTab({ modelId }: { modelId: string }) {
  const mutation = usePermutationImportance()

  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate({ model_id: modelId, n_permutations: 3 })
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate({ model_id: modelId })} />
  if (!mutation.data) return null

  const importance = extractImportance(mutation.data)
  if (!importance) return <NoDataState label="de permutation" />

  return (
    <div className="space-y-3">
      <HorizontalBarChart features={importance.features} values={importance.values} color="#f59e0b" title="Permutation Importance" />
      <p className="text-[10px] text-text-secondary">
        Measures the degradation of the prediction when each variable is randomly shuffled. The stronger the degradation, the more important the variable.
      </p>
    </div>
  )
}

function ShapSubTab({ modelId }: { modelId: string }) {
  const mutation = useShapAnalysis()

  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate({ model_id: modelId })
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate({ model_id: modelId })} />
  if (!mutation.data) return null

  const importance = extractImportance(mutation.data)
  if (!importance) return <NoDataState label="SHAP" />

  return (
    <div className="space-y-3">
      <HorizontalBarChart features={importance.features} values={importance.values} color="#8b5cf6" title="SHAP Feature Importance" />
      <p className="text-[10px] text-text-secondary">
        SHAP values: average contribution of each variable to each individual prediction. Method derived from game theory.
      </p>
    </div>
  )
}

function AttentionSubTab({ modelId }: { modelId: string }) {
  const mutation = useAttentionAnalysis()

  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate({ model_id: modelId })
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate({ model_id: modelId })} />
  if (!mutation.data) return null

  const data = mutation.data
  const hasEncoder = data.encoder_importance && Object.keys(data.encoder_importance).length > 0
  const hasDecoder = data.decoder_importance && Object.keys(data.decoder_importance).length > 0
  const hasHeatmap = data.attention_weights && data.attention_weights.length > 0

  if (!hasEncoder && !hasDecoder && !hasHeatmap) return <NoDataState label="d'attention" />

  const encoderExtracted = hasEncoder
    ? (() => {
        const entries = Object.entries(data.encoder_importance!).sort(([, a], [, b]) => b - a)
        return { features: entries.map(([k]) => k), values: entries.map(([, v]) => v) }
      })()
    : null

  const decoderExtracted = hasDecoder
    ? (() => {
        const entries = Object.entries(data.decoder_importance!).sort(([, a], [, b]) => b - a)
        return { features: entries.map(([k]) => k), values: entries.map(([, v]) => v) }
      })()
    : null

  const heatmapLayout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Query position' } },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Key position' }, autorange: 'reversed' as const },
    margin: { t: 30, r: 20, b: 50, l: 60 },
    title: { text: 'Attention Weights', font: { size: 12, color: '#94a3b8' } },
  }

  return (
    <div className="space-y-4">
      <p className="text-[10px] text-text-secondary">
        Transformer model attention weights: shows which variables and time steps the model "looks at" most.
      </p>
      {encoderExtracted && (
        <HorizontalBarChart features={encoderExtracted.features} values={encoderExtracted.values} color="#f59e0b" title="Encoder Importance" />
      )}
      {decoderExtracted && (
        <HorizontalBarChart features={decoderExtracted.features} values={decoderExtracted.values} color="#f97316" title="Decoder Importance" />
      )}
      {hasHeatmap && (
        <div className="h-[350px]">
          <Plot
            data={[{
              type: 'heatmap',
              z: data.attention_weights!,
              colorscale: 'Viridis',
              colorbar: { title: { text: 'Weight', side: 'right' }, tickfont: { color: '#94a3b8' } },
            }]}
            layout={heatmapLayout}
            config={plotlyConfig}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}
    </div>
  )
}

function GradientsSubTab({ modelId }: { modelId: string }) {
  const mutation = useGradientAnalysis()

  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate({ model_id: modelId, method: 'integrated_gradients' })
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return (
      <ErrorState
        message={(mutation.error as Error).message}
        onRetry={() => mutation.mutate({ model_id: modelId, method: 'integrated_gradients' })}
      />
    )
  if (!mutation.data) return null

  const data = mutation.data
  const hasTemporal = data.temporal_importance && data.temporal_importance.length > 0
  const importance = extractImportance(data)

  if (!hasTemporal && !importance) return <NoDataState label="de gradient" />

  const temporalLayout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Time step' } },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Attribution' } },
    margin: { t: 30, r: 20, b: 50, l: 60 },
    title: { text: 'Temporal attribution', font: { size: 12, color: '#94a3b8' } },
  }

  return (
    <div className="space-y-4">
      <p className="text-[10px] text-text-secondary">
        Integrated gradients: measures the sensitivity of the prediction to each input variable via gradient computation.
      </p>
      {hasTemporal && (
        <div className="h-[300px]">
          <Plot
            data={[{
              type: 'scatter',
              mode: 'lines+markers',
              x: data.temporal_importance!.map((_, i) => i),
              y: data.temporal_importance!,
              line: { color: '#10b981', width: 2 },
              marker: { color: '#10b981', size: 4 },
            }]}
            layout={temporalLayout}
            config={plotlyConfig}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}
      {importance && (
        <HorizontalBarChart features={importance.features} values={importance.values} color="#10b981" title="Feature Attribution" />
      )}
    </div>
  )
}

const ADVANCED_SUB_TABS: { key: AdvancedSubTab; label: string }[] = [
  { key: 'permutation', label: 'Permutation' },
  { key: 'shap', label: 'SHAP' },
  { key: 'attention', label: 'Attention' },
  { key: 'gradients', label: 'Gradients' },
]

function AdvancedTab({ modelId }: { modelId: string }) {
  const [subTab, setSubTab] = useState<AdvancedSubTab>('permutation')
  const [mountedSubs, setMountedSubs] = useState<Set<AdvancedSubTab>>(new Set(['permutation']))

  const handleSubTabChange = (tab: AdvancedSubTab) => {
    setSubTab(tab)
    setMountedSubs((prev) => new Set(prev).add(tab))
  }

  return (
    <div className="space-y-4">
      <TabDescription text="Advanced analysis tools for expert users. These methods offer different perspectives on the internal workings of the model." />

      {/* Sub-tab selector */}
      <div className="flex gap-1 border-b border-white/5 pb-2">
        {ADVANCED_SUB_TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => handleSubTabChange(tab.key)}
            className={`px-3 py-1 text-[11px] font-medium rounded transition-colors ${
              subTab === tab.key
                ? 'bg-white/10 text-text-primary'
                : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Sub-tab content */}
      <div>
        {subTab === 'permutation' && mountedSubs.has('permutation') && <PermutationSubTab modelId={modelId} />}
        {subTab === 'shap' && mountedSubs.has('shap') && <ShapSubTab modelId={modelId} />}
        {subTab === 'attention' && mountedSubs.has('attention') && <AttentionSubTab modelId={modelId} />}
        {subTab === 'gradients' && mountedSubs.has('gradients') && <GradientsSubTab modelId={modelId} />}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function ExplainabilityPanel({ modelId, className = '' }: ExplainabilityPanelProps) {
  const [open, setOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<TabKey>('influence')
  const [mountedTabs, setMountedTabs] = useState<Set<TabKey>>(new Set())

  const handleToggle = () => {
    if (!open) {
      setMountedTabs((prev) => new Set(prev).add('influence'))
    }
    setOpen(!open)
  }

  const handleTabChange = (tab: TabKey) => {
    setActiveTab(tab)
    setMountedTabs((prev) => new Set(prev).add(tab))
  }

  return (
    <div className={className}>
      <button
        onClick={handleToggle}
        className="flex items-center gap-2 text-sm font-semibold text-text-primary hover:text-accent-cyan transition-colors"
      >
        {open ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        Explainability
      </button>

      {open && (
        <div className="mt-3 bg-bg-card rounded-xl border border-white/5 p-4">
          {/* Tab bar - scrollable */}
          <div className="flex gap-1 mb-4 border-b border-white/10 pb-2 overflow-x-auto">
            {TABS.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.key}
                  onClick={() => handleTabChange(tab.key)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                    activeTab === tab.key
                      ? 'bg-accent-cyan/20 text-accent-cyan'
                      : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
                  }`}
                  title={tab.description}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {tab.label}
                </button>
              )
            })}
          </div>

          {/* Tab content */}
          <div>
            {activeTab === 'influence' && mountedTabs.has('influence') && <InfluenceTab modelId={modelId} />}
            {activeTab === 'fiabilite' && mountedTabs.has('fiabilite') && <FiabiliteTab modelId={modelId} />}
            {activeTab === 'memoire' && mountedTabs.has('memoire') && <MemoireTab modelId={modelId} />}
            {activeTab === 'cycles' && mountedTabs.has('cycles') && <CyclesTab modelId={modelId} />}
            {activeTab === 'avancee' && mountedTabs.has('avancee') && <AdvancedTab modelId={modelId} />}
          </div>
        </div>
      )}
    </div>
  )
}
