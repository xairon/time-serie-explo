import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import {
  useFeatureImportance,
  useShapAnalysis,
  useAttentionAnalysis,
  useGradientAnalysis,
} from '@/hooks/useForecasting'
import type { Layout } from 'plotly.js-dist-min'
import type { ExplainResult } from '@/lib/types'

interface ExplainabilityPanelProps {
  modelId: string
  className?: string
}

type TabKey = 'importance' | 'shap' | 'attention' | 'gradients'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'importance', label: 'Importance' },
  { key: 'shap', label: 'SHAP' },
  { key: 'attention', label: 'Attention' },
  { key: 'gradients', label: 'Gradients' },
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
      <p className="text-xs text-accent-red mb-2">Erreur : {message}</p>
      <button onClick={onRetry} className="text-xs text-accent-cyan hover:underline">
        Réessayer
      </button>
    </div>
  )
}

function NoDataState({ label }: { label: string }) {
  return (
    <div className="text-center py-8">
      <p className="text-xs text-text-secondary">
        Pas de données {label} disponibles pour ce modèle.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Horizontal bar chart helper
// ---------------------------------------------------------------------------

function HorizontalBarChart({
  features,
  values,
  color = '#06b6d4',
  title,
}: {
  features: string[]
  values: number[]
  color?: string
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
            marker: { color },
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
// Extract features/values from feature_importance record
// ---------------------------------------------------------------------------

function extractImportance(data: ExplainResult): { features: string[]; values: number[] } | null {
  if (!data.feature_importance) return null
  const entries = Object.entries(data.feature_importance).sort(([, a], [, b]) => b - a)
  return {
    features: entries.map(([k]) => k),
    values: entries.map(([, v]) => v),
  }
}

// ---------------------------------------------------------------------------
// Tab content components
// ---------------------------------------------------------------------------

function ImportanceTab({ modelId }: { modelId: string }) {
  const mutation = useFeatureImportance()

  // Trigger on first render
  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate(modelId)
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const importance = extractImportance(mutation.data)
  if (!importance) return <NoDataState label="d'importance" />

  return <HorizontalBarChart features={importance.features} values={importance.values} color="#06b6d4" />
}

function ShapTab({ modelId }: { modelId: string }) {
  const mutation = useShapAnalysis()

  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate({ model_id: modelId })
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return (
      <ErrorState
        message={(mutation.error as Error).message}
        onRetry={() => mutation.mutate({ model_id: modelId })}
      />
    )
  if (!mutation.data) return null

  const importance = extractImportance(mutation.data)
  if (!importance) return <NoDataState label="SHAP" />

  return <HorizontalBarChart features={importance.features} values={importance.values} color="#8b5cf6" />
}

function AttentionTab({ modelId }: { modelId: string }) {
  const mutation = useAttentionAnalysis()

  if (!mutation.data && !mutation.isPending && !mutation.isError) {
    mutation.mutate({ model_id: modelId })
  }

  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError)
    return (
      <ErrorState
        message={(mutation.error as Error).message}
        onRetry={() => mutation.mutate({ model_id: modelId })}
      />
    )
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
      {encoderExtracted && (
        <HorizontalBarChart
          features={encoderExtracted.features}
          values={encoderExtracted.values}
          color="#f59e0b"
          title="Encoder Importance"
        />
      )}
      {decoderExtracted && (
        <HorizontalBarChart
          features={decoderExtracted.features}
          values={decoderExtracted.values}
          color="#f97316"
          title="Decoder Importance"
        />
      )}
      {hasHeatmap && (
        <div className="h-[350px]">
          <Plot
            data={[
              {
                type: 'heatmap',
                z: data.attention_weights!,
                colorscale: 'Viridis',
                colorbar: { title: { text: 'Weight', side: 'right' }, tickfont: { color: '#94a3b8' } },
              },
            ]}
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

function GradientsTab({ modelId }: { modelId: string }) {
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
    title: { text: 'Temporal Attribution', font: { size: 12, color: '#94a3b8' } },
  }

  return (
    <div className="space-y-4">
      {hasTemporal && (
        <div className="h-[300px]">
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines+markers',
                x: data.temporal_importance!.map((_, i) => i),
                y: data.temporal_importance!,
                line: { color: '#10b981', width: 2 },
                marker: { color: '#10b981', size: 4 },
              },
            ]}
            layout={temporalLayout}
            config={plotlyConfig}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}
      {importance && (
        <HorizontalBarChart
          features={importance.features}
          values={importance.values}
          color="#10b981"
          title="Feature Attribution"
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function ExplainabilityPanel({ modelId, className = '' }: ExplainabilityPanelProps) {
  const [open, setOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<TabKey>('importance')
  // Track which tabs have been opened so we mount them lazily but keep them alive
  const [mountedTabs, setMountedTabs] = useState<Set<TabKey>>(new Set())

  const handleToggle = () => {
    if (!open) {
      setMountedTabs((prev) => new Set(prev).add('importance'))
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
        Explicabilité
      </button>

      {open && (
        <div className="mt-3 bg-bg-card rounded-xl border border-white/5 p-4">
          {/* Tab bar */}
          <div className="flex gap-1 mb-4 border-b border-white/10 pb-2">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                onClick={() => handleTabChange(tab.key)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                  activeTab === tab.key
                    ? 'bg-accent-cyan/20 text-accent-cyan'
                    : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content -- only mount a tab once it has been selected */}
          <div>
            {activeTab === 'importance' && mountedTabs.has('importance') && (
              <ImportanceTab modelId={modelId} />
            )}
            {activeTab === 'shap' && mountedTabs.has('shap') && <ShapTab modelId={modelId} />}
            {activeTab === 'attention' && mountedTabs.has('attention') && (
              <AttentionTab modelId={modelId} />
            )}
            {activeTab === 'gradients' && mountedTabs.has('gradients') && (
              <GradientsTab modelId={modelId} />
            )}
          </div>
        </div>
      )}
    </div>
  )
}
