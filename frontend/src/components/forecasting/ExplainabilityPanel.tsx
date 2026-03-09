import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import { useFeatureImportance } from '@/hooks/useForecasting'
import type { Layout } from 'plotly.js-dist-min'

interface ExplainabilityPanelProps {
  modelId: string
  className?: string
}

export function ExplainabilityPanel({ modelId, className = '' }: ExplainabilityPanelProps) {
  const [open, setOpen] = useState(false)
  const fiMutation = useFeatureImportance()

  const handleToggle = () => {
    if (!open && !fiMutation.data) {
      fiMutation.mutate(modelId)
    }
    setOpen(!open)
  }

  const layout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Importance' } },
    yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const },
    margin: { t: 10, r: 20, b: 40, l: 120 },
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
          {fiMutation.isPending && (
            <div className="h-[200px] bg-bg-hover rounded-lg animate-pulse" />
          )}
          {fiMutation.isError && (
            <div className="text-center py-8">
              <p className="text-xs text-accent-red mb-2">
                Erreur : {(fiMutation.error as Error).message}
              </p>
              <button
                onClick={() => fiMutation.mutate(modelId)}
                className="text-xs text-accent-cyan hover:underline"
              >
                Réessayer
              </button>
            </div>
          )}
          {fiMutation.data && (
            <div className="h-[300px]">
              <Plot
                data={[
                  {
                    type: 'bar',
                    orientation: 'h',
                    y: fiMutation.data.features,
                    x: fiMutation.data.importances,
                    marker: { color: '#06b6d4' },
                  },
                ]}
                layout={layout}
                config={plotlyConfig}
                useResizeHandler
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
