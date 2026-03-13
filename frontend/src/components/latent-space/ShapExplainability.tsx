import Plot from 'react-plotly.js'
import type { Data, Layout } from 'plotly.js-dist-min'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import { AlertTriangle } from 'lucide-react'

interface ShapExplainabilityProps {
  shap: Record<string, unknown>
}

function accuracyColor(acc: number): string {
  if (acc >= 0.8) return 'bg-green-500/20 text-green-400'
  if (acc >= 0.5) return 'bg-amber-500/20 text-amber-400'
  return 'bg-red-500/20 text-red-400'
}

export function ShapExplainability({ shap }: ShapExplainabilityProps) {
  const data = shap as {
    feature_importance: Record<string, number>
    shap_per_cluster: Record<string, Record<string, number>>
    proxy_accuracy: number
    warning: string | null
  }

  if (!data.feature_importance || Object.keys(data.feature_importance).length === 0) {
    if (data.warning) {
      return (
        <div className="bg-bg-card rounded-xl border border-white/5 p-4">
          <h3 className="text-text-primary text-sm font-medium mb-3">SHAP Explainability</h3>
          <div className="flex items-center gap-2 bg-amber-500/10 text-amber-400 px-3 py-2 rounded-lg text-xs">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
            <span>{data.warning}</span>
          </div>
        </div>
      )
    }
    return null
  }

  const sortedFeatures = Object.entries(data.feature_importance)
    .sort((a, b) => b[1] - a[1])
    .map(([f]) => f)

  const clusterIds = Object.keys(data.shap_per_cluster).sort(
    (a, b) => Number(a) - Number(b),
  )

  const cols = Math.min(4, clusterIds.length)
  const rows = Math.ceil(clusterIds.length / cols)

  const COLORS_POS = '#ef4444'
  const COLORS_NEG = '#3b82f6'

  const traces: Data[] = []

  clusterIds.forEach((cid, idx) => {
    const shapVals = data.shap_per_cluster[cid]
    const xaxis = idx === 0 ? 'x' : `x${idx + 1}`
    const yaxis = idx === 0 ? 'y' : `y${idx + 1}`

    traces.push({
      type: 'bar',
      orientation: 'h',
      y: sortedFeatures,
      x: sortedFeatures.map((f) => shapVals[f] ?? 0),
      marker: {
        color: sortedFeatures.map((f) => (shapVals[f] ?? 0) >= 0 ? COLORS_POS : COLORS_NEG),
      },
      showlegend: false,
      hovertemplate: '%{y}: %{x:.4f}<extra>Cluster ' + cid + '</extra>',
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      xaxis: xaxis as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      yaxis: yaxis as any,
    } as Data)
  })

  const layout: Partial<Layout> = {
    ...darkLayout,
    margin: { l: 100, r: 20, t: 30, b: 30 },
    height: rows * 200 + 50,
    grid: { rows, columns: cols, pattern: 'independent' },
    annotations: clusterIds.map((cid, idx) => ({
      text: `Cluster ${cid}`,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      xref: `${idx === 0 ? 'x' : `x${idx + 1}`} domain` as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      yref: `${idx === 0 ? 'y' : `y${idx + 1}`} domain` as any,
      x: 0.5,
      y: 1.05,
      xanchor: 'center' as const,
      yanchor: 'bottom' as const,
      showarrow: false,
      font: { color: '#9ca3af', size: 10 },
    })) as Layout['annotations'],
    showlegend: false,
  }

  clusterIds.forEach((_, idx) => {
    const xKey = idx === 0 ? 'xaxis' : `xaxis${idx + 1}`
    const yKey = idx === 0 ? 'yaxis' : `yaxis${idx + 1}`
    ;(layout as Record<string, unknown>)[xKey] = {
      ...darkLayout.xaxis,
      zeroline: true,
      zerolinecolor: 'rgba(255,255,255,0.15)',
      tickfont: { size: 9 },
    }
    ;(layout as Record<string, unknown>)[yKey] = {
      ...darkLayout.yaxis,
      autorange: 'reversed',
      tickfont: { size: 9 },
    }
  })

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4">
      <div className="flex items-center gap-3 mb-1">
        <h3 className="text-text-primary text-sm font-medium">SHAP Explainability</h3>
        <span className={`px-2 py-0.5 rounded text-xs font-mono ${accuracyColor(data.proxy_accuracy)}`}>
          Proxy accuracy: {(data.proxy_accuracy * 100).toFixed(1)}%
        </span>
      </div>
      <div className="flex items-center gap-4 mb-3">
        <p className="text-text-muted text-xs">Which features drive cluster membership.</p>
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-500 inline-block" /> pushes toward</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-500 inline-block" /> pushes away</span>
        </div>
      </div>
      {data.warning && (
        <div className="flex items-center gap-2 bg-amber-500/10 text-amber-400 px-3 py-2 rounded-lg text-xs mb-3">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span>{data.warning}</span>
        </div>
      )}
      <Plot
        data={traces}
        layout={layout}
        config={plotlyConfig}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
