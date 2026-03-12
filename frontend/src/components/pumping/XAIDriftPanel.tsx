import Plot from 'react-plotly.js'
import type { Data, Layout } from 'plotly.js-dist-min'

interface XAIDriftData {
  windows: string[]
  features: string[]
  attributions: number[][]  // [features x windows]
  jsDivergences?: { window: string; js_div: number }[]
}

interface XAIDriftPanelProps {
  data?: XAIDriftData
  isPending?: boolean
}

const DARK_BASE: Partial<Layout> = {
  plot_bgcolor: 'transparent',
  paper_bgcolor: 'transparent',
  font: { color: '#e2e8f0' },
}

const DARK_AXIS = {
  gridcolor: 'rgba(255,255,255,0.05)',
  tickfont: { color: '#94a3b8' },
}

export function XAIDriftPanel({ data, isPending }: XAIDriftPanelProps) {
  if (isPending || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-48 gap-3 text-text-secondary">
        {isPending ? (
          <>
            <div className="w-8 h-8 border-2 border-accent-cyan/30 border-t-accent-cyan rounded-full animate-spin" />
            <span className="text-sm">Analyse XAI en cours (couche 2)…</span>
          </>
        ) : (
          <span className="text-sm">En attente des données XAI…</span>
        )}
      </div>
    )
  }

  const heatmapTraces: Data[] = [
    {
      type: 'heatmap',
      z: data.attributions,
      x: data.windows,
      y: data.features,
      colorscale: [
        [0, 'rgba(30,27,75,1)'],
        [0.5, 'rgba(124,58,237,0.7)'],
        [1, 'rgba(34,211,238,1)'],
      ],
      hovertemplate: 'Fenêtre: %{x}<br>Feature: %{y}<br>Attribution: %{z:.4f}<extra></extra>',
    },
  ]

  const jsTraces: Data[] = data.jsDivergences?.length
    ? [
        {
          x: data.jsDivergences.map(d => d.window),
          y: data.jsDivergences.map(d => d.js_div),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'JS divergence',
          line: { color: '#f97316', width: 2 },
          marker: { color: '#f97316', size: 5 },
          hovertemplate: '%{x}<br>JS: %{y:.4f}<extra></extra>',
        },
      ]
    : []

  return (
    <div className="space-y-4">
      <Plot
        data={heatmapTraces}
        layout={{
          ...DARK_BASE,
          title: { text: 'Attributions par fenêtre temporelle', font: { color: '#e2e8f0', size: 13 } },
          xaxis: { ...DARK_AXIS, tickangle: -45 },
          yaxis: { tickfont: { color: '#94a3b8' } },
          margin: { t: 40, r: 80, b: 80, l: 120 },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: 260 }}
      />

      {jsTraces.length > 0 && (
        <Plot
          data={jsTraces}
          layout={{
            ...DARK_BASE,
            title: { text: 'Divergence JS (drift des attributions)', font: { color: '#e2e8f0', size: 13 } },
            xaxis: { ...DARK_AXIS, tickangle: -45 },
            yaxis: { ...DARK_AXIS, rangemode: 'tozero' },
            margin: { t: 40, r: 20, b: 80, l: 60 },
            showlegend: false,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: 200 }}
        />
      )}
    </div>
  )
}
