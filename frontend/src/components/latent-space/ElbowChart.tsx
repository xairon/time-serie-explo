import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface ElbowChartProps {
  elbow: Array<{ k: number; silhouette: number; inertia: number }>
  selectedK: number | null
  onSelectK: (k: number) => void
}

export function ElbowChart({ elbow, selectedK, onSelectK }: ElbowChartProps) {
  if (!elbow || elbow.length === 0) return null

  const ks = elbow.map((d) => d.k)
  const inertias = elbow.map((d) => d.inertia)
  const silhouettes = elbow.map((d) => d.silhouette)

  // Marker sizes: larger for selected k
  const inertiaMarkerSizes = ks.map((k) => (k === selectedK ? 10 : 5))
  const silhouetteMarkerSizes = ks.map((k) => (k === selectedK ? 10 : 5))

  return (
    <Plot
      data={[
        {
          x: ks,
          y: inertias,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Inertie',
          yaxis: 'y',
          line: { color: '#22d3ee', width: 1.5 },
          marker: { color: '#22d3ee', size: inertiaMarkerSizes },
          hovertemplate: 'k=%{x}<br>Inertia=%{y:.1f}<extra></extra>',
        },
        {
          x: ks,
          y: silhouettes,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Silhouette',
          yaxis: 'y2',
          line: { color: '#fbbf24', width: 1.5 },
          marker: { color: '#fbbf24', size: silhouetteMarkerSizes },
          hovertemplate: 'k=%{x}<br>Silhouette=%{y:.3f}<extra></extra>',
        },
      ]}
      layout={{
        ...darkLayout,
        height: 160,
        margin: { t: 10, r: 45, b: 30, l: 45 },
        xaxis: {
          ...darkLayout.xaxis,
          title: { text: 'k', font: { size: 10, color: '#9ca3af' } },
          dtick: 1,
        },
        yaxis: {
          ...darkLayout.yaxis,
          title: { text: 'Inertia', font: { size: 10, color: '#22d3ee' } },
          side: 'left',
        },
        yaxis2: {
          ...darkLayout.yaxis,
          title: { text: 'Silhouette', font: { size: 10, color: '#fbbf24' } },
          side: 'right',
          overlaying: 'y',
        },
        legend: {
          ...darkLayout.legend,
          orientation: 'h',
          x: 0.5,
          xanchor: 'center',
          y: 1.15,
          font: { size: 9, color: '#9ca3af' },
        },
        hovermode: 'x unified',
      }}
      config={{
        ...plotlyConfig,
        displayModeBar: false,
      }}
      style={{ width: '100%', height: 160 }}
      onClick={(event) => {
        const point = event.points?.[0]
        if (point?.x != null) {
          onSelectK(point.x as number)
        }
      }}
    />
  )
}
