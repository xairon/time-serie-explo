import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Layout } from 'plotly.js-dist-min'

interface MetricsRadarProps {
  metrics: Record<string, number>
  label?: string
  color?: string
  className?: string
}

export function MetricsRadar({
  metrics,
  label = 'Modèle',
  color = '#06b6d4',
  className = '',
}: MetricsRadarProps) {
  const keys = Object.keys(metrics)
  const values = Object.values(metrics)

  const layout: Partial<Layout> = {
    ...darkLayout,
    polar: {
      bgcolor: 'transparent',
      radialaxis: {
        visible: true,
        gridcolor: 'rgba(255,255,255,0.1)',
        tickfont: { color: '#9ca3af' },
      },
      angularaxis: {
        gridcolor: 'rgba(255,255,255,0.1)',
        tickfont: { color: '#e5e7eb' },
      },
    },
    showlegend: false,
  }

  return (
    <div className={className}>
      <Plot
        data={[
          {
            type: 'scatterpolar',
            r: [...values, values[0]],
            theta: [...keys, keys[0]],
            fill: 'toself',
            fillcolor: `${color}20`,
            line: { color },
            name: label,
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
