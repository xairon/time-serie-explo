import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Layout } from 'plotly.js-dist-min'

interface RadarPlotProps {
  theta: Record<string, number>
  className?: string
}

export function RadarPlot({ theta, className = '' }: RadarPlotProps) {
  const keys = Object.keys(theta)
  const values = Object.values(theta)

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
            fillcolor: 'rgba(99,102,241,0.15)',
            line: { color: '#6366f1', width: 2 },
            name: 'Theta',
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
