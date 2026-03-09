import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Layout } from 'plotly.js-dist-min'

interface CorrelationMatrixProps {
  labels: string[]
  matrix: number[][]
  className?: string
}

export function CorrelationMatrix({ labels, matrix, className = '' }: CorrelationMatrixProps) {
  const layout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, tickangle: -45 },
    yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const },
    margin: { t: 20, r: 20, b: 80, l: 80 },
  }

  return (
    <div className={className}>
      <Plot
        data={[
          {
            z: matrix,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: [
              [0, '#ef4444'],
              [0.5, '#1f2937'],
              [1, '#06b6d4'],
            ],
            zmin: -1,
            zmax: 1,
            showscale: true,
            colorbar: { tickfont: { color: '#9ca3af' } },
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
