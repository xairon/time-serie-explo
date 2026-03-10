import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Layout } from 'plotly.js-dist-min'

interface ConvergencePlotProps {
  lossHistory: number[]
  className?: string
}

export function ConvergencePlot({ lossHistory, className = '' }: ConvergencePlotProps) {
  const iterations = lossHistory.map((_, i) => i + 1)

  const layout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Iteration' } },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Loss' } },
    margin: { ...darkLayout.margin, t: 20 },
  }

  return (
    <div className={className}>
      <h4 className="text-xs text-text-secondary uppercase mb-2">Courbe de convergence</h4>
      <Plot
        data={[
          {
            x: iterations,
            y: lossHistory,
            type: 'scatter',
            mode: 'lines',
            name: 'Loss',
            line: { color: '#06b6d4', width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(6, 182, 212, 0.08)',
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
