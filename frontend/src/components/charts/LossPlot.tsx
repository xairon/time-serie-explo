import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Layout } from 'plotly.js-dist-min'

interface LossPlotProps {
  trainLoss: number[]
  valLoss: number[]
  className?: string
}

export function LossPlot({ trainLoss, valLoss, className = '' }: LossPlotProps) {
  const epochs = trainLoss.map((_, i) => i + 1)

  const layout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Epoch' } },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Loss' } },
  }

  return (
    <div className={className}>
      <Plot
        data={[
          {
            x: epochs,
            y: trainLoss,
            type: 'scatter',
            mode: 'lines',
            name: 'Train',
            line: { color: '#06b6d4', width: 2 },
          },
          {
            x: epochs.slice(0, valLoss.length),
            y: valLoss,
            type: 'scatter',
            mode: 'lines',
            name: 'Validation',
            line: { color: '#f59e0b', width: 2 },
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
