import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Layout } from 'plotly.js-dist-min'
import type { CounterfactualResult } from '@/lib/types'

interface CFOverlayPlotProps {
  result: CounterfactualResult
  className?: string
}

export function CFOverlayPlot({ result, className = '' }: CFOverlayPlotProps) {
  const inner = result.result
  if (!inner) return null
  const { dates, original, counterfactual } = inner

  const layout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Date' } },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Niveau piézométrique' } },
  }

  return (
    <div className={className}>
      <Plot
        data={[
          {
            x: dates,
            y: original,
            type: 'scatter',
            mode: 'lines',
            name: 'Original',
            line: { color: '#e5e7eb', width: 2 },
          },
          {
            x: dates,
            y: counterfactual,
            type: 'scatter',
            mode: 'lines',
            name: 'Contrefactuel',
            line: { color: '#06b6d4', width: 2, dash: 'dash' },
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
