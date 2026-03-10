import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Data, Layout, Shape } from 'plotly.js-dist-min'
import type { ForecastResult } from '@/lib/types'

interface ForecastPlotProps {
  result: ForecastResult
  splitIndex?: number
  className?: string
}

export function ForecastPlot({ result, splitIndex, className = '' }: ForecastPlotProps) {
  const { dates, actuals, predictions, confidence_low, confidence_high } = result

  const traces: Data[] = []

  // Confidence band
  if (confidence_low && confidence_high) {
    traces.push({
      x: [...dates, ...([...dates].reverse())],
      y: [
        ...(confidence_high as number[]),
        ...([...(confidence_low as number[])].reverse()),
      ],
      fill: 'toself',
      fillcolor: 'rgba(99,102,241,0.1)',
      line: { color: 'transparent' },
      showlegend: false,
      hoverinfo: 'skip',
      type: 'scatter',
    })
  }

  // Actuals
  traces.push({
    x: dates,
    y: actuals as number[],
    type: 'scatter',
    mode: 'lines',
    name: 'Observations',
    line: { color: '#e5e7eb', width: 2 },
  })

  // Predictions (autoregressive)
  traces.push({
    x: dates,
    y: predictions as number[],
    type: 'scatter',
    mode: 'lines',
    name: result.predictions_onestep ? 'Autoregressif' : 'Prédictions',
    line: { color: '#06b6d4', width: 2 },
  })

  // One-step predictions (comparison mode)
  if (result.predictions_onestep) {
    traces.push({
      x: dates,
      y: result.predictions_onestep as number[],
      type: 'scatter',
      mode: 'lines',
      name: 'One-step',
      line: { color: '#a78bfa', width: 2, dash: 'dot' },
    })
  }

  const shapes: Partial<Shape>[] = []
  if (splitIndex !== undefined && splitIndex < dates.length) {
    shapes.push({
      type: 'line',
      x0: dates[splitIndex],
      x1: dates[splitIndex],
      y0: 0,
      y1: 1,
      yref: 'paper',
      line: { color: '#f59e0b', width: 1, dash: 'dash' },
    })
  }

  const layout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Date' } },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Niveau piézométrique' } },
    shapes,
  }

  return (
    <div className={className}>
      <Plot
        data={traces}
        layout={layout}
        config={plotlyConfig}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
