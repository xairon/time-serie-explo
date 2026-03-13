import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Data, Layout, Shape, Annotations } from 'plotly.js-dist-min'
import type { ForecastResult } from '@/lib/types'

interface ForecastPlotProps {
  result: ForecastResult
  splitIndex?: number
  /** Input chunk length from model hyperparams */
  inputChunkLength?: number
  className?: string
}

export function ForecastPlot({
  result,
  splitIndex,
  inputChunkLength,
  className = '',
}: ForecastPlotProps) {
  const { dates, actuals, predictions } = result

  const traces: Data[] = []

  // Confidence band
  if (result.confidence_low?.length && result.confidence_high?.length) {
    traces.push({
      x: [...dates, ...([...dates].reverse())],
      y: [
        ...(result.confidence_high as number[]),
        ...([...(result.confidence_low as number[])].reverse()),
      ],
      fill: 'toself',
      fillcolor: 'rgba(99,102,241,0.1)',
      line: { color: 'transparent' },
      showlegend: false,
      hoverinfo: 'skip',
      type: 'scatter',
    })
  }

  // Ground truth (blue line — Streamlit: #2E86AB)
  traces.push({
    x: dates,
    y: actuals as number[],
    type: 'scatter',
    mode: 'lines',
    name: 'Observations',
    line: { color: '#2E86AB', width: 2 },
  })

  // Prediction (pink line with markers — Streamlit: #E91E63)
  traces.push({
    x: dates,
    y: predictions as number[],
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Prediction',
    line: { color: '#E91E63', width: 3 },
    marker: { size: 5 },
  })

  // One-step predictions (comparison mode — if different from main predictions)
  if (result.predictions_onestep && result.predictions_onestep !== predictions) {
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
  const annotations: Partial<Annotations>[] = []

  // Auto-detect prediction window from non-null predictions
  const firstPredIdx = predictions.findIndex((v) => v !== null)
  let lastPredIdx = predictions.length - 1
  while (lastPredIdx > firstPredIdx && predictions[lastPredIdx] === null) lastPredIdx--

  // Input window rectangle (light blue — Streamlit: rgba(46, 134, 171, 0.15))
  if (inputChunkLength && firstPredIdx >= 0) {
    const contextStartIdx = Math.max(0, firstPredIdx - inputChunkLength)
    if (contextStartIdx < firstPredIdx && dates[contextStartIdx] && dates[firstPredIdx]) {
      shapes.push({
        type: 'rect',
        x0: dates[contextStartIdx],
        x1: dates[firstPredIdx],
        y0: 0,
        y1: 1,
        yref: 'paper',
        fillcolor: 'rgba(46, 134, 171, 0.15)',
        line: { color: 'rgba(46, 134, 171, 0.4)', width: 1 },
        layer: 'below',
      })
      annotations.push({
        x: dates[contextStartIdx],
        y: 0,
        yref: 'paper',
        text: `Input (${inputChunkLength}j)`,
        showarrow: false,
        font: { size: 10, color: 'rgba(46, 134, 171, 0.8)' },
        xanchor: 'left',
        yanchor: 'top',
      })
    }
  }

  // Prediction window rectangle (yellow — Streamlit: rgba(255, 200, 0, 0.25))
  if (firstPredIdx >= 0 && lastPredIdx > firstPredIdx) {
    const predLen = lastPredIdx - firstPredIdx + 1
    shapes.push({
      type: 'rect',
      x0: dates[firstPredIdx],
      x1: dates[lastPredIdx],
      y0: 0,
      y1: 1,
      yref: 'paper',
      fillcolor: 'rgba(255, 200, 0, 0.25)',
      line: { color: 'rgba(255, 200, 0, 0.6)', width: 1 },
      layer: 'below',
    })
    annotations.push({
      x: dates[lastPredIdx],
      y: 1,
      yref: 'paper',
      text: `Prediction (${predLen}j)`,
      showarrow: false,
      font: { size: 10, color: 'rgba(255, 200, 0, 0.9)' },
      xanchor: 'right',
      yanchor: 'bottom',
    })
  }

  // Split line
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
    yaxis: { ...darkLayout.yaxis, title: { text: 'Piezometric level' } },
    shapes,
    annotations,
    hovermode: 'x unified',
    legend: {
      orientation: 'h',
      y: 1.02,
      yanchor: 'bottom',
      x: 0.5,
      xanchor: 'center',
      font: { color: '#9ca3af', size: 11 },
    },
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
