import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Data, Layout, Shape, Annotations } from 'plotly.js-dist-min'
import type { ForecastResult } from '@/lib/types'

interface ForecastPlotProps {
  result: ForecastResult
  splitIndex?: number
  /** Start/end date indices for context (input) window highlight */
  contextWindow?: { start: number; end: number }
  /** Start/end date indices for prediction window highlight */
  predictionWindow?: { start: number; end: number }
  /** Input chunk length from model hyperparams */
  inputChunkLength?: number
  /** Output chunk length from model hyperparams */
  outputChunkLength?: number
  className?: string
}

export function ForecastPlot({
  result,
  splitIndex,
  contextWindow,
  predictionWindow,
  inputChunkLength,
  outputChunkLength,
  className = '',
}: ForecastPlotProps) {
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

  // Actuals (gray line as background context)
  traces.push({
    x: dates,
    y: actuals as number[],
    type: 'scatter',
    mode: 'lines',
    name: 'Observations',
    line: { color: '#9ca3af', width: 2 },
  })

  // Predictions (cyan overlay)
  traces.push({
    x: dates,
    y: predictions as number[],
    type: 'scatter',
    mode: 'lines',
    name: result.predictions_onestep ? 'Autoregressif' : 'Predictions',
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
  const annotations: Partial<Annotations>[] = []

  // Auto-detect context and prediction windows from split index or predictions
  let effectiveContextWindow = contextWindow
  let effectivePredictionWindow = predictionWindow

  if (!effectiveContextWindow) {
    if (splitIndex !== undefined && splitIndex > 0) {
      effectiveContextWindow = { start: 0, end: splitIndex }
    } else {
      const firstPredIdx = predictions.findIndex((v) => v !== null)
      if (firstPredIdx > 0) {
        effectiveContextWindow = { start: 0, end: firstPredIdx }
      }
    }
  }

  if (!effectivePredictionWindow) {
    if (splitIndex !== undefined && splitIndex < dates.length) {
      effectivePredictionWindow = { start: splitIndex, end: dates.length - 1 }
    } else {
      const firstPredIdx = predictions.findIndex((v) => v !== null)
      if (firstPredIdx >= 0) {
        let lastPredIdx = predictions.length - 1
        while (lastPredIdx > firstPredIdx && predictions[lastPredIdx] === null) lastPredIdx--
        effectivePredictionWindow = { start: firstPredIdx, end: lastPredIdx }
      }
    }
  }

  // If we have chunk lengths, try to compute windows from predictions
  if (inputChunkLength && !contextWindow) {
    const firstPredIdx = predictions.findIndex((v) => v !== null)
    if (firstPredIdx >= 0) {
      const contextStart = Math.max(0, firstPredIdx - inputChunkLength)
      effectiveContextWindow = { start: contextStart, end: firstPredIdx }
    }
  }

  if (outputChunkLength && !predictionWindow) {
    const firstPredIdx = predictions.findIndex((v) => v !== null)
    if (firstPredIdx >= 0) {
      const predEnd = Math.min(dates.length - 1, firstPredIdx + outputChunkLength - 1)
      effectivePredictionWindow = { start: firstPredIdx, end: predEnd }
    }
  }

  // Context window rectangle (light blue, 15% opacity)
  if (effectiveContextWindow && effectiveContextWindow.end > effectiveContextWindow.start) {
    const contextLen = effectiveContextWindow.end - effectiveContextWindow.start
    shapes.push({
      type: 'rect',
      x0: dates[effectiveContextWindow.start],
      x1: dates[effectiveContextWindow.end],
      y0: 0,
      y1: 1,
      yref: 'paper',
      fillcolor: 'rgba(59, 130, 246, 0.15)',
      line: { width: 0 },
      layer: 'below',
    })
    annotations.push({
      x: dates[Math.floor((effectiveContextWindow.start + effectiveContextWindow.end) / 2)],
      y: 1,
      yref: 'paper',
      text: `Input (${contextLen}j)`,
      showarrow: false,
      font: { size: 10, color: 'rgba(147, 197, 253, 0.8)' },
      yanchor: 'bottom',
    })
  }

  // Prediction window rectangle (light orange, 25% opacity)
  if (effectivePredictionWindow && effectivePredictionWindow.end > effectivePredictionWindow.start) {
    const predLen = effectivePredictionWindow.end - effectivePredictionWindow.start
    shapes.push({
      type: 'rect',
      x0: dates[effectivePredictionWindow.start],
      x1: dates[effectivePredictionWindow.end],
      y0: 0,
      y1: 1,
      yref: 'paper',
      fillcolor: 'rgba(251, 146, 60, 0.25)',
      line: { width: 0 },
      layer: 'below',
    })
    annotations.push({
      x: dates[Math.floor((effectivePredictionWindow.start + effectivePredictionWindow.end) / 2)],
      y: 1,
      yref: 'paper',
      text: `Prediction (${predLen}j)`,
      showarrow: false,
      font: { size: 10, color: 'rgba(253, 186, 116, 0.9)' },
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
    yaxis: { ...darkLayout.yaxis, title: { text: 'Niveau piezometrique' } },
    shapes,
    annotations,
    hovermode: 'x unified',
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
