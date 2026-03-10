import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Data, Layout, Shape } from 'plotly.js-dist-min'
import type { ForecastResult } from '@/lib/types'

interface ForecastPlotProps {
  result: ForecastResult
  splitIndex?: number
  /** Start/end date indices for context (input) window highlight */
  contextWindow?: { start: number; end: number }
  /** Start/end date indices for prediction window highlight */
  predictionWindow?: { start: number; end: number }
  className?: string
}

export function ForecastPlot({ result, splitIndex, contextWindow, predictionWindow, className = '' }: ForecastPlotProps) {
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

  // Auto-detect context and prediction windows from split index or predictions
  const effectiveContextWindow = contextWindow ?? (() => {
    if (splitIndex !== undefined && splitIndex > 0) {
      return { start: 0, end: splitIndex }
    }
    // Detect from predictions: context = where predictions are null, prediction = where they exist
    const firstPredIdx = predictions.findIndex((v) => v !== null)
    if (firstPredIdx > 0) {
      return { start: 0, end: firstPredIdx }
    }
    return undefined
  })()

  const effectivePredictionWindow = predictionWindow ?? (() => {
    if (splitIndex !== undefined && splitIndex < dates.length) {
      return { start: splitIndex, end: dates.length - 1 }
    }
    const firstPredIdx = predictions.findIndex((v) => v !== null)
    if (firstPredIdx >= 0) {
      let lastPredIdx = predictions.length - 1
      while (lastPredIdx > firstPredIdx && predictions[lastPredIdx] === null) lastPredIdx--
      return { start: firstPredIdx, end: lastPredIdx }
    }
    return undefined
  })()

  // Context window rectangle (light blue background)
  if (effectiveContextWindow && effectiveContextWindow.end > effectiveContextWindow.start) {
    shapes.push({
      type: 'rect',
      x0: dates[effectiveContextWindow.start],
      x1: dates[effectiveContextWindow.end],
      y0: 0,
      y1: 1,
      yref: 'paper',
      fillcolor: 'rgba(59, 130, 246, 0.06)',
      line: { width: 0 },
      layer: 'below',
    })
  }

  // Prediction window rectangle (light yellow background)
  if (effectivePredictionWindow && effectivePredictionWindow.end > effectivePredictionWindow.start) {
    shapes.push({
      type: 'rect',
      x0: dates[effectivePredictionWindow.start],
      x1: dates[effectivePredictionWindow.end],
      y0: 0,
      y1: 1,
      yref: 'paper',
      fillcolor: 'rgba(234, 179, 8, 0.06)',
      line: { width: 0 },
      layer: 'below',
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
