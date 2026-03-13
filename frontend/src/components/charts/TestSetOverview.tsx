import { memo } from 'react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Data, Layout, Shape, Annotations } from 'plotly.js-dist-min'
import type { ForecastResult } from '@/lib/types'

interface TestSetOverviewProps {
  /** All test set dates */
  testDates: string[]
  /** All test set values (target column) */
  testValues: (number | null)[]
  /** Current slider index in test set */
  sliderIdx: number
  /** Input chunk length */
  inputChunkLength: number
  /** Output chunk length */
  outputChunkLength: number
  /** Current window forecast result (if available) */
  windowResult?: ForecastResult | null
  className?: string
}

export const TestSetOverview = memo(function TestSetOverview({
  testDates,
  testValues,
  sliderIdx,
  inputChunkLength,
  outputChunkLength,
  windowResult,
  className = '',
}: TestSetOverviewProps) {
  const traces: Data[] = []

  // Full test set line (blue)
  traces.push({
    x: testDates,
    y: testValues as number[],
    type: 'scatter',
    mode: 'lines',
    name: 'Test set',
    line: { color: '#2E86AB', width: 1.5 },
    hovertemplate: '%{x|%d/%m/%Y}<br>%{y:.4f}<extra></extra>',
  })

  // Overlay current window predictions if available
  if (windowResult && windowResult.dates.length > 0) {
    traces.push({
      x: windowResult.dates,
      y: windowResult.predictions as number[],
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Prediction',
      line: { color: '#E91E63', width: 2.5 },
      marker: { size: 4 },
      hovertemplate: '%{x|%d/%m/%Y}<br>%{y:.4f}<extra></extra>',
    })
  }

  const shapes: Partial<Shape>[] = []
  const annotations: Partial<Annotations>[] = []

  // Input window rectangle (light blue)
  const contextStartIdx = Math.max(0, sliderIdx - inputChunkLength)
  if (contextStartIdx < sliderIdx && testDates[contextStartIdx] && testDates[sliderIdx]) {
    shapes.push({
      type: 'rect',
      x0: testDates[contextStartIdx],
      x1: testDates[sliderIdx],
      y0: 0,
      y1: 1,
      yref: 'paper',
      fillcolor: 'rgba(46, 134, 171, 0.15)',
      line: { color: 'rgba(46, 134, 171, 0.4)', width: 1 },
      layer: 'below',
    })
    annotations.push({
      x: testDates[contextStartIdx],
      y: 1,
      yref: 'paper',
      text: `Input (${inputChunkLength}j)`,
      showarrow: false,
      font: { size: 9, color: 'rgba(46, 134, 171, 0.8)' },
      xanchor: 'left',
      yanchor: 'bottom',
    })
  }

  // Prediction window rectangle (yellow)
  const predEndIdx = Math.min(sliderIdx + outputChunkLength - 1, testDates.length - 1)
  if (testDates[sliderIdx] && testDates[predEndIdx]) {
    shapes.push({
      type: 'rect',
      x0: testDates[sliderIdx],
      x1: testDates[predEndIdx],
      y0: 0,
      y1: 1,
      yref: 'paper',
      fillcolor: 'rgba(255, 200, 0, 0.25)',
      line: { color: 'rgba(255, 200, 0, 0.6)', width: 1 },
      layer: 'below',
    })
    annotations.push({
      x: testDates[predEndIdx],
      y: 0,
      yref: 'paper',
      text: `Prediction (${outputChunkLength}j)`,
      showarrow: false,
      font: { size: 9, color: 'rgba(255, 200, 0, 0.9)' },
      xanchor: 'right',
      yanchor: 'top',
    })
  }

  const layout: Partial<Layout> = {
    ...darkLayout,
    height: 250,
    margin: { t: 20, r: 20, b: 30, l: 50 },
    xaxis: { ...darkLayout.xaxis, title: { text: '' } },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Piezometric level' } },
    shapes,
    annotations,
    datarevision: sliderIdx,
    hovermode: 'x unified',
    legend: {
      orientation: 'h',
      y: 1.05,
      yanchor: 'bottom',
      x: 0.5,
      xanchor: 'center',
      font: { color: '#9ca3af', size: 10 },
    },
  }

  return (
    <div className={className} style={{ minHeight: 270 }}>
      <Plot
        data={traces}
        layout={layout}
        config={plotlyConfig}
        revision={sliderIdx}
        useResizeHandler
        style={{ width: '100%', height: 270 }}
      />
    </div>
  )
})
