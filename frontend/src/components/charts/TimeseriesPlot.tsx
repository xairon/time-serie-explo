import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Data, Layout } from 'plotly.js-dist-min'

interface TimeseriesPlotProps {
  dates: string[]
  values: (number | null)[]
  label?: string
  color?: string
  confidenceLow?: (number | null)[]
  confidenceHigh?: (number | null)[]
  title?: string
  yAxisLabel?: string
  className?: string
}

export function TimeseriesPlot({
  dates,
  values,
  label = 'Valeur',
  color = '#06b6d4',
  confidenceLow,
  confidenceHigh,
  title,
  yAxisLabel,
  className = '',
}: TimeseriesPlotProps) {
  const traces: Data[] = []

  if (confidenceLow && confidenceHigh) {
    traces.push({
      x: [...dates, ...([...dates].reverse())],
      y: [...(confidenceHigh as number[]), ...([...(confidenceLow as number[])].reverse())],
      fill: 'toself',
      fillcolor: `${color}15`,
      line: { color: 'transparent' },
      showlegend: false,
      hoverinfo: 'skip',
      type: 'scatter',
    })
  }

  traces.push({
    x: dates,
    y: values as number[],
    type: 'scatter',
    mode: 'lines',
    name: label,
    line: { color, width: 2 },
  })

  const layout: Partial<Layout> = {
    ...darkLayout,
    title: title ? { text: title, font: { size: 14, color: '#e5e7eb' } } : undefined,
    yaxis: { ...darkLayout.yaxis, title: yAxisLabel ? { text: yAxisLabel } : undefined },
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
