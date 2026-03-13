import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Layout, Shape } from 'plotly.js-dist-min'
import type { IPSBoundsRow } from '@/lib/types'

const IPS_CLASSES = [
  'very_low',
  'low',
  'moderately_low',
  'normal',
  'moderately_high',
  'high',
  'very_high',
] as const

interface IPSBandsChartProps {
  testDates: string[]
  testValues: (number | null)[]
  ipsBounds: IPSBoundsRow[]
  ipsColors: Record<string, string>
  ipsLabels: Record<string, string>
  contextStart: string
  contextEnd: string
  predStart: string
  predEnd: string
  cfDates?: string[]
  cfOriginal?: number[]
  cfCounterfactual?: number[]
  className?: string
}

export function IPSBandsChart({
  testDates,
  testValues,
  ipsBounds,
  ipsColors,
  ipsLabels,
  contextStart,
  contextEnd,
  predStart,
  predEnd,
  cfDates,
  cfOriginal,
  cfCounterfactual,
  className = '',
}: IPSBandsChartProps) {
  // Build IPS band shapes
  const bandShapes: Partial<Shape>[] = ipsBounds.flatMap((row) =>
    IPS_CLASSES.map((cls) => {
      const lower = Number(row[`${cls}_lower`])
      const upper = Number(row[`${cls}_upper`])
      const color = ipsColors[cls] ?? '#888888'
      return {
        type: 'rect' as const,
        xref: 'x' as const,
        yref: 'y' as const,
        x0: row.month_start,
        x1: row.month_end,
        y0: lower,
        y1: upper,
        fillcolor: color,
        opacity: 0.12,
        line: { width: 0 },
        layer: 'below' as const,
      }
    }),
  )

  // Context and prediction vrects
  const contextShape: Partial<Shape> = {
    type: 'rect',
    xref: 'x',
    yref: 'paper',
    x0: contextStart,
    x1: contextEnd,
    y0: 0,
    y1: 1,
    fillcolor: 'rgba(96,165,250,0.1)',
    line: { color: 'rgba(96,165,250,0.5)', width: 1, dash: 'dot' },
    layer: 'below',
  }

  const predShape: Partial<Shape> = {
    type: 'rect',
    xref: 'x',
    yref: 'paper',
    x0: predStart,
    x1: predEnd,
    y0: 0,
    y1: 1,
    fillcolor: 'rgba(250,204,21,0.1)',
    line: { color: 'rgba(250,204,21,0.5)', width: 1, dash: 'dot' },
    layer: 'below',
  }

  const shapes = [...bandShapes, contextShape, predShape]

  // Add invisible traces for IPS class legend entries
  const legendTraces = IPS_CLASSES.map((cls) => ({
    x: [null],
    y: [null],
    type: 'scatter' as const,
    mode: 'markers' as const,
    name: ipsLabels[cls] ?? cls,
    marker: { color: ipsColors[cls] ?? '#888888', size: 10, symbol: 'square' as const },
    showlegend: true,
  }))

  // Ground truth trace
  const groundTruth = {
    x: testDates,
    y: testValues,
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: 'Mesures',
    line: { color: '#60a5fa', width: 2 },
    connectgaps: false,
  }

  // CF overlay traces
  const cfTraces =
    cfDates && cfOriginal && cfCounterfactual
      ? [
          {
            x: cfDates,
            y: cfOriginal,
            type: 'scatter' as const,
            mode: 'lines' as const,
            name: 'Original',
            line: { color: '#9ca3af', width: 2 },
          },
          {
            x: cfDates,
            y: cfCounterfactual,
            type: 'scatter' as const,
            mode: 'lines' as const,
            name: 'Counterfactual',
            line: { color: '#06b6d4', width: 2, dash: 'dash' as const },
          },
        ]
      : []

  const layout: Partial<Layout> = {
    ...darkLayout,
    xaxis: {
      ...darkLayout.xaxis,
      title: { text: 'Date' },
    },
    yaxis: {
      ...darkLayout.yaxis,
      title: { text: 'Piezometric level (m NGF)' },
    },
    legend: {
      ...darkLayout.legend,
      orientation: 'h',
      y: 1.12,
      x: 0.5,
      xanchor: 'center',
    },
    shapes,
  }

  return (
    <div className={className}>
      <Plot
        data={[...legendTraces, groundTruth, ...cfTraces]}
        layout={layout}
        config={plotlyConfig}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
