import Plot from 'react-plotly.js'
import type { Data, Layout } from 'plotly.js-dist-min'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface TemporalPrototypesProps {
  prototypes: Record<string, unknown>[]
}

const COLORS = [
  '#06b6d4', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981',
  '#ec4899', '#3b82f6', '#f97316', '#14b8a6', '#a855f7',
]

export function TemporalPrototypes({ prototypes }: TemporalPrototypesProps) {
  const protos = prototypes as {
    cluster_id: number
    medoid_id: string
    n_members: number
    dates: string[]
    medoid_values: (number | null)[]
    p10: (number | null)[]
    p90: (number | null)[]
  }[]

  if (protos.length === 0) return null

  // Compute shared Y-axis range across all prototypes
  let yMin = Infinity
  let yMax = -Infinity
  for (const proto of protos) {
    for (const arr of [proto.medoid_values, proto.p10, proto.p90]) {
      for (const v of arr) {
        if (v != null && isFinite(v)) {
          if (v < yMin) yMin = v
          if (v > yMax) yMax = v
        }
      }
    }
  }
  const yPad = (yMax - yMin) * 0.05
  const sharedYRange: [number, number] = [yMin - yPad, yMax + yPad]

  const cols = Math.min(4, protos.length)
  const rows = Math.ceil(protos.length / cols)

  const traces: Data[] = []
  const annotations: Partial<Layout['annotations']>[number][] = []

  protos.forEach((proto, idx) => {
    const xaxis = idx === 0 ? 'x' : `x${idx + 1}`
    const yaxis = idx === 0 ? 'y' : `y${idx + 1}`
    const color = COLORS[idx % COLORS.length]

    // P10/P90 envelope (fill)
    traces.push({
      type: 'scatter',
      x: [...proto.dates, ...proto.dates.slice().reverse()],
      y: [...(proto.p90 as number[]), ...(proto.p10 as number[]).slice().reverse()],
      fill: 'toself',
      fillcolor: color + '26',
      line: { color: 'transparent' },
      showlegend: false,
      hoverinfo: 'skip',
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      xaxis: xaxis as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      yaxis: yaxis as any,
    } as Data)

    // Medoid line
    traces.push({
      type: 'scatter',
      x: proto.dates,
      y: proto.medoid_values as number[],
      mode: 'lines',
      line: { color, width: 1.5 },
      name: `Cluster ${proto.cluster_id}`,
      showlegend: false,
      hovertemplate: '%{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>',
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      xaxis: xaxis as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      yaxis: yaxis as any,
    } as Data)

    annotations.push({
      text: `Cluster ${proto.cluster_id} (n=${proto.n_members}) — ${proto.medoid_id}`,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      xref: `${xaxis} domain` as any,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      yref: `${yaxis} domain` as any,
      x: 0.5,
      y: 1.05,
      xanchor: 'center',
      yanchor: 'bottom',
      showarrow: false,
      font: { color: '#9ca3af', size: 10 },
    })
  })

  const layout: Partial<Layout> = {
    ...darkLayout,
    margin: { l: 50, r: 20, t: 30, b: 30 },
    height: rows * 200 + 50,
    grid: { rows, columns: cols, pattern: 'independent' },
    annotations: annotations as Layout['annotations'],
    showlegend: false,
  }

  protos.forEach((_, idx) => {
    const xKey = idx === 0 ? 'xaxis' : `xaxis${idx + 1}`
    const yKey = idx === 0 ? 'yaxis' : `yaxis${idx + 1}`
    ;(layout as Record<string, unknown>)[xKey] = {
      ...darkLayout.xaxis,
      tickfont: { size: 9 },
    }
    ;(layout as Record<string, unknown>)[yKey] = {
      ...darkLayout.yaxis,
      tickfont: { size: 9 },
      range: sharedYRange,
    }
  })

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4">
      <h3 className="text-text-primary text-sm font-medium mb-3">Temporal Prototypes</h3>
      <Plot
        data={traces}
        layout={layout}
        config={plotlyConfig}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
