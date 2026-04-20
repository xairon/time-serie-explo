import Plot from 'react-plotly.js'
import type { TimeSeriesData } from '@/lib/types'

const COLORS = ['#60a5fa', '#34d399', '#f97316', '#a78bfa', '#f43f5e']

interface Props {
  contributions: Record<string, TimeSeriesData>
  observed?: TimeSeriesData
}

export function ContributionsChart({ contributions, observed }: Props) {
  const entries = Object.entries(contributions)
  if (entries.length === 0) return null

  const traces: Plotly.Data[] = entries.map(([name, ts], i) => ({
    x: ts.index,
    y: ts.values,
    name,
    type: 'scatter' as const,
    mode: 'lines' as const,
    stackgroup: 'one',
    line: { color: COLORS[i % COLORS.length], width: 0 },
    fillcolor: COLORS[i % COLORS.length] + '40',
  }))

  if (observed) {
    traces.push({
      x: observed.index,
      y: observed.values,
      name: 'Observed',
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#ffffff', width: 1.5 },
    } as Plotly.Data)
  }

  return (
    <div className="bg-bg-card rounded-lg border border-white/5 p-2">
      <div className="text-xs font-semibold text-text-secondary mb-1 px-1">Stress Decomposition</div>
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: '#9ca3af', size: 10 },
          margin: { t: 10, r: 20, b: 30, l: 50 },
          height: 220,
          xaxis: { gridcolor: 'rgba(255,255,255,0.03)' },
          yaxis: { title: { text: 'm' }, gridcolor: 'rgba(255,255,255,0.05)' },
          legend: { orientation: 'h', y: -0.2, font: { size: 10 } },
        }}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
