import Plot from 'react-plotly.js'
import type { TimeSeriesData } from '@/lib/types'

const COLORS = ['#60a5fa', '#34d399', '#f97316', '#a78bfa', '#f43f5e']

interface Props {
  contributions: Record<string, TimeSeriesData>
  observed?: TimeSeriesData
  simulated?: TimeSeriesData
}

export function ContributionsChart({ contributions, observed, simulated }: Props) {
  const entries = Object.entries(contributions)
  if (entries.length === 0) return null

  const traces: Plotly.Data[] = entries.map(([name, ts], i) => ({
    x: ts.index,
    y: ts.values,
    name,
    type: 'scatter' as const,
    mode: 'lines' as const,
    fill: 'tozeroy' as const,
    line: { color: COLORS[i % COLORS.length], width: 1.5 },
    fillcolor: COLORS[i % COLORS.length] + '30',
  }))

  if (simulated) {
    traces.push({
      x: simulated.index,
      y: simulated.values,
      name: 'Simulated',
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#22d3ee', width: 2 },
    } as Plotly.Data)
  }

  if (observed) {
    traces.push({
      x: observed.index,
      y: observed.values,
      name: 'Observed',
      type: 'scatter' as const,
      mode: 'lines' as const,
      line: { color: '#6b7280', width: 1 },
    } as Plotly.Data)
  }

  return (
    <div className="bg-bg-card rounded-lg border border-white/5 p-2">
      <div className="text-xs font-semibold text-text-secondary mb-1 px-1">Décomposition des stress</div>
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: '#9ca3af', size: 10 },
          margin: { t: 10, r: 20, b: 30, l: 50 },
          height: 250,
          xaxis: { gridcolor: 'rgba(255,255,255,0.03)' },
          yaxis: { title: { text: 'm' }, gridcolor: 'rgba(255,255,255,0.05)' },
          legend: { orientation: 'h', y: -0.15, font: { size: 10 } },
        }}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
