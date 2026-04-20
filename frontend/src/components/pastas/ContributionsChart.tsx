import Plot from 'react-plotly.js'
import type { TimeSeriesData } from '@/lib/types'

const COLORS = ['#60a5fa', '#34d399', '#f97316', '#a78bfa', '#f43f5e']

const LABELS: Record<string, string> = {
  recharge: 'Recharge (P − f·E)',
  constant_d: 'Niveau de base',
}

interface Props {
  contributions: Record<string, TimeSeriesData>
}

export function ContributionsChart({ contributions }: Props) {
  const entries = Object.entries(contributions)
  if (entries.length === 0) return null

  const traces: Plotly.Data[] = entries.map(([name, ts], i) => ({
    x: ts.index,
    y: ts.values,
    name: LABELS[name] ?? name,
    type: 'scatter' as const,
    mode: 'lines' as const,
    fill: 'tozeroy' as const,
    line: { color: COLORS[i % COLORS.length], width: 1.5 },
    fillcolor: COLORS[i % COLORS.length] + '25',
  }))

  return (
    <div className="bg-bg-card rounded-lg border border-white/5 p-3">
      <div className="text-xs font-semibold text-text-secondary mb-1">
        Contributions de chaque stress
      </div>
      <p className="text-[10px] text-text-muted mb-2">
        Le signal simulé est la somme de ces contributions. La recharge montre l'effet de la pluie et de l'ETP sur la nappe.
      </p>
      <Plot
        data={traces}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: '#9ca3af', size: 10 },
          margin: { t: 10, r: 20, b: 30, l: 50 },
          height: 200,
          xaxis: { gridcolor: 'rgba(255,255,255,0.03)' },
          yaxis: { title: { text: 'Contribution (m)' }, gridcolor: 'rgba(255,255,255,0.05)' },
          legend: { orientation: 'h', y: -0.2, font: { size: 10 } },
        }}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
