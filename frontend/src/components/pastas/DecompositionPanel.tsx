import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface Props { data: any }

export function DecompositionPanel({ data }: Props) {
  if (!data?.index?.length) return <p className="text-xs text-text-muted">Not enough data for decomposition (need ≥5 years)</p>

  const baseLayout = { ...darkLayout, margin: { t: 5, r: 20, b: 5, l: 55 }, height: 100, showlegend: false }

  return (
    <div className="space-y-1">
      <div className="flex gap-3 mb-2">
        <span className="text-[10px] text-text-muted">Trend strength: <span className="text-text-primary font-mono">{data.trend_strength}</span></span>
        <span className="text-[10px] text-text-muted">Seasonal strength: <span className="text-text-primary font-mono">{data.seasonal_strength}</span></span>
      </div>
      {[
        { y: data.observed, label: 'Observed', color: '#9ca3af' },
        { y: data.trend, label: 'Trend', color: '#22d3ee' },
        { y: data.seasonal, label: 'Seasonal', color: '#a78bfa' },
        { y: data.residual, label: 'Residual', color: '#f97316' },
      ].map(({ y, label, color }) => (
        <div key={label} className="relative">
          <span className="absolute top-1 left-1 text-[9px] text-text-muted z-10">{label}</span>
          <Plot
            data={[{ x: data.index, y, type: 'scatter' as const, mode: 'lines' as const, line: { color, width: 1.2 } }]}
            layout={baseLayout} config={plotlyConfig} style={{ width: '100%' }}
          />
        </div>
      ))}
    </div>
  )
}
