import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface Props { data: any }

export function SpectralPanel({ data }: Props) {
  if (!data?.frequencies?.length) return <p className="text-xs text-text-muted">Not enough data for spectral analysis (need ≥1 year)</p>

  return (
    <Plot
      data={[
        { x: data.periods_days, y: data.psd_observed, type: 'scatter' as const, mode: 'lines' as const,
          name: 'Observed', line: { color: '#9ca3af', width: 1.5 } },
        { x: data.periods_days, y: data.psd_simulated, type: 'scatter' as const, mode: 'lines' as const,
          name: 'Simulated', line: { color: '#22d3ee', width: 1.5 } },
      ]}
      layout={{ ...darkLayout, margin: { t: 10, r: 20, b: 40, l: 55 }, height: 250,
        xaxis: { ...darkLayout.xaxis, type: 'log' as const, title: { text: 'Period (days)' }, autorange: 'reversed' as const },
        yaxis: { ...darkLayout.yaxis, type: 'log' as const, title: { text: 'PSD' } },
        shapes: [
          { type: 'line', x0: 365, x1: 365, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(239,68,68,0.3)', dash: 'dot' } },
          { type: 'line', x0: 182, x1: 182, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(251,191,36,0.3)', dash: 'dot' } },
        ],
        annotations: [
          { x: Math.log10(365), y: 1.02, yref: 'paper', text: 'Annual', showarrow: false, font: { size: 9, color: '#ef4444' } },
          { x: Math.log10(182), y: 1.02, yref: 'paper', text: 'Semi-annual', showarrow: false, font: { size: 9, color: '#fbbf24' } },
        ],
      }}
      config={plotlyConfig} style={{ width: '100%' }}
    />
  )
}
