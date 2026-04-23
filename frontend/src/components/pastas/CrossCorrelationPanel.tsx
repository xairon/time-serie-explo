import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface Props { data: any }

export function CrossCorrelationPanel({ data }: Props) {
  if (!data?.lags_months?.length) return <p className="text-xs text-text-muted">Not enough data for cross-correlation</p>

  const shapes: any[] = []
  const annotations: any[] = []

  if (data.max_lag_months != null) {
    shapes.push({ type: 'line', x0: data.max_lag_months, x1: data.max_lag_months, y0: 0, y1: 1, yref: 'paper', line: { color: '#22d3ee', dash: 'dash', width: 1.5 } })
    annotations.push({ x: data.max_lag_months, y: 1.05, yref: 'paper', text: `Peak: ${data.max_lag_months}mo (r=${data.max_correlation})`, showarrow: false, font: { size: 10, color: '#22d3ee' } })
  }
  if (data.t95_months != null) {
    shapes.push({ type: 'line', x0: data.t95_months, x1: data.t95_months, y0: 0, y1: 1, yref: 'paper', line: { color: '#f97316', dash: 'dot', width: 1.5 } })
    annotations.push({ x: data.t95_months, y: 0.95, yref: 'paper', text: `T95: ${data.t95_months}mo`, showarrow: false, font: { size: 10, color: '#f97316' } })
  }

  return (
    <div>
      <Plot
        data={[{
          x: data.lags_months, y: data.correlation, type: 'bar' as const,
          marker: { color: data.lags_months.map((l: number) => l === data.max_lag_months ? '#22d3ee' : 'rgba(156,163,175,0.4)') },
        }]}
        layout={{ ...darkLayout, margin: { t: 20, r: 20, b: 40, l: 55 }, height: 220,
          xaxis: { ...darkLayout.xaxis, title: { text: 'Lag (months) — positive = precip leads' } },
          yaxis: { ...darkLayout.yaxis, title: { text: 'Correlation' } },
          shapes, annotations,
        }}
        config={plotlyConfig} style={{ width: '100%' }}
      />
      {data.t95_months != null && data.max_lag_months != null && (
        <p className="text-[10px] text-text-muted mt-1">
          Empirical response peak at {data.max_lag_months} months — Pastas T95: {data.t95_months} months
          {Math.abs(data.max_lag_months - data.t95_months) > 2 && <span className="text-orange-400"> (mismatch &gt;2 months)</span>}
        </p>
      )}
    </div>
  )
}
