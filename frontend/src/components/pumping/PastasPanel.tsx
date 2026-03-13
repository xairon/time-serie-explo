import Plot from 'react-plotly.js'
import type { Data, Layout, Shape } from 'plotly.js-dist-min'

interface PastasMetrics {
  evp?: number
  rmse?: number
  nse?: number
}

interface PastasPanelProps {
  metrics?: PastasMetrics
  residuals?: { time: string; value: number }[]
  acf?: number[]
  pacf?: number[]
  changepointDates?: string[]
}

const DARK_BASE: Partial<Layout> = {
  plot_bgcolor: 'transparent',
  paper_bgcolor: 'transparent',
  font: { color: '#e2e8f0' },
}

const DARK_AXIS = {
  gridcolor: 'rgba(255,255,255,0.05)',
  linecolor: 'rgba(255,255,255,0.1)',
  tickfont: { color: '#94a3b8' },
}

function MetricBadge({ label, value, unit = '' }: { label: string; value?: number; unit?: string }) {
  if (value == null) return null
  const good = label === 'EVP' ? value >= 70 : label === 'NSE' ? value >= 0.7 : value <= 0.05
  return (
    <div className="bg-bg-primary/50 rounded-lg px-3 py-2 flex flex-col items-center gap-0.5">
      <span className="text-xs text-text-secondary">{label}</span>
      <span className={`text-lg font-semibold ${good ? 'text-accent-green' : 'text-accent-orange'}`}>
        {value.toFixed(2)}{unit}
      </span>
    </div>
  )
}

export function PastasPanel({ metrics, residuals, acf, pacf, changepointDates }: PastasPanelProps) {
  const hasMetrics = metrics && Object.values(metrics).some(v => v != null)
  const hasResiduals = (residuals?.length ?? 0) > 0
  const hasAcf = (acf?.length ?? 0) > 0 || (pacf?.length ?? 0) > 0

  if (!hasMetrics && !hasResiduals && !hasAcf) {
    return (
      <div className="flex items-center justify-center h-40 text-text-secondary text-sm">
        Waiting for Pastas results...
      </div>
    )
  }

  const residualShapes: Partial<Shape>[] = (changepointDates ?? []).map(d => ({
    type: 'line' as const,
    xref: 'x' as const,
    yref: 'paper' as const,
    x0: d,
    x1: d,
    y0: 0,
    y1: 1,
    line: { color: 'rgba(251,191,36,0.7)', width: 1.5, dash: 'dash' as const },
  }))

  const residualTraces: Data[] = hasResiduals
    ? [
        {
          x: residuals!.map(r => r.time),
          y: residuals!.map(r => r.value),
          type: 'scatter',
          mode: 'lines',
          name: 'Residuals',
          line: { color: '#a78bfa', width: 1 },
          hovertemplate: '%{x}<br>%{y:.4f}<extra></extra>',
        },
      ]
    : []

  const acfLen = acf?.length ?? 0
  const pacfLen = pacf?.length ?? 0
  const maxLen = Math.max(acfLen, pacfLen)
  const lags = Array.from({ length: maxLen }, (_, i) => i)

  const acfTraces: Data[] = []
  if (acfLen > 0) {
    acfTraces.push({
      x: lags.slice(0, acfLen),
      y: acf!,
      type: 'bar',
      name: 'ACF',
      marker: { color: 'rgba(34,211,238,0.7)' },
    })
  }
  if (pacfLen > 0) {
    acfTraces.push({
      x: lags.slice(0, pacfLen),
      y: pacf!,
      type: 'bar',
      name: 'PACF',
      marker: { color: 'rgba(167,139,250,0.7)' },
    })
  }

  return (
    <div className="space-y-4">
      {hasMetrics && (
        <div className="flex gap-3 flex-wrap">
          <MetricBadge label="EVP" value={metrics?.evp} unit="%" />
          <MetricBadge label="RMSE" value={metrics?.rmse} />
          <MetricBadge label="NSE" value={metrics?.nse} />
        </div>
      )}

      {hasResiduals && (
        <Plot
          data={residualTraces}
          layout={{
            ...DARK_BASE,
            title: { text: 'Pastas residuals', font: { color: '#e2e8f0', size: 13 } },
            xaxis: DARK_AXIS,
            yaxis: DARK_AXIS,
            shapes: residualShapes,
            margin: { t: 40, r: 20, b: 50, l: 60 },
            showlegend: false,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: 220 }}
        />
      )}

      {hasAcf && (
        <Plot
          data={acfTraces}
          layout={{
            ...DARK_BASE,
            title: { text: 'ACF / PACF of residuals', font: { color: '#e2e8f0', size: 13 } },
            xaxis: { ...DARK_AXIS, title: { text: 'Lag' } },
            yaxis: DARK_AXIS,
            barmode: 'group',
            margin: { t: 40, r: 20, b: 50, l: 60 },
            legend: { font: { color: '#94a3b8' } },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: 220 }}
        />
      )}
    </div>
  )
}
