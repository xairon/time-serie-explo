import { useMemo, useState } from 'react'
import Plot from 'react-plotly.js'
import type { StationPercentiles } from '@/lib/observatory-types'

interface Props {
  data: any[]
  valueKey: string
  valueLabel: string
  unit: string
  precipKey?: string
  percentiles?: StationPercentiles | null
  resolution?: 'daily' | 'monthly' | 'yearly'
  defaultPeriod?: number
  onPeriodChange?: (months: number) => void
}

const PERIODS = [
  { label: '1y', months: 12 },
  { label: '5y', months: 60 },
  { label: '10y', months: 120 },
  { label: 'Max', months: Infinity },
] as const

export function TimeseriesChart({ data, valueKey, valueLabel, unit, precipKey = 'precipitation_totale', percentiles, resolution = 'monthly', defaultPeriod = Infinity, onPeriodChange }: Props) {
  const isYearly = resolution === 'yearly'
  const [period, setPeriod] = useState<number>(defaultPeriod)

  const dateAccessor = (d: any): string => d.mois ?? d.date ?? d.date_mesure ?? d.date_obs_elab ?? String(d.annee)

  const filteredData = useMemo(() => {
    if (isYearly || period === Infinity) return data
    const cutoff = new Date()
    cutoff.setMonth(cutoff.getMonth() - period)
    return data.filter((d: any) => {
      const dateStr = dateAccessor(d)
      if (!dateStr) return false
      return new Date(dateStr).getTime() >= cutoff.getTime()
    })
  }, [data, period, isYearly])

  if (!filteredData.length) {
    return <div className="flex items-center justify-center h-64 text-text-secondary text-sm">No data</div>
  }

  const dates = filteredData.map(dateAccessor)
  const values = filteredData.map((d: any) => d[valueKey] ?? null)
  const precipValues = filteredData.map((d: any) => d[precipKey] ?? null)
  const hasPrecip = precipValues.some((v: number | null) => v != null && v > 0)

  const traces: Plotly.Data[] = []

  // Percentile bands (subtle, no legend clutter)
  if (percentiles?.p10 != null && percentiles.p90 != null && !isYearly) {
    traces.push({
      x: [...dates, ...dates.slice().reverse()],
      y: [...Array(dates.length).fill(percentiles.p90), ...Array(dates.length).fill(percentiles.p10).reverse()],
      fill: 'toself', fillcolor: 'rgba(99,102,241,0.06)', line: { width: 0 },
      type: 'scatter', mode: 'lines', name: 'P10–P90', showlegend: true, hoverinfo: 'skip',
    })
    if (percentiles.p25 != null && percentiles.p75 != null) {
      traces.push({
        x: [...dates, ...dates.slice().reverse()],
        y: [...Array(dates.length).fill(percentiles.p75), ...Array(dates.length).fill(percentiles.p25).reverse()],
        fill: 'toself', fillcolor: 'rgba(34,197,94,0.08)', line: { width: 0 },
        type: 'scatter', mode: 'lines', name: 'P25–P75', showlegend: true, hoverinfo: 'skip',
      })
    }
  }

  const useWebGL = dates.length > 2000

  // Main line
  traces.push({
    x: dates, y: values,
    type: useWebGL ? 'scattergl' : 'scatter', mode: 'lines',
    name: valueLabel,
    line: { color: '#06b6d4', width: 1.5 },
    yaxis: 'y',
    connectgaps: false,
  })

  // Precipitation bars (inverted on y2)
  if (hasPrecip) {
    traces.push({
      x: dates, y: precipValues,
      type: 'bar',
      name: 'Precipitation (mm)',
      marker: { color: 'rgba(56,189,248,0.25)' },
      yaxis: 'y2',
    })
  }

  const layout: Partial<Plotly.Layout> = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#9ca3af', size: 11 },
    margin: { t: 10, r: hasPrecip ? 50 : 20, b: 40, l: 55 },
    height: 340,
    xaxis: {
      gridcolor: 'rgba(255,255,255,0.04)',
      rangeslider: useWebGL ? { visible: false } : { visible: true, thickness: 0.06 },
    },
    yaxis: {
      title: { text: unit },
      gridcolor: 'rgba(255,255,255,0.05)',
      domain: hasPrecip ? [0, 0.72] : [0, 1],
    },
    ...(hasPrecip ? {
      yaxis2: {
        title: { text: 'mm' },
        gridcolor: 'rgba(255,255,255,0.03)',
        domain: [0.78, 1],
        autorange: 'reversed' as const,
      },
    } : {}),
    legend: {
      orientation: 'h' as const,
      y: -0.22,
      font: { size: 10, color: '#9ca3af' },
    },
    hovermode: 'x unified' as const,
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-text-primary">{valueLabel}</h3>
        {!isYearly && (
          <div className="flex gap-1">
            {PERIODS.map(({ label, months }) => (
              <button
                key={label}
                onClick={() => { setPeriod(months); onPeriodChange?.(months) }}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  period === months ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </div>
      <Plot
        data={traces}
        layout={layout}
        config={{ displayModeBar: false }}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
