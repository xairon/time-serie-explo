import { useTranslation } from 'react-i18next'
import Plot from 'react-plotly.js'
import type { ClimatPointSeriesEntry } from '@/lib/observatory-types'

interface Props {
  series: ClimatPointSeriesEntry[]
}

const DEFAULT_YEARS = 10

/** Monthly precipitation vs. the calendar-month normal (1991-2020 climatology),
 *  bars + line. Same Plotly conventions as TimeseriesChart.tsx (dark styling,
 *  range slider for zoom/brush) — the full 1950→présent history is always loaded,
 *  the range slider just starts scoped to the last 10 years (plan B2 default). */
export function PrecipNormalChart({ series }: Props) {
  const { t } = useTranslation()

  if (!series.length) {
    return <div className="flex items-center justify-center h-40 text-text-secondary text-sm">{t('climat.pointPanel.noData')}</div>
  }

  const dates = series.map((d) => d.month)
  const precip = series.map((d) => d.precipitation_totale)
  const normal = series.map((d) => d.precipitation_normale)

  const lastDate = new Date(dates[dates.length - 1])
  const defaultStart = new Date(lastDate)
  defaultStart.setFullYear(defaultStart.getFullYear() - DEFAULT_YEARS)

  const traces: Plotly.Data[] = [
    {
      x: dates, y: precip,
      type: 'bar',
      name: t('climat.pointPanel.precipitation'),
      marker: { color: 'rgba(56,189,248,0.55)' },
    },
    {
      x: dates, y: normal,
      type: 'scatter', mode: 'lines',
      name: t('climat.pointPanel.normal'),
      line: { color: '#f59e0b', width: 1.5, dash: 'dot' },
    },
  ]

  const layout: Partial<Plotly.Layout> = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#9ca3af', size: 11 },
    margin: { t: 10, r: 20, b: 40, l: 55 },
    height: 260,
    xaxis: {
      type: 'date' as const,
      gridcolor: 'rgba(255,255,255,0.04)',
      rangeslider: { visible: true, thickness: 0.08 },
      range: [defaultStart.toISOString().slice(0, 10), dates[dates.length - 1]],
    },
    yaxis: {
      title: { text: 'mm' },
      gridcolor: 'rgba(255,255,255,0.05)',
    },
    legend: {
      orientation: 'h' as const,
      y: -0.3,
      font: { size: 10, color: '#9ca3af' },
    },
    hovermode: 'x unified' as const,
  }

  return (
    <div>
      <h3 className="text-sm font-semibold text-text-primary mb-2">{t('climat.pointPanel.precipVsNormal')}</h3>
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
