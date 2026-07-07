import { useTranslation } from 'react-i18next'
import Plot from 'react-plotly.js'
import type { ClimatCompareYears } from '@/lib/observatory-types'
import { buildCumulSeries, buildNormalSeries } from '@/lib/climat-compare'

interface Props {
  data: ClimatCompareYears | undefined
  years: number[]
}

// Distinct, colour-blind-tolerant palette, cycled if > 6 years (shouldn't happen —
// the multi-select caps at MAX_COMPARE_YEARS).
const YEAR_COLORS = ['#38bdf8', '#f472b6', '#a3e635', '#fb923c', '#c084fc', '#facc15']

/** Superposed monthly cumulative-precipitation curves, one per selected year, plus the
 *  1991-2020 normale as a distinct dashed reference (Task B3). Same Plotly conventions
 *  as PrecipNormalChart.tsx (dark styling, unified hover). x is the calendar month
 *  (jan → déc) so years line up regardless of how many months of data each has. */
export function CompareCumulChart({ data, years }: Props) {
  const { t, i18n } = useTranslation()
  const localeTag = i18n.language?.startsWith('en') ? 'en-US' : 'fr-FR'
  const monthLabels = Array.from({ length: 12 }, (_, i) =>
    new Intl.DateTimeFormat(localeTag, { month: 'short' }).format(new Date(2000, i, 1)),
  )

  const cumulSeries = buildCumulSeries(data, years)
  const normal = buildNormalSeries(data, years)
  const hasData = cumulSeries.some((s) => s.y.some((v) => v != null))

  if (!data || !hasData) {
    return <div className="flex items-center justify-center h-40 text-text-secondary text-sm">{t('climat.pointPanel.noData')}</div>
  }

  const normalLabel = t('climat.compare.normal')
  const traces: Plotly.Data[] = [
    ...cumulSeries.map((s, i) => ({
      x: monthLabels,
      y: s.y,
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name: String(s.year),
      line: { color: YEAR_COLORS[i % YEAR_COLORS.length], width: 2 },
      marker: { size: 4 },
      hovertemplate: `${s.year} · %{x} : %{y} mm<extra></extra>`,
    })),
    {
      x: monthLabels,
      y: normal.y,
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: normalLabel,
      line: { color: '#9ca3af', width: 1.5, dash: 'dot' },
      hovertemplate: `${normalLabel} · %{x} : %{y} mm<extra></extra>`,
    },
  ]

  const layout: Partial<Plotly.Layout> = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#9ca3af', size: 11 },
    margin: { t: 10, r: 20, b: 30, l: 55 },
    height: 260,
    xaxis: {
      type: 'category' as const,
      gridcolor: 'rgba(255,255,255,0.04)',
    },
    yaxis: {
      title: { text: 'mm' },
      gridcolor: 'rgba(255,255,255,0.05)',
    },
    legend: {
      orientation: 'h' as const,
      y: -0.25,
      font: { size: 10, color: '#9ca3af' },
    },
    hovermode: 'x unified' as const,
  }

  return (
    <div>
      <h4 className="text-xs font-semibold text-text-primary mb-2">{t('climat.compare.cumulTitle')}</h4>
      <Plot data={traces} layout={layout} config={{ displayModeBar: false }} useResizeHandler className="w-full" />
    </div>
  )
}
