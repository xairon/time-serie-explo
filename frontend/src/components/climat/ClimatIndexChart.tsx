import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ComposedChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceArea, ReferenceLine } from 'recharts'
import { CHART_TOOLTIP_STYLE } from '@/lib/observatory-types'
import type { ClimatPointSeriesEntry } from '@/lib/observatory-types'
import { classifyIndex, SPI_CLASS_COLORS, SPI_CLASS_ORDER, STI_CLASS_COLORS, STI_CLASS_ORDER } from '@/lib/era5-colors'
import { CLIMAT_WINDOWS } from '@/lib/climat-colors'

interface Props {
  series: ClimatPointSeriesEntry[]
}

// WMO 7-class boundaries (±0.84/±1.28/±1.75σ), same thresholds as DroughtIndexChart /
// api/era5_anomaly.py::classify_index — ordered coldest/driest → hottest/wettest.
const ZONE_BOUNDS: { y1?: number; y2?: number }[] = [
  { y2: -1.75 },
  { y1: -1.75, y2: -1.28 },
  { y1: -1.28, y2: -0.84 },
  { y1: -0.84, y2: 0.84 },
  { y1: 0.84, y2: 1.28 },
  { y1: 1.28, y2: 1.75 },
  { y1: 1.75 },
]

const PERIODS = [
  { key: 'period5y', months: 60 },
  { key: 'period10y', months: 120 },
  { key: 'periodMax', months: Infinity },
] as const

/** SPI/STI multi-window chart (Task B2) — switchable index (SPI ⇄ STI) and window
 *  (1/3/6/12 months), 7-class WMO severity bands (reusing the same thresholds/colours
 *  as the Situation map and DroughtIndexChart). Fed by point-series (no separate
 *  endpoint call — the 4 windows are already in the payload). */
export function ClimatIndexChart({ series }: Props) {
  const { t, i18n } = useTranslation()
  const [index, setIndex] = useState<'spi' | 'sti'>('spi')
  const [window, setWindowState] = useState(3)
  const [period, setPeriod] = useState<number>(120)
  const localeTag = i18n.language?.startsWith('en') ? 'en-US' : 'fr-FR'

  const order = index === 'spi' ? SPI_CLASS_ORDER : STI_CLASS_ORDER
  const colors = index === 'spi' ? SPI_CLASS_COLORS : STI_CLASS_COLORS
  const ns = index === 'spi' ? 'observatory.spi' : 'observatory.sti'
  const fieldKey = (index === 'spi' ? `spi_${window}` : `sti_${window}`) as keyof ClimatPointSeriesEntry

  const filtered = useMemo(() => {
    if (period === Infinity) return series
    const cutoff = new Date()
    cutoff.setMonth(cutoff.getMonth() - period)
    const cutoffMs = cutoff.getTime()
    return series.filter((d) => new Date(d.month).getTime() >= cutoffMs)
  }, [series, period])

  const chartData = useMemo(
    () => filtered
      .map((d) => ({ month: d.month, value: d[fieldKey] as number | null }))
      .filter((d): d is { month: string; value: number } => d.value != null)
      .map((d) => ({ ...d, fill: colors[classifyIndex(d.value)] ?? colors.UNKNOWN })),
    [filtered, fieldKey, colors],
  )

  if (!chartData.length) {
    return <div className="flex items-center justify-center h-40 text-text-secondary text-sm">{t('climat.pointPanel.noIndexData')}</div>
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <div className="flex gap-1" role="radiogroup" aria-label={t('climat.pointPanel.indexLabel')}>
          {(['spi', 'sti'] as const).map((k) => (
            <button
              key={k} type="button" role="radio" aria-checked={index === k}
              onClick={() => setIndex(k)}
              className={`text-xs px-2.5 py-1 rounded-md transition-colors ${index === k ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`}
            >
              {t(k === 'spi' ? 'climat.variables.spi' : 'climat.variables.sti')}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1" role="radiogroup" aria-label={t('climat.picker.windowLabel')}>
          <span className="text-[10px] text-text-secondary mr-1">{t('climat.picker.windowLabel')}</span>
          {CLIMAT_WINDOWS.map((w) => (
            <button
              key={w} type="button" role="radio" aria-checked={window === w}
              onClick={() => setWindowState(w)}
              className={`text-xs px-2 py-0.5 rounded-md transition-colors ${window === w ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`}
            >
              {w}
            </button>
          ))}
        </div>
        <div className="flex gap-1">
          {PERIODS.map(({ key, months }) => (
            <button
              key={key} type="button" onClick={() => setPeriod(months)} aria-pressed={period === months}
              className={`px-2 py-1 rounded text-xs font-medium transition-colors ${period === months ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`}
            >
              {t(`climat.pointPanel.${key}`)}
            </button>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          {ZONE_BOUNDS.map((z, i) => (
            <ReferenceArea key={i} yAxisId="left" y1={z.y1} y2={z.y2} fill={colors[order[i]]} fillOpacity={0.1} ifOverflow="visible" />
          ))}
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <ReferenceLine yAxisId="left" y={0} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4" />
          <XAxis dataKey="month" tick={{ fill: '#9ca3af', fontSize: 11 }} tickFormatter={(v: string) => { const d = new Date(v); return `${d.getMonth() + 1}/${String(d.getFullYear()).slice(2)}` }} stroke="transparent" />
          <YAxis yAxisId="left" domain={[-3, 3]} tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="transparent" label={{ value: index.toUpperCase(), angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }} />
          <Tooltip
            contentStyle={CHART_TOOLTIP_STYLE}
            labelFormatter={(v: any) => new Date(v).toLocaleDateString(localeTag, { year: 'numeric', month: 'long' })}
            formatter={(value: any) => [Number(value).toFixed(2), index.toUpperCase()]}
          />
          <Bar yAxisId="left" dataKey="value" name={index.toUpperCase()} isAnimationActive={false}>
            {chartData.map((entry, i) => (<Cell key={i} fill={entry.fill} />))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 justify-end">
        {order.map((cls) => (
          <span key={cls} className="flex items-center gap-1 text-xs text-gray-500">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: colors[cls] }} />
            {t(`${ns}.${cls}`, { defaultValue: cls })}
          </span>
        ))}
      </div>
    </div>
  )
}
