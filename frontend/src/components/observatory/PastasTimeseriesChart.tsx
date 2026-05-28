import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { CHART_TOOLTIP_STYLE } from '@/lib/observatory-types'
import type { ObsPastasTimeseriesPoint } from '@/lib/observatory-types'

interface Props { data: ObsPastasTimeseriesPoint[] }

export function PastasTimeseriesChart({ data }: Props) {
  const { t, i18n } = useTranslation()
  const localeTag = i18n.language?.startsWith('en') ? 'en-US' : 'fr-FR'
  const PERIODS = [
    { label: t('observatory.timeseries.period5y'), months: 60 },
    { label: t('observatory.timeseries.period10y'), months: 120 },
    { label: t('observatory.timeseries.periodMax'), months: Infinity },
  ] as const
  const [period, setPeriod] = useState<number>(120)
  const chartData = useMemo(() => { let filtered = data; if (period !== Infinity) { const cutoff = new Date(); cutoff.setMonth(cutoff.getMonth() - period); const cutoffMs = cutoff.getTime(); filtered = data.filter(d => new Date(d.date).getTime() >= cutoffMs) }; return filtered }, [data, period])
  if (!chartData.length) return null
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-text-primary">{t('observatory.pastas.observedVsSimulated')}</h3>
        <div className="flex gap-1">{PERIODS.map(({ label, months }) => (<button key={label} onClick={() => setPeriod(months)} aria-pressed={period === months} className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${period === months ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`}>{label}</button>))}</div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 11 }} tickFormatter={(v: string) => { const d = new Date(v); return `${d.getMonth() + 1}/${String(d.getFullYear()).slice(2)}` }} stroke="transparent" />
          <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="transparent" domain={['auto', 'auto']} label={{ value: 'm NGF', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }} />
          <Tooltip contentStyle={CHART_TOOLTIP_STYLE} labelFormatter={(v: any) => new Date(v).toLocaleDateString(localeTag, { year: 'numeric', month: 'long' })} formatter={(value: any, name: any) => [value != null ? Number(value).toFixed(2) : '--', name === 'simulated' ? t('observatory.pastas.simulated') : t('observatory.pastas.observed')]} />
          <Line dataKey="observed" name="observed" stroke="transparent" dot={{ r: 2, fill: '#9ca3af', strokeWidth: 0 }} activeDot={{ r: 3, fill: '#9ca3af' }} connectNulls={false} isAnimationActive={false} />
          <Line dataKey="simulated" name="simulated" stroke="#06b6d4" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
