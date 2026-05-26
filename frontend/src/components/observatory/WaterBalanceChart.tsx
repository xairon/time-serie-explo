import { useMemo, useState } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { CHART_TOOLTIP_STYLE } from '@/lib/observatory-types'
import type { ObsPastasTimeseriesPoint } from '@/lib/observatory-types'

const PERIODS = [{ label: '5 ans', months: 60 }, { label: '10 ans', months: 120 }, { label: 'Max', months: Infinity }] as const
const COMPONENTS = [
  { key: 'wb_effective_precip', label: 'Précip. efficace', color: '#22c55e' },
  { key: 'wb_recharge', label: 'Recharge', color: '#3b82f6' },
  { key: 'wb_actual_evaporation', label: 'Évaporation', color: '#ef4444' },
  { key: 'wb_surface_runoff', label: 'Ruissellement', color: '#f97316' },
] as const
interface Props { data: ObsPastasTimeseriesPoint[] }

export function WaterBalanceChart({ data }: Props) {
  const [period, setPeriod] = useState<number>(120)
  const chartData = useMemo(() => { let filtered = data; if (period !== Infinity) { const cutoff = new Date(); cutoff.setMonth(cutoff.getMonth() - period); const cutoffMs = cutoff.getTime(); filtered = data.filter(d => new Date(d.date).getTime() >= cutoffMs) }; return filtered.map(d => ({ ...d, wb_actual_evaporation: d.wb_actual_evaporation != null ? -Math.abs(d.wb_actual_evaporation) : null, wb_surface_runoff: d.wb_surface_runoff != null ? -Math.abs(d.wb_surface_runoff) : null })) }, [data, period])
  const hasWB = chartData.some(d => d.wb_recharge != null || d.wb_actual_evaporation != null || d.wb_surface_runoff != null || d.wb_effective_precip != null)
  if (!chartData.length || !hasWB) return null
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-text-primary">Bilan hydrique</h3>
        <div className="flex gap-1">{PERIODS.map(({ label, months }) => (<button key={label} onClick={() => setPeriod(months)} aria-pressed={period === months} className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${period === months ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`}>{label}</button>))}</div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 11 }} tickFormatter={(v: string) => { const d = new Date(v); return `${d.getMonth() + 1}/${String(d.getFullYear()).slice(2)}` }} stroke="transparent" />
          <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="transparent" label={{ value: 'mm', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }} />
          <Tooltip contentStyle={CHART_TOOLTIP_STYLE} labelFormatter={(v: any) => new Date(v).toLocaleDateString('fr-FR', { year: 'numeric', month: 'long' })} formatter={(value: any, name: any) => { const comp = COMPONENTS.find((c: any) => c.key === name); return [value != null ? `${Number(value).toFixed(1)} mm` : '--', comp?.label ?? name] }} />
          {COMPONENTS.map(c => (<Area key={c.key} type="monotone" dataKey={c.key} name={c.key} stroke={c.color} fill={c.color} fillOpacity={0.15} strokeWidth={1.5} isAnimationActive={false} connectNulls />))}
        </AreaChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 justify-end">{COMPONENTS.map(c => (<span key={c.key} className="flex items-center gap-1 text-xs text-gray-500"><span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: c.color }} />{c.label}</span>))}</div>
    </div>
  )
}
