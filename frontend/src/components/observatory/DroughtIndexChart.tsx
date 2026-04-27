import { useMemo, useState } from 'react'
import { ComposedChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceArea, ReferenceLine } from 'recharts'
import { CHART_TOOLTIP_STYLE } from '@/lib/observatory-types'

interface DataPoint { mois: string; value: number | null; spli?: number | null; ssfi?: number | null; sgi?: number | null; classification: string }
interface Props { data: DataPoint[]; indexKey: 'spi' | 'spli' | 'ssfi' | 'sgi'; label: string }

const ZONE_BANDS: { y1?: number; y2?: number; color: string; label: string }[] = [
  { y2: -1.75, color: 'rgba(153,27,27,0.15)', label: 'Extremely low' }, { y1: -1.75, y2: -1.28, color: 'rgba(239,68,68,0.12)', label: 'Very low' },
  { y1: -1.28, y2: -0.84, color: 'rgba(249,115,22,0.12)', label: 'Low' }, { y1: -0.84, y2: 0.84, color: 'rgba(34,197,94,0.08)', label: 'Normal' },
  { y1: 0.84, y2: 1.28, color: 'rgba(59,130,246,0.12)', label: 'High' }, { y1: 1.28, y2: 1.75, color: 'rgba(99,102,241,0.12)', label: 'Very high' },
  { y1: 1.75, color: 'rgba(49,46,129,0.15)', label: 'Extremely high' },
]
const CLASS_COLORS: Record<string, string> = { EXTREMEMENT_BAS: '#991b1b', TRES_BAS: '#ef4444', BAS: '#f97316', NORMAL: '#10b981', HAUT: '#3b82f6', TRES_HAUT: '#1d4ed8', EXTREMEMENT_HAUT: '#312e81', UNKNOWN: '#6b7280' }
const PERIODS = [{ label: '5y', months: 60 }, { label: '10y', months: 120 }, { label: 'Max', months: Infinity }] as const

export function DroughtIndexChart({ data, indexKey, label }: Props) {
  const [period, setPeriod] = useState<number>(120)
  const filtered = useMemo(() => { if (period === Infinity) return data; const cutoff = new Date(); cutoff.setMonth(cutoff.getMonth() - period); const cutoffMs = cutoff.getTime(); return data.filter(d => new Date(d.mois).getTime() >= cutoffMs) }, [data, period])
  const chartData = useMemo(() => filtered.map(d => ({ ...d, fill: CLASS_COLORS[d.classification] ?? CLASS_COLORS.UNKNOWN })), [filtered])
  if (!chartData.length) return <div className="flex items-center justify-center h-48 text-text-secondary text-sm">Insufficient data (min. 5 years)</div>
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-text-primary">{label}</h3>
        <div className="flex gap-1">{PERIODS.map(({ label: l, months }) => (<button key={l} onClick={() => setPeriod(months)} aria-pressed={period === months} className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${period === months ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`}>{l}</button>))}</div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          {ZONE_BANDS.map((z, i) => (<ReferenceArea key={i} yAxisId="left" y1={z.y1} y2={z.y2} fill={z.color} ifOverflow="visible" />))}
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <ReferenceLine yAxisId="left" y={0} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4" />
          <XAxis dataKey="mois" tick={{ fill: '#9ca3af', fontSize: 11 }} tickFormatter={(v: string) => { const d = new Date(v); return `${d.getMonth() + 1}/${String(d.getFullYear()).slice(2)}` }} stroke="transparent" />
          <YAxis yAxisId="left" domain={[-3, 3]} tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="transparent" label={{ value: indexKey.toUpperCase(), angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }} />
          <Tooltip contentStyle={CHART_TOOLTIP_STYLE} labelFormatter={(v: any) => new Date(v).toLocaleDateString('en-GB', { year: 'numeric', month: 'long' })} formatter={(value: any, name: any) => { if (name === indexKey.toUpperCase()) return [value?.toFixed(2) ?? '--', name]; return [value, name] }} />
          <Bar yAxisId="left" dataKey={indexKey} name={indexKey.toUpperCase()} isAnimationActive={false}>{chartData.map((entry, i) => (<Cell key={i} fill={entry.fill} />))}</Bar>
        </ComposedChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 justify-end">
        {[{ label: 'Extr. low (< -1.75)', color: '#991b1b' }, { label: 'Very low', color: '#ef4444' }, { label: 'Low', color: '#f97316' }, { label: 'Normal', color: '#10b981' }, { label: 'High', color: '#3b82f6' }, { label: 'Very high', color: '#1d4ed8' }, { label: 'Extr. high (> 1.75)', color: '#312e81' }].map(z => (<span key={z.label} className="flex items-center gap-1 text-xs text-gray-500"><span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: z.color }} />{z.label}</span>))}
      </div>
    </div>
  )
}
