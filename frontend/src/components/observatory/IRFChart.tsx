import { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { CHART_TOOLTIP_STYLE } from '@/lib/observatory-types'

interface Props { blockResponse: number[]; tmaxDays: number | null }

export function IRFChart({ blockResponse, tmaxDays }: Props) {
  const chartData = useMemo(() => {
    const maxLen = Math.min(blockResponse.length, 1500); const slice = blockResponse.slice(0, maxLen)
    const peak = Math.max(...slice.map(Math.abs)); if (peak === 0) return []
    return slice.map((v, i) => ({ day: i, response: v / peak }))
  }, [blockResponse])
  if (!chartData.length) return null
  return (
    <div>
      <h3 className="text-sm font-semibold text-text-primary mb-3">Fonction de réponse impulsionnelle (IRF)</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="day" tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="transparent" label={{ value: 'jours', position: 'insideBottom', offset: -2, fill: '#9ca3af', fontSize: 11 }} />
          <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="transparent" domain={[0, 1]} />
          <Tooltip contentStyle={CHART_TOOLTIP_STYLE} labelFormatter={(v: any) => `Jour ${v}`} formatter={(value: any) => [Number(value).toFixed(3), 'Réponse']} />
          {tmaxDays != null && (<ReferenceLine x={Math.round(tmaxDays)} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: `tmax = ${Math.round(tmaxDays)}j`, fill: '#f59e0b', fontSize: 11, position: 'top' }} />)}
          <Line dataKey="response" stroke="#8b5cf6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
