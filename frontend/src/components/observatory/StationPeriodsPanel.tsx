import { useEffect, useMemo, useState } from 'react'
import Plot from 'react-plotly.js'
import { useTranslation } from 'react-i18next'
import { CalendarRange } from 'lucide-react'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Data, Layout } from 'plotly.js-dist-min'
import { usePiezoMonthly, useHydroMonthly } from '@/hooks/useObservatory'

type Props = { code: string; type: 'piezo' | 'hydro'; unit: string }

const PALETTE = ['#22d3ee', '#a78bfa', '#f472b6', '#fbbf24', '#34d399', '#f87171', '#60a5fa', '#fb923c']

/** Compare a station to itself: overlay its yearly trajectories by calendar month. */
export function StationPeriodsPanel({ code, type, unit }: Props) {
  const { t, i18n } = useTranslation()
  const isPiezo = type === 'piezo'
  const meanKey = isPiezo ? 'niveau_moyen' : 'resultat_moyen'

  // Monthly series (shared cache with the page; no extra fetch)
  const piezoM = usePiezoMonthly(isPiezo ? code : '')
  const hydroM = useHydroMonthly(!isPiezo ? code : '')
  const monthly = useMemo(
    () => ((isPiezo ? piezoM.data : hydroM.data) ?? []) as Array<Record<string, any>>,
    [isPiezo, piezoM.data, hydroM.data],
  )

  const monthLabels = useMemo(
    () => Array.from({ length: 12 }, (_, m) => new Date(2001, m, 1).toLocaleDateString(i18n.language, { month: 'short' })),
    [i18n.language],
  )

  const years = useMemo(() => {
    const s = new Set<number>()
    for (const r of monthly) { const y = parseInt(String(r.mois).slice(0, 4), 10); if (!Number.isNaN(y)) s.add(y) }
    return [...s].sort((a, b) => a - b)
  }, [monthly])

  const [selectedYears, setSelectedYears] = useState<number[]>([])
  useEffect(() => {
    if (selectedYears.length === 0 && years.length > 0) setSelectedYears(years.slice(-3))
  }, [years]) // eslint-disable-line react-hooks/exhaustive-deps

  const traces: Data[] = useMemo(() => selectedYears.map((yr, i) => {
    const byMonth: (number | null)[] = Array(12).fill(null)
    for (const r of monthly) {
      const ms = String(r.mois)
      if (parseInt(ms.slice(0, 4), 10) !== yr) continue
      const mIdx = parseInt(ms.slice(5, 7), 10) - 1
      if (mIdx >= 0 && mIdx < 12) byMonth[mIdx] = r[meanKey] ?? null
    }
    return {
      x: monthLabels,
      y: byMonth,
      type: 'scatter',
      mode: 'lines+markers',
      name: String(yr),
      line: { color: PALETTE[i % PALETTE.length], width: 2 },
      marker: { size: 5 },
      connectgaps: false,
    } as Data
  }), [selectedYears, monthly, meanKey, monthLabels])

  const layout: Partial<Layout> = useMemo(() => ({
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, type: 'category' as const },
    yaxis: { ...darkLayout.yaxis, title: { text: unit } },
    hovermode: 'x unified',
    legend: { orientation: 'h', y: 1.04, yanchor: 'bottom', x: 0.5, xanchor: 'center', font: { color: '#9ca3af', size: 11 } },
    margin: { l: 50, r: 16, t: 24, b: 32 },
  }), [unit])

  if (monthly.length === 0) return null

  return (
    <section className="bg-bg-card border border-white/5 rounded-xl p-5 space-y-3">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <CalendarRange className="w-4 h-4" />{t('mainPages.station.periods.title')}
        </h2>
        <button type="button" onClick={() => setSelectedYears(years.slice(-3))} className="px-2 py-0.5 rounded text-[10px] font-medium text-accent-cyan hover:bg-accent-cyan/10">
          {t('mainPages.station.periods.last3')}
        </button>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[11px] text-text-secondary">{t('mainPages.station.periods.years')}</span>
        <div className="flex gap-1 flex-wrap">
          {years.map(y => {
            const on = selectedYears.includes(y)
            return (
              <button
                key={y}
                type="button"
                aria-pressed={on}
                onClick={() => setSelectedYears(prev => on ? prev.filter(v => v !== y) : [...prev, y].sort((a, b) => a - b))}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${on ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:bg-bg-hover'}`}
              >
                {y}
              </button>
            )
          })}
        </div>
      </div>

      {selectedYears.length === 0 ? (
        <p className="text-xs text-text-secondary">{t('mainPages.station.periods.pickYears')}</p>
      ) : (
        <Plot data={traces} layout={layout} config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '320px' }} />
      )}
    </section>
  )
}
