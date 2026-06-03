import { useEffect, useMemo, useState } from 'react'
import Plot from 'react-plotly.js'
import { useTranslation } from 'react-i18next'
import { CalendarRange } from 'lucide-react'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Data, Layout } from 'plotly.js-dist-min'
import { usePiezoMonthly, useHydroMonthly, usePiezoDaily, useHydroDaily } from '@/hooks/useObservatory'

type Props = { code: string; type: 'piezo' | 'hydro'; unit: string }

const PALETTE = ['#22d3ee', '#a78bfa', '#f472b6', '#fbbf24', '#34d399', '#f87171', '#60a5fa', '#fb923c']

function fmt(v: number | null | undefined, unit: string): string {
  return v == null || Number.isNaN(v) ? '—' : `${v.toFixed(2)} ${unit}`
}

export function StationPeriodsPanel({ code, type, unit }: Props) {
  const { t, i18n } = useTranslation()
  const isPiezo = type === 'piezo'
  const meanKey = isPiezo ? 'niveau_moyen' : 'resultat_moyen'
  const minKey = isPiezo ? 'niveau_min' : 'resultat_min'
  const maxKey = isPiezo ? 'niveau_max' : 'resultat_max'
  const dailyKey = isPiezo ? 'niveau_nappe_eau' : 'resultat_obs_elab'

  // Monthly series (shared cache with the page; no extra fetch)
  const piezoM = usePiezoMonthly(isPiezo ? code : '')
  const hydroM = useHydroMonthly(!isPiezo ? code : '')
  const monthly = useMemo(() => ((isPiezo ? piezoM.data : hydroM.data) ?? []) as Array<Record<string, any>>, [isPiezo, piezoM.data, hydroM.data])

  const [mode, setMode] = useState<'read' | 'compare'>('read')
  const [readMode, setReadMode] = useState<'date' | 'range'>('date')
  const [singleDate, setSingleDate] = useState('')
  const [rangeStart, setRangeStart] = useState('')
  const [rangeEnd, setRangeEnd] = useState('')

  // Exact-date lookup uses the daily endpoint (one day)
  const piezoD = usePiezoDaily(isPiezo && readMode === 'date' && singleDate ? code : '', singleDate, singleDate, 5)
  const hydroD = useHydroDaily(!isPiezo && readMode === 'date' && singleDate ? code : '', singleDate, singleDate, 5)
  const dailyData = ((isPiezo ? piezoD.data : hydroD.data) ?? []) as Array<Record<string, any>>

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

  // --- Read: single date ---
  const singleReadout = useMemo(() => {
    if (readMode !== 'date' || !singleDate) return null
    const exact = dailyData.find(d => String(d.date_mesure ?? '').slice(0, 10) === singleDate)
    if (exact && exact[dailyKey] != null) return { value: exact[dailyKey] as number, exact: true }
    const month = singleDate.slice(0, 7)
    const mrow = monthly.find(r => String(r.mois).slice(0, 7) === month)
    if (mrow && mrow[meanKey] != null) return { value: mrow[meanKey] as number, exact: false }
    return { value: null, exact: false }
  }, [readMode, singleDate, dailyData, dailyKey, monthly, meanKey])

  // --- Read: range stats (from monthly) ---
  const rangeReadout = useMemo(() => {
    if (readMode !== 'range' || !rangeStart || !rangeEnd) return null
    const rows = monthly.filter(r => {
      const m = String(r.mois).slice(0, 10)
      return m >= rangeStart && m <= rangeEnd
    })
    const means = rows.map(r => r[meanKey]).filter((v): v is number => v != null)
    const mins = rows.map(r => r[minKey] ?? r[meanKey]).filter((v): v is number => v != null)
    const maxs = rows.map(r => r[maxKey] ?? r[meanKey]).filter((v): v is number => v != null)
    if (means.length === 0) return { n: 0, min: null, mean: null, max: null, amp: null }
    const min = Math.min(...mins), max = Math.max(...maxs)
    const mean = means.reduce((a, b) => a + b, 0) / means.length
    return { n: rows.length, min, mean, max, amp: max - min }
  }, [readMode, rangeStart, rangeEnd, monthly, meanKey, minKey, maxKey])

  // --- Compare: overlay by calendar month ---
  const compareTraces: Data[] = useMemo(() => {
    if (mode !== 'compare') return []
    return selectedYears.map((yr, i) => {
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
    })
  }, [mode, selectedYears, monthly, meanKey, monthLabels])

  const compareLayout: Partial<Layout> = useMemo(() => ({
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, type: 'category' as const },
    yaxis: { ...darkLayout.yaxis, title: { text: unit } },
    hovermode: 'x unified',
    legend: { orientation: 'h', y: 1.04, yanchor: 'bottom', x: 0.5, xanchor: 'center', font: { color: '#9ca3af', size: 11 } },
    margin: { l: 50, r: 16, t: 24, b: 32 },
  }), [unit])

  if (monthly.length === 0) return null

  const tabBtn = (active: boolean) =>
    `px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${active ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`
  const dateInput = 'bg-bg-card border border-white/10 rounded-lg px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-cyan/50'

  return (
    <section className="bg-bg-card border border-white/5 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <CalendarRange className="w-4 h-4" />{t('mainPages.station.periods.title')}
        </h2>
        <div className="flex gap-1">
          <button type="button" onClick={() => setMode('read')} className={tabBtn(mode === 'read')}>{t('mainPages.station.periods.read')}</button>
          <button type="button" onClick={() => setMode('compare')} className={tabBtn(mode === 'compare')}>{t('mainPages.station.periods.compare')}</button>
        </div>
      </div>

      {mode === 'read' ? (
        <div className="space-y-3">
          <div className="flex gap-1">
            <button type="button" onClick={() => setReadMode('date')} className={tabBtn(readMode === 'date')}>{t('mainPages.station.periods.atDate')}</button>
            <button type="button" onClick={() => setReadMode('range')} className={tabBtn(readMode === 'range')}>{t('mainPages.station.periods.betweenDates')}</button>
          </div>

          {readMode === 'date' ? (
            <div className="flex items-center gap-3 flex-wrap">
              <input type="date" value={singleDate} onChange={e => setSingleDate(e.target.value)} className={dateInput} aria-label={t('mainPages.station.periods.atDate')} />
              {singleReadout && (
                <p className="text-sm text-text-primary">
                  <span className="font-bold text-accent-cyan">{fmt(singleReadout.value, unit)}</span>
                  <span className="text-[11px] text-text-secondary ml-2">
                    {singleReadout.value == null
                      ? t('mainPages.station.periods.noValue')
                      : singleReadout.exact ? t('mainPages.station.periods.exactDay') : t('mainPages.station.periods.monthlyMean')}
                  </span>
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center gap-2 flex-wrap">
                <input type="date" value={rangeStart} onChange={e => setRangeStart(e.target.value)} className={dateInput} aria-label={t('mainPages.station.startDate')} />
                <span className="text-xs text-text-secondary">→</span>
                <input type="date" value={rangeEnd} onChange={e => setRangeEnd(e.target.value)} className={dateInput} aria-label={t('mainPages.station.endDate')} />
              </div>
              {rangeReadout && (rangeReadout.n === 0 ? (
                <p className="text-xs text-text-secondary">{t('mainPages.station.periods.noValue')}</p>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {([['min', rangeReadout.min], ['mean', rangeReadout.mean], ['max', rangeReadout.max], ['amplitude', rangeReadout.amp]] as const).map(([k, v]) => (
                    <div key={k} className="bg-bg-hover rounded-lg p-3 text-center">
                      <p className="text-[10px] text-text-secondary uppercase mb-1">{t(`mainPages.station.periods.${k}`)}</p>
                      <p className="text-sm font-bold text-text-primary">{fmt(v, unit)}</p>
                    </div>
                  ))}
                </div>
              ))}
              {rangeReadout && rangeReadout.n > 0 && (
                <p className="text-[10px] text-text-secondary">{t('mainPages.station.periods.rangeNote', { count: rangeReadout.n })}</p>
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-3">
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
            <button type="button" onClick={() => setSelectedYears(years.slice(-3))} className="px-2 py-0.5 rounded text-[10px] font-medium text-accent-cyan hover:bg-accent-cyan/10">
              {t('mainPages.station.periods.last3')}
            </button>
          </div>
          {selectedYears.length === 0 ? (
            <p className="text-xs text-text-secondary">{t('mainPages.station.periods.pickYears')}</p>
          ) : (
            <Plot data={compareTraces} layout={compareLayout} config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '320px' }} />
          )}
        </div>
      )}
    </section>
  )
}
