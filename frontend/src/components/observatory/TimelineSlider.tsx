import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { Play, Pause, X, ChevronFirst, ChevronLast } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { observatoryApi } from '@/lib/observatory-api'
import type { ClassificationTimeline } from '@/lib/observatory-types'

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function formatPeriod(period: string): string {
  const [year, month] = period.split('-')
  return `${MONTH_NAMES[parseInt(month, 10) - 1]} ${year}`
}

type Season = 'winter' | 'spring' | 'summer' | 'autumn'
const SEASONS: { id: Season; label: string; months: number[] }[] = [
  { id: 'winter', label: 'Winter', months: [12, 1, 2] },
  { id: 'spring', label: 'Spring', months: [3, 4, 5] },
  { id: 'summer', label: 'Summer', months: [6, 7, 8] },
  { id: 'autumn', label: 'Autumn', months: [9, 10, 11] },
]

function getSeason(period: string): string {
  const month = parseInt(period.split('-')[1], 10)
  if (month <= 2 || month === 12) return 'Winter'
  if (month <= 5) return 'Spring'
  if (month <= 8) return 'Summer'
  return 'Autumn'
}

const SPEEDS = [
  { label: 'x0.5', interval: 3000, step: 1 },
  { label: 'x1', interval: 1500, step: 1 },
  { label: 'x2', interval: 750, step: 1 },
  { label: 'x5', interval: 300, step: 1 },
  { label: 'x10', interval: 150, step: 1 },
] as const

interface Props { onPeriodChange: (periodIndex: number | null, timeline: ClassificationTimeline | null) => void }

export function TimelineSlider({ onPeriodChange }: Props) {
  const { data: timeline, isLoading } = useQuery({ queryKey: ['obs-classification-timeline'], queryFn: () => observatoryApi.common.classificationTimeline(), staleTime: 3_600_000 * 24 })
  const [active, setActive] = useState(false)
  const [sliderPos, setSliderPos] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speedIdx, setSpeedIdx] = useState(1)
  const playRef = useRef(false); const sliderPosRef = useRef(0); const speedRef = useRef<{ interval: number; step: number }>(SPEEDS[1])
  const [startYear, setStartYear] = useState<number | null>(null); const [endYear, setEndYear] = useState<number | null>(null)
  const [activeSeasons, setActiveSeasons] = useState<Set<Season>>(new Set(['winter', 'spring', 'summer', 'autumn']))

  const allYears = useMemo(() => { if (!timeline) return []; const s = new Set(timeline.periods.map(p => parseInt(p.split('-')[0], 10))); return [...s].sort((a, b) => a - b) }, [timeline])

  const filteredIndices = useMemo(() => {
    if (!timeline) return []
    const allowedMonths = new Set<number>(); activeSeasons.forEach(s => { const season = SEASONS.find(ss => ss.id === s); season?.months.forEach(m => allowedMonths.add(m)) })
    return timeline.periods.map((p, i) => ({ period: p, origIdx: i })).filter(({ period }) => { const [yearStr, monthStr] = period.split('-'); const year = parseInt(yearStr, 10); const month = parseInt(monthStr, 10); if (startYear != null && year < startYear) return false; if (endYear != null && year > endYear) return false; if (!allowedMonths.has(month)) return false; return true }).map(({ origIdx }) => origIdx)
  }, [timeline, startYear, endYear, activeSeasons])

  const maxSliderPos = Math.max(0, filteredIndices.length - 1)

  useEffect(() => { playRef.current = playing; sliderPosRef.current = sliderPos; speedRef.current = SPEEDS[speedIdx] }, [playing, sliderPos, speedIdx])
  useEffect(() => { if (sliderPos > maxSliderPos) { setSliderPos(maxSliderPos); if (timeline && filteredIndices.length > 0) onPeriodChange(filteredIndices[maxSliderPos], timeline) } }, [maxSliderPos]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!playing || !timeline || filteredIndices.length === 0) return
    const tick = () => { if (!playRef.current) return; const s = speedRef.current; const next = sliderPosRef.current + s.step; if (next > maxSliderPos) { setSliderPos(maxSliderPos); onPeriodChange(filteredIndices[maxSliderPos], timeline); setPlaying(false); return }; setSliderPos(next); onPeriodChange(filteredIndices[next], timeline) }
    const interval = setInterval(tick, SPEEDS[speedIdx].interval); return () => clearInterval(interval)
  }, [playing, timeline, maxSliderPos, onPeriodChange, speedIdx, filteredIndices])

  const handleActivate = useCallback(() => { if (!timeline || filteredIndices.length === 0) return; setActive(true); const last = filteredIndices.length - 1; setSliderPos(last); onPeriodChange(filteredIndices[last], timeline) }, [timeline, filteredIndices, onPeriodChange])
  const handleDeactivate = useCallback(() => { setActive(false); setPlaying(false); setSliderPos(0); onPeriodChange(null, null) }, [onPeriodChange])
  const handleSliderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => { const val = parseInt(e.target.value, 10); setSliderPos(val); if (timeline && filteredIndices[val] != null) onPeriodChange(filteredIndices[val], timeline) }, [timeline, filteredIndices, onPeriodChange])
  const togglePlay = useCallback(() => { if (!timeline || filteredIndices.length === 0) return; if (!playing && sliderPos >= maxSliderPos) { setSliderPos(0); onPeriodChange(filteredIndices[0], timeline) }; setPlaying(p => !p) }, [playing, sliderPos, maxSliderPos, timeline, filteredIndices, onPeriodChange])
  const jumpTo = useCallback((target: number) => { if (!timeline || filteredIndices.length === 0) return; const clamped = Math.max(0, Math.min(target, maxSliderPos)); setSliderPos(clamped); onPeriodChange(filteredIndices[clamped], timeline) }, [timeline, filteredIndices, maxSliderPos, onPeriodChange])
  const toggleSeason = useCallback((s: Season) => { setActiveSeasons(prev => { const next = new Set(prev); if (next.has(s)) { if (next.size > 1) next.delete(s) } else next.add(s); return next }) }, [])

  if (isLoading || !timeline) return null

  if (!active) return (<button onClick={handleActivate} className="absolute bottom-16 left-1/2 -translate-x-1/2 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-4 py-2 text-sm text-text-secondary hover:text-text-primary hover:border-accent-cyan/30 transition-colors"><Play className="w-3.5 h-3.5 inline mr-2 -mt-0.5" />Historical Timeline</button>)

  const currentOrigIdx = filteredIndices[sliderPos]; const currentPeriod = currentOrigIdx != null ? timeline.periods[currentOrigIdx] ?? '' : ''
  const filteredPeriods = filteredIndices.map(i => timeline.periods[i]); const years = [...new Set(filteredPeriods.map(p => p.split('-')[0]))].sort()
  const effectiveStart = startYear ?? allYears[0] ?? 2000; const effectiveEnd = endYear ?? allYears[allYears.length - 1] ?? 2026

  return (
    <div className="absolute bottom-16 left-1/2 -translate-x-1/2 z-10 w-[calc(100%-2rem)] max-w-4xl bg-bg-card/95 backdrop-blur-md border border-white/10 rounded-lg px-4 py-3 shadow-xl">
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        <div className="flex items-center gap-1.5 text-[11px]">
          <span className="text-text-secondary">From</span>
          <select value={effectiveStart} onChange={e => setStartYear(parseInt(e.target.value, 10))} className="bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-text-primary text-[11px] focus:outline-none focus:border-accent-cyan/50">{allYears.map(y => <option key={y} value={y}>{y}</option>)}</select>
          <span className="text-text-secondary">to</span>
          <select value={effectiveEnd} onChange={e => setEndYear(parseInt(e.target.value, 10))} className="bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-text-primary text-[11px] focus:outline-none focus:border-accent-cyan/50">{allYears.map(y => <option key={y} value={y}>{y}</option>)}</select>
        </div>
        <div className="flex items-center gap-0.5 bg-white/5 rounded-full border border-white/10 px-0.5 py-0.5">
          {SEASONS.map(s => (<button key={s.id} onClick={() => toggleSeason(s.id)} className={`px-2 py-0.5 rounded-full text-[10px] font-medium transition-colors ${activeSeasons.has(s.id) ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary/50 hover:text-text-secondary'}`}>{s.label}</button>))}
        </div>
        <span className="text-[10px] text-text-secondary/60 ml-auto">{filteredIndices.length} months</span>
      </div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <button onClick={() => jumpTo(0)} className="w-7 h-7 flex items-center justify-center rounded-full text-text-secondary hover:text-text-primary hover:bg-white/10 transition-colors" title="Start"><ChevronFirst className="w-4 h-4" /></button>
          <button onClick={togglePlay} className="w-8 h-8 flex items-center justify-center rounded-full bg-accent-cyan/20 text-accent-cyan hover:bg-accent-cyan/30 transition-colors">{playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}</button>
          <button onClick={() => jumpTo(maxSliderPos)} className="w-7 h-7 flex items-center justify-center rounded-full text-text-secondary hover:text-text-primary hover:bg-white/10 transition-colors" title="End"><ChevronLast className="w-4 h-4" /></button>
          {currentPeriod && (<div className="text-sm ml-1"><span className="text-text-primary font-medium">{formatPeriod(currentPeriod)}</span><span className="text-text-secondary ml-2">- {getSeason(currentPeriod)}</span></div>)}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-white/5 rounded-full border border-white/10 px-0.5 py-0.5">{SPEEDS.map((s, i) => (<button key={s.label} onClick={() => setSpeedIdx(i)} className={`px-2 py-0.5 rounded-full text-[10px] font-medium transition-colors ${i === speedIdx ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`}>{s.label}</button>))}</div>
          <button onClick={handleDeactivate} className="p-1.5 rounded hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors" title="Close timeline"><X className="w-4 h-4" /></button>
        </div>
      </div>
      <div className="relative">
        <input type="range" min={0} max={maxSliderPos} value={sliderPos} onChange={handleSliderChange} disabled={filteredIndices.length === 0} className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent-cyan [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(34,211,238,0.4)] [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-accent-cyan [&::-moz-range-thumb]:border-0" />
        <div className="flex justify-between mt-1 text-[9px] text-text-secondary px-1">{years.filter((_, i) => i % Math.max(1, Math.floor(years.length / 10)) === 0 || i === years.length - 1).map(year => (<span key={year}>{year}</span>))}</div>
      </div>
    </div>
  )
}
