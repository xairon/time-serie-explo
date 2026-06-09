import { useRef, useCallback } from 'react'

interface Props {
  periods: string[]
  selectedPeriod: string | null
  onChange: (p: string) => void
}

const FR_MONTHS = [
  'janv.', 'févr.', 'mars', 'avr.', 'mai', 'juin',
  'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.',
]

function formatPeriodFR(period: string): string {
  const [year, month] = period.split('-')
  const monthIndex = parseInt(month, 10) - 1
  if (monthIndex < 0 || monthIndex > 11) return period
  return `${FR_MONTHS[monthIndex]} ${year}`
}

function getMonth(period: string): number {
  return parseInt(period.split('-')[1], 10) // 1-based
}

function getYear(period: string): string {
  return period.split('-')[0]
}

export function SituationTimelineSlider({ periods, selectedPeriod, onChange }: Props) {
  if (periods.length === 0) return null

  const currentIndex = selectedPeriod
    ? Math.max(0, periods.indexOf(selectedPeriod))
    : periods.length - 1

  const displayPeriod = periods[currentIndex] ?? periods[periods.length - 1]

  // Decide label density: show abbrev label every N ticks
  const n = periods.length
  const labelEvery = n <= 24 ? 1 : n <= 48 ? 2 : 3

  // Fraction position of a tick (0..1)
  function tickPct(i: number): number {
    if (periods.length === 1) return 0
    return (i / (periods.length - 1)) * 100
  }

  const filledPct = periods.length === 1 ? 100 : (currentIndex / (periods.length - 1)) * 100

  const trackRef = useRef<HTMLDivElement>(null)

  // Allow clicking directly on the track to jump to nearest tick
  const handleTrackClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = trackRef.current?.getBoundingClientRect()
    if (!rect || periods.length <= 1) return
    const x = e.clientX - rect.left
    const ratio = Math.max(0, Math.min(1, x / rect.width))
    const idx = Math.round(ratio * (periods.length - 1))
    onChange(periods[idx])
  }, [periods, onChange])

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const idx = parseInt(e.target.value, 10)
    const clamped = Math.max(0, Math.min(idx, periods.length - 1))
    onChange(periods[clamped])
  }

  return (
    <div className="absolute bottom-0 left-0 right-0 z-10 bg-white border-t border-slate-200 shadow-md"
      style={{ height: 60 }}>

      {/* Current selected label — top left */}
      <div className="absolute top-1 left-4 text-xs font-bold text-slate-800 whitespace-nowrap leading-tight">
        {formatPeriodFR(displayPeriod)}
      </div>

      {/* Track area */}
      <div
        className="absolute left-4 right-4"
        style={{ top: 22, height: 28 }}
      >
        {/* Clickable background track */}
        <div
          ref={trackRef}
          className="absolute left-0 right-0 cursor-pointer"
          style={{ top: 6, height: 16 }}
          onClick={handleTrackClick}
        >
          {/* Track line */}
          <div className="absolute left-0 right-0 top-[7px] h-[3px] rounded-full bg-slate-200" />

          {/* Filled portion */}
          <div
            className="absolute left-0 top-[7px] h-[3px] rounded-full bg-blue-500 transition-all"
            style={{ width: `${filledPct}%` }}
          />

          {/* Tick marks */}
          {periods.map((p, i) => {
            const pct = tickPct(i)
            const isJan = getMonth(p) === 1
            const isSelected = i === currentIndex
            return (
              <div
                key={p}
                className="absolute top-0 -translate-x-1/2"
                style={{ left: `${pct}%` }}
              >
                {/* Tick line */}
                <div
                  className={`absolute left-0 -translate-x-1/2 ${
                    isJan ? 'h-4 w-[2px] bg-slate-400 top-[1px]' : 'h-2.5 w-px bg-slate-300 top-[3px]'
                  }`}
                />
                {/* Selected dot */}
                {isSelected && (
                  <div className="absolute left-0 -translate-x-1/2 top-[3px] w-3.5 h-3.5 rounded-full bg-blue-600 border-2 border-white shadow-md -translate-y-[3px]" />
                )}
              </div>
            )
          })}
        </div>

        {/* Accessible range input layered over the track (invisible but functional) */}
        <input
          type="range"
          min={0}
          max={periods.length - 1}
          step={1}
          value={currentIndex}
          onChange={handleChange}
          className="absolute left-0 right-0 w-full opacity-0 cursor-pointer"
          style={{ top: 0, height: 16 }}
          aria-label="Sélectionner une période"
          aria-valuetext={formatPeriodFR(displayPeriod)}
        />

        {/* Labels row below ticks */}
        <div className="absolute left-0 right-0" style={{ top: 18 }}>
          {periods.map((p, i) => {
            const pct = tickPct(i)
            const month = getMonth(p)
            const isJan = month === 1
            const showLabel = isJan || (i % labelEvery === 0)
            if (!showLabel) return null

            return (
              <span
                key={`lbl-${p}`}
                className={`absolute -translate-x-1/2 leading-none whitespace-nowrap select-none pointer-events-none ${
                  isJan
                    ? 'text-[9px] font-bold text-slate-700'
                    : 'text-[8px] text-slate-400'
                }`}
                style={{ left: `${pct}%` }}
              >
                {isJan ? getYear(p) : FR_MONTHS[month - 1]}
              </span>
            )
          })}
        </div>
      </div>
    </div>
  )
}
