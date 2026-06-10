// frontend/src/components/meteo/MeteoDatePicker.tsx
// Month/year picker over the full data history (our edge over the original).
import { useState, useEffect } from 'react'
import { FR_MONTHS_SHORT } from '@/lib/meteo-timeline'

interface Props {
  periods: string[]          // all available 'YYYY-MM', ascending — must be non-empty
  selected: string
  onSelect: (p: string) => void
  onClose: () => void
}

export function MeteoDatePicker({ periods, selected, onSelect, onClose }: Props) {
  const [year, setYear] = useState(Number(selected.split('-')[0]))

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  if (periods.length === 0) return null
  const available = new Set(periods)
  const firstYear = Number(periods[0].split('-')[0])
  const lastYear = Number(periods[periods.length - 1].split('-')[0])

  return (
    <div className="absolute bottom-12 left-0 z-30 bg-white rounded-lg shadow-lg border border-slate-200 p-3 w-60">
      <div className="flex items-center justify-between mb-2">
        <button
          onClick={() => setYear(y => Math.max(firstYear, y - 1))}
          disabled={year <= firstYear}
          aria-label="Année précédente"
          className="px-2 py-0.5 rounded hover:bg-slate-100 disabled:opacity-30 text-slate-600"
        >‹</button>
        <span className="text-sm font-semibold text-slate-800">{year}</span>
        <button
          onClick={() => setYear(y => Math.min(lastYear, y + 1))}
          disabled={year >= lastYear}
          aria-label="Année suivante"
          className="px-2 py-0.5 rounded hover:bg-slate-100 disabled:opacity-30 text-slate-600"
        >›</button>
      </div>
      <div className="grid grid-cols-4 gap-1">
        {FR_MONTHS_SHORT.map((label, i) => {
          const p = `${year}-${String(i + 1).padStart(2, '0')}`
          const ok = available.has(p)
          const isSel = p === selected
          return (
            <button
              key={p}
              disabled={!ok}
              onClick={() => { onSelect(p); onClose() }}
              className={`text-[11px] rounded px-1 py-1.5 ${
                isSel ? 'bg-blue-600 text-white font-semibold'
                : ok ? 'text-slate-700 hover:bg-slate-100'
                : 'text-slate-300 cursor-not-allowed'
              }`}
            >
              {label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
