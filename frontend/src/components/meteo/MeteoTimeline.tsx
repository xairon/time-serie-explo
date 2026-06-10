// frontend/src/components/meteo/MeteoTimeline.tsx
// Bottom bar clone: rolling 12-month chips + 3 greyed future slots,
// year labels at January, date chip bottom-left opening the full-history picker.
import { useState } from 'react'
import { buildTimelineWindow, formatPeriodLongFR, FR_MONTHS_LONG } from '@/lib/meteo-timeline'
import { MeteoDatePicker } from './MeteoDatePicker'

interface Props {
  periods: string[]              // all available 'YYYY-MM', ascending
  selected: string
  onChange: (p: string) => void
}

function monthLong(period: string): string {
  return FR_MONTHS_LONG[parseInt(period.split('-')[1], 10) - 1] ?? period
}

export function MeteoTimeline({ periods, selected, onChange }: Props) {
  const [pickerOpen, setPickerOpen] = useState(false)
  if (periods.length === 0) return null
  const latest = periods[periods.length - 1]
  const cells = buildTimelineWindow(periods, selected)

  return (
    <div className="absolute bottom-0 left-0 right-0 z-20 bg-white/95 border-t border-slate-200 shadow-[0_-2px_8px_rgba(0,0,0,0.06)] h-12 flex items-center">
      {/* Date chip + picker */}
      <div className="relative flex items-center gap-1 pl-3 pr-4 flex-shrink-0">
        {pickerOpen && (
          <MeteoDatePicker
            periods={periods}
            selected={selected}
            onSelect={onChange}
            onClose={() => setPickerOpen(false)}
          />
        )}
        <button
          onClick={() => setPickerOpen(o => !o)}
          aria-label="Choisir une date"
          className="flex items-center gap-1.5 border border-slate-300 rounded px-2.5 py-1 text-xs text-slate-700 hover:border-slate-400 bg-white"
        >
          {formatPeriodLongFR(selected)}
          <svg width="9" height="6" viewBox="0 0 9 6" aria-hidden="true"><path d="M1 1l3.5 3.5L8 1" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" /></svg>
        </button>
        {selected !== latest && (
          <button
            onClick={() => onChange(latest)}
            aria-label="Revenir au mois le plus récent"
            className="text-slate-400 hover:text-slate-600 px-1 text-sm leading-none"
          >×</button>
        )}
      </div>

      {/* Month chips */}
      <div className="flex-1 flex items-center pr-4 min-w-0">
        {cells.map((c) => {
          const isSelected = c.period === selected
          const year = c.period.split('-')[0]
          return (
            <button
              key={c.period}
              disabled={!c.available}
              onClick={() => onChange(c.period)}
              aria-label={`mois ${monthLong(c.period)} ${year}`}
              aria-current={isSelected ? 'date' : undefined}
              className="flex-1 flex flex-col items-center gap-0.5 group min-w-0 disabled:cursor-default"
            >
              <span className={`text-[11px] leading-none truncate max-w-full px-0.5 ${
                isSelected ? 'font-bold text-blue-700'
                : c.available ? 'text-slate-600 group-hover:text-slate-900'
                : 'text-slate-300'
              }`}>
                {monthLong(c.period)}
                {c.showYear && <span className="ml-1 font-semibold text-slate-400">{year}</span>}
              </span>
              <span className={`w-2 h-2 rounded-full border ${
                isSelected ? 'bg-blue-600 border-blue-600 scale-125'
                : c.available ? 'bg-white border-slate-300 group-hover:border-slate-400'
                : 'bg-slate-100 border-slate-200'
              }`} aria-hidden="true" />
            </button>
          )
        })}
      </div>
    </div>
  )
}
