import { ChevronLeft, ChevronRight } from 'lucide-react'
import { addMonths, comparePeriods, formatPeriodLongFR } from '@/lib/meteo-timeline'

interface Props {
  /** Current month, 'YYYY-MM'. */
  month: string
  onChange: (month: string) => void
  /** Bounds as 'YYYY-MM' (or full 'YYYY-MM-DD', only the first 7 chars are used). */
  minMonth?: string
  maxMonth?: string
}

/** ‹ month › stepper for the Climat Situation view — pure month arithmetic via meteo-timeline's addMonths. */
export function MonthStepper({ month, onChange, minMonth, maxMonth }: Props) {
  const min = minMonth?.slice(0, 7)
  const max = maxMonth?.slice(0, 7)
  const prevMonth = addMonths(month, -1)
  const nextMonth = addMonths(month, 1)
  const canPrev = !min || comparePeriods(prevMonth, min) >= 0
  const canNext = !max || comparePeriods(nextMonth, max) <= 0

  return (
    <div className="flex items-center gap-1 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-1 py-1 shadow-lg">
      <button
        type="button"
        onClick={() => canPrev && onChange(prevMonth)}
        disabled={!canPrev}
        aria-label="Mois précédent"
        className="p-1.5 rounded hover:bg-bg-hover disabled:opacity-30 disabled:hover:bg-transparent text-text-secondary"
      >
        <ChevronLeft className="w-4 h-4" />
      </button>
      <span className="text-sm text-text-primary min-w-[9rem] text-center capitalize select-none">
        {formatPeriodLongFR(month)}
      </span>
      <button
        type="button"
        onClick={() => canNext && onChange(nextMonth)}
        disabled={!canNext}
        aria-label="Mois suivant"
        className="p-1.5 rounded hover:bg-bg-hover disabled:opacity-30 disabled:hover:bg-transparent text-text-secondary"
      >
        <ChevronRight className="w-4 h-4" />
      </button>
    </div>
  )
}
