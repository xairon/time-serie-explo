import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { addDays, comparePeriods, formatDayLabel } from '@/lib/climat-day-stepper'

interface Props {
  /** Current day, 'YYYY-MM-DD'. */
  day: string
  onChange: (day: string) => void
  /** Bounds as 'YYYY-MM-DD'. */
  minDay?: string
  maxDay?: string
}

/** ‹ day › stepper for the Climat daily-temperature layer (Tx/Tn/Tmoy) —
 *  day-granularity sibling of MonthStepper, same visual style and step logic
 *  (pure date arithmetic via climat-day-stepper's addDays/comparePeriods, the
 *  latter reused as-is from period-arithmetic since ISO date strings already sort
 *  lexicographically). Replaces MonthStepper on the map when a daily-temp
 *  variable is selected (see ClimatPage). */
export function DayStepper({ day, onChange, minDay, maxDay }: Props) {
  const { t, i18n } = useTranslation()
  const prevDay = addDays(day, -1)
  const nextDay = addDays(day, 1)
  const canPrev = !minDay || comparePeriods(prevDay, minDay) >= 0
  const canNext = !maxDay || comparePeriods(nextDay, maxDay) <= 0
  const dayLabel = formatDayLabel(day, i18n.language)

  return (
    <div className="flex items-center gap-1 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-1 py-1 shadow-lg">
      <button
        type="button"
        onClick={() => canPrev && onChange(prevDay)}
        disabled={!canPrev}
        aria-label={t('climat.stepper.prevDay')}
        className="p-1.5 rounded hover:bg-bg-hover disabled:opacity-30 disabled:hover:bg-transparent text-text-secondary"
      >
        <ChevronLeft className="w-4 h-4" />
      </button>
      <span className="text-sm text-text-primary min-w-[9rem] text-center capitalize select-none">
        {dayLabel}
      </span>
      <button
        type="button"
        onClick={() => canNext && onChange(nextDay)}
        disabled={!canNext}
        aria-label={t('climat.stepper.nextDay')}
        className="p-1.5 rounded hover:bg-bg-hover disabled:opacity-30 disabled:hover:bg-transparent text-text-secondary"
      >
        <ChevronRight className="w-4 h-4" />
      </button>
    </div>
  )
}
