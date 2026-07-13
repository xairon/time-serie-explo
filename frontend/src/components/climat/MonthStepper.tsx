import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { addMonths, comparePeriods } from '@/lib/period-arithmetic'

interface Props {
  /** Current month, 'YYYY-MM'. */
  month: string
  onChange: (month: string) => void
  /** Bounds as 'YYYY-MM' (or full 'YYYY-MM-DD', only the first 7 chars are used). */
  minMonth?: string
  maxMonth?: string
}

/** ‹ month › stepper for the Climat Situation view — pure month arithmetic via period-arithmetic's addMonths. */
export function MonthStepper({ month, onChange, minMonth, maxMonth }: Props) {
  const { t, i18n } = useTranslation()
  const min = minMonth?.slice(0, 7)
  const max = maxMonth?.slice(0, 7)
  const prevMonth = addMonths(month, -1)
  const nextMonth = addMonths(month, 1)
  const canPrev = !min || comparePeriods(prevMonth, min) >= 0
  const canNext = !max || comparePeriods(nextMonth, max) <= 0

  const m = month.match(/^(\d{4})-(\d{2})/)
  const monthLabel = m
    ? new Intl.DateTimeFormat(i18n.language, { month: 'long', year: 'numeric' }).format(new Date(Number(m[1]), Number(m[2]) - 1, 1))
    : month

  return (
    <div className="flex items-center gap-1 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-1 py-1 shadow-lg">
      <button
        type="button"
        onClick={() => canPrev && onChange(prevMonth)}
        disabled={!canPrev}
        aria-label={t('climat.stepper.prevMonth')}
        className="p-1.5 rounded hover:bg-bg-hover disabled:opacity-30 disabled:hover:bg-transparent text-text-secondary"
      >
        <ChevronLeft className="w-4 h-4" />
      </button>
      <span className="text-sm text-text-primary min-w-[9rem] text-center capitalize select-none">
        {monthLabel}
      </span>
      <button
        type="button"
        onClick={() => canNext && onChange(nextMonth)}
        disabled={!canNext}
        aria-label={t('climat.stepper.nextMonth')}
        className="p-1.5 rounded hover:bg-bg-hover disabled:opacity-30 disabled:hover:bg-transparent text-text-secondary"
      >
        <ChevronRight className="w-4 h-4" />
      </button>
    </div>
  )
}
