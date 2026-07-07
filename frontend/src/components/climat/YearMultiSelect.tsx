import { useTranslation } from 'react-i18next'
import { X } from 'lucide-react'
import { DROUGHT_YEAR_PRESETS, MAX_COMPARE_YEARS, MIN_COMPARE_YEARS, toggleYear } from '@/lib/climat-year-select'

interface Props {
  years: number[]
  onChange: (years: number[]) => void
}

const CURRENT_YEAR = new Date().getFullYear()
// Descending so the current year sits at the top of the "add year" dropdown.
const ADDABLE_YEARS = Array.from({ length: CURRENT_YEAR - 1949 }, (_, i) => CURRENT_YEAR - i)

/** Year multi-select for the Comparaison view (Task B3) — famous-drought preset chips
 *  (always shown, toggleable) plus already-selected non-preset years, and a dropdown
 *  to add any other year. Bounds (2-6) are enforced by climat-year-select's toggleYear,
 *  which this component treats as the single source of truth (a chip that would break
 *  a bound just gets disabled rather than duplicating the rule here). */
export function YearMultiSelect({ years, onChange }: Props) {
  const { t } = useTranslation()
  const chipYears = Array.from(new Set<number>([...DROUGHT_YEAR_PRESETS, ...years])).sort((a, b) => a - b)
  const atMax = years.length >= MAX_COMPARE_YEARS

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <span className="text-[10px] text-text-secondary">
          {t('climat.compare.yearsHint', { min: MIN_COMPARE_YEARS, max: MAX_COMPARE_YEARS })}
        </span>
        <select
          value=""
          onChange={(e) => { if (e.target.value) onChange(toggleYear(years, Number(e.target.value))) }}
          aria-label={t('climat.compare.addYear')}
          disabled={atMax}
          className="bg-bg-hover border border-white/10 rounded px-1.5 py-0.5 text-text-primary text-[11px] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <option value="">{t('climat.compare.addYear')}</option>
          {ADDABLE_YEARS.filter((y) => !years.includes(y)).map((y) => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>
      </div>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label={t('climat.compare.yearsLabel')}>
        {chipYears.map((year) => {
          const selected = years.includes(year)
          const disabled = selected ? years.length <= MIN_COMPARE_YEARS : atMax
          return (
            <button
              key={year}
              type="button"
              onClick={() => onChange(toggleYear(years, year))}
              aria-pressed={selected}
              disabled={disabled}
              className={`flex items-center gap-1 text-xs px-2.5 py-1 rounded-full transition-colors border ${
                selected
                  ? 'bg-accent-cyan/20 text-accent-cyan border-accent-cyan/30'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover border-white/10'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {year}
              {selected && <X className="w-3 h-3" />}
            </button>
          )
        })}
      </div>
    </div>
  )
}
