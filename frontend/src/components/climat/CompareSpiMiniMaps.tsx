import { useTranslation } from 'react-i18next'
import { useClimatCompareGridIndices } from '@/hooks/useClimat'
import { MiniSpiMap } from './MiniSpiMap'

interface Props {
  years: number[]
  month: number
  onMonthChange: (month: number) => void
}

/** Petits multiples SPI (Task B3) — one mini-map per selected year, all showing the
 *  same chosen calendar month, side by side for a quick "how did this month compare
 *  across drought years" read. The month picker stays deliberately simple (a single
 *  <select>, default June) rather than another full MonthStepper. */
export function CompareSpiMiniMaps({ years, month, onMonthChange }: Props) {
  const { t, i18n } = useTranslation()
  const localeTag = i18n.language?.startsWith('en') ? 'en-US' : 'fr-FR'
  const monthNames = Array.from({ length: 12 }, (_, i) =>
    new Intl.DateTimeFormat(localeTag, { month: 'long' }).format(new Date(2000, i, 1)),
  )
  const results = useClimatCompareGridIndices(years, month, years.length > 0)

  return (
    <div>
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <h4 className="text-xs font-semibold text-text-primary">{t('climat.compare.miniMapsTitle')}</h4>
        <label className="flex items-center gap-1.5 text-[11px] text-text-secondary">
          {t('climat.compare.monthLabel')}
          <select
            value={month}
            onChange={(e) => onMonthChange(Number(e.target.value))}
            className="bg-bg-hover border border-white/10 rounded px-1.5 py-0.5 text-text-primary text-xs capitalize"
          >
            {monthNames.map((name, i) => (
              <option key={i + 1} value={i + 1}>{name}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="flex flex-wrap gap-3 justify-center">
        {years.map((year, i) => {
          const q = results[i]
          if (q.isLoading) {
            return <div key={year} className="w-[180px] h-[204px] bg-white/5 rounded animate-pulse" />
          }
          if (q.isError || !q.data) {
            return (
              <div
                key={year}
                className="w-[180px] h-[204px] flex flex-col items-center justify-center gap-1 rounded border border-white/10 bg-black/20 text-center px-2"
              >
                <span className="text-[11px] text-text-secondary">{t('climat.compare.dataUnavailable')}</span>
                <span className="text-[11px] text-text-secondary">{year}</span>
              </div>
            )
          }
          return <MiniSpiMap key={year} points={q.data} label={String(year)} />
        })}
      </div>
    </div>
  )
}
