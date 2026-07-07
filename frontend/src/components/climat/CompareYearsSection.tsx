import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useClimatCompareYears } from '@/hooks/useClimat'
import { defaultCompareYears } from '@/lib/climat-year-select'
import { YearMultiSelect } from './YearMultiSelect'
import { CompareCumulChart } from './CompareCumulChart'
import { CompareSpiMiniMaps } from './CompareSpiMiniMaps'

interface Props {
  lat: number
  lon: number
}

// June — a reasonable default for "how dry was mid-year across drought years"; the
// picker lets the user move it (plan: "keep simple: a single month picker").
const DEFAULT_MONTH = 6

/** Comparaison section (Task B3) of the Point panel — attaches to the same selected
 *  cell as PointPanel's history/episodes sections. Multi-select of years (default:
 *  famous drought presets + current year), superposed cumulative-precipitation
 *  curves, and petits multiples SPI maps for one chosen month across the selected
 *  years. */
export function CompareYearsSection({ lat, lon }: Props) {
  const { t } = useTranslation()
  const [years, setYears] = useState<number[]>(() => defaultCompareYears(new Date().getFullYear()))
  const [month, setMonth] = useState(DEFAULT_MONTH)
  const { data, isLoading, isError } = useClimatCompareYears(lat, lon, years)

  return (
    <div className="border-t border-white/10 pt-4">
      <h3 className="text-sm font-semibold text-text-primary mb-2">{t('climat.compare.title')}</h3>
      <YearMultiSelect years={years} onChange={setYears} />

      {isError && <p className="text-xs text-red-400 mt-2">{t('climat.pointPanel.loadFailed')}</p>}

      {isLoading && !isError && (
        <div className="mt-3 space-y-3">
          <div className="h-48 w-full bg-white/5 rounded-lg animate-pulse" />
          <div className="h-40 w-full bg-white/5 rounded-lg animate-pulse" />
        </div>
      )}

      {!isLoading && !isError && (
        <div className="mt-3 space-y-4">
          <CompareCumulChart data={data} years={years} />
          <CompareSpiMiniMaps years={years} month={month} onMonthChange={setMonth} />
        </div>
      )}
    </div>
  )
}
