import { useTranslation } from 'react-i18next'
import type { ClimatDailyTempPoint } from '@/lib/observatory-types'
import type { ClimatVariable } from '@/lib/climat-colors'
import { CLIMAT_VARIABLES, PRECIP_DAILY_BOUNDS } from '@/lib/climat-colors'
import { buildDailyBannerData } from '@/lib/climat-daily-format'

interface Props {
  variable: ClimatVariable
  points: ClimatDailyTempPoint[] | undefined
  isLoading: boolean
}

/** Territory-wide synthesis for the daily layers (Tx/Tn/Tmoy and Pluie) —
 *  "Tx max France : 43,2 °C · 12 cellules > 35 °C" or "Pluie max France : 98,8 mm
 *  · 3 cellules ≥ 50 mm", computed client-side from the already-loaded grid
 *  response (see climat-daily-format.ts). Renders in the same slot as
 *  SituationBanner (SPI/STI) but is NOT server-aggregated: there is no
 *  situation-summary equivalent for the daily grid. */
export function DailyBanner({ variable, points, isLoading }: Props) {
  const { t, i18n } = useTranslation()

  if (isLoading || !points) {
    return (
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-4 py-2 shadow-lg text-xs text-text-secondary">
        {t('climat.banner.loading')}
      </div>
    )
  }

  const locale = i18n.language?.startsWith('en') ? 'en' : 'fr'
  const isPrecip = variable === 'precip_daily'
  const TOP_PRECIP = PRECIP_DAILY_BOUNDS[PRECIP_DAILY_BOUNDS.length - 1]
  const data = buildDailyBannerData(points, locale, isPrecip ? (v) => v >= TOP_PRECIP : undefined)

  if (!data) {
    return (
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-4 py-2 shadow-lg text-xs text-text-secondary">
        {t('climat.banner.dailyTempUnavailable')}
      </div>
    )
  }

  return (
    <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 max-w-2xl bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-4 py-2.5 shadow-lg">
      <span className="font-semibold text-accent-cyan text-xs">
        {t(isPrecip ? 'climat.banner.dailyPrecipSummary' : 'climat.banner.dailyTempSummary', {
          variable: t(CLIMAT_VARIABLES[variable].labelKey),
          max: data.maxValueLabel,
          count: data.countAboveThreshold,
          threshold: TOP_PRECIP,
        })}
      </span>
    </div>
  )
}
