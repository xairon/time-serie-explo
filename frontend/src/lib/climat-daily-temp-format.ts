// Pure formatting for the Climat daily-temperature synthesis banner. Unlike
// SituationBanner's server-aggregated SPI summary (climat-situation-format.ts),
// this is computed client-side straight from the already-loaded grid response —
// no extra endpoint for a one-line max + heat-cell count. Kept React/i18n-free
// so the text-shaping logic is unit-testable directly (same split as
// climat-situation-format.ts).
import type { ClimatDailyTempPoint } from './observatory-types'
import type { FormatLocale } from './climat-situation-format'

/** Threshold (°C) above which a cell counts as a "hot cell" in the banner —
 *  the classic canicule/heatwave-day marker used in French climate reporting. */
export const HEAT_CELL_THRESHOLD_C = 35

export interface DailyTempBannerData {
  /** Locale-formatted max value, one decimal, no unit (e.g. "43,2" fr / "43.2" en). */
  maxValueLabel: string
  countAboveThreshold: number
}

/** One-decimal locale-aware number formatting — 'fr' uses a comma decimal
 *  separator (fr-FR), 'en' a period (en-US). Mirrors observatory-utils'
 *  formatNumber but locale-aware (that helper is hardcoded to fr-FR). */
export function formatTemperature(value: number, locale: FormatLocale = 'fr'): string {
  return value.toLocaleString(locale === 'en' ? 'en-US' : 'fr-FR', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })
}

/** Derives the banner's max value + hot-cell count from the grid response.
 *  Returns null when every cell is null (no data yet for this day) so the
 *  caller can fall back to an "unavailable" message instead of a misleading
 *  "0 cellules". */
export function buildDailyTempBannerData(
  points: ClimatDailyTempPoint[],
  locale: FormatLocale = 'fr',
): DailyTempBannerData | null {
  const values = points.map((p) => p.value).filter((v): v is number => v != null)
  if (values.length === 0) return null
  const max = Math.max(...values)
  const countAboveThreshold = values.filter((v) => v > HEAT_CELL_THRESHOLD_C).length
  return { maxValueLabel: formatTemperature(max, locale), countAboveThreshold }
}
