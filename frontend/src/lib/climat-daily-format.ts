// Pure formatting for the Climat daily-layer synthesis banner (temperature AND
// precipitation). Unlike SituationBanner's server-aggregated SPI summary
// (climat-situation-format.ts), this is computed client-side straight from the
// already-loaded grid response — no extra endpoint for a one-line max + count.
// Kept React/i18n-free so the text-shaping logic is unit-testable directly
// (same split as climat-situation-format.ts).
import type { ClimatDailyTempPoint } from './observatory-types'
import type { FormatLocale } from './climat-situation-format'

/** Threshold (°C) above which a cell counts as a "hot cell" in the banner —
 *  the classic canicule/heatwave-day marker used in French climate reporting. */
export const HEAT_CELL_THRESHOLD_C = 35

export interface DailyBannerData {
  /** Locale-formatted max value, one decimal, no unit (e.g. "43,2" fr / "43.2" en). */
  maxValueLabel: string
  countAboveThreshold: number
}

/** One-decimal locale-aware number formatting — 'fr' uses a comma decimal
 *  separator (fr-FR), 'en' a period (en-US). Mirrors observatory-utils'
 *  formatNumber but locale-aware (that helper is hardcoded to fr-FR). */
export function formatOneDecimal(value: number, locale: FormatLocale = 'fr'): string {
  return value.toLocaleString(locale === 'en' ? 'en-US' : 'fr-FR', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })
}

/** Max + comptage au-dessus d'un seuil, pour le bandeau des couches journalières.
 *  Le prédicat est passé par l'appelant : la température compte `> 35 °C` (marqueur
 *  canicule), la pluie `>= 50 mm` (la borne haute de sa légende, donc vérifiable
 *  à l'œil dessus). Rend null quand aucune maille n'a de valeur, pour que
 *  l'appelant affiche « indisponible » au lieu d'un « 0 cellules » trompeur. */
export function buildDailyBannerData(
  points: ClimatDailyTempPoint[],
  locale: FormatLocale = 'fr',
  countIf: (v: number) => boolean = (v) => v > HEAT_CELL_THRESHOLD_C,
): DailyBannerData | null {
  const values = points.map((p) => p.value).filter((v): v is number => v != null)
  if (values.length === 0) return null
  return {
    maxValueLabel: formatOneDecimal(Math.max(...values), locale),
    countAboveThreshold: values.filter(countIf).length,
  }
}
