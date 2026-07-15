// Pure formatting helpers for the Climat "Situation" synthesis banner
// (fed by GET /observatory/climat/situation-summary). Kept separate from the
// component so the text-shaping logic is unit-testable without rendering React
// or initialising i18next.
import type { ClimatSituationSummary } from './observatory-types'

/** fr/en locale tag — plain string, no i18next dependency (this module stays
 *  React/i18n-free, see buildSituationBannerData below). */
export type FormatLocale = 'fr' | 'en'

/** "48.2°N, 1.7°E" (east) or "48.2°N, 1.7°O"/"1.7°W" (west, locale-aware) —
 *  coarse lat/lon label (no reverse geocoding). Not currently consumed by
 *  SituationBanner (which now shows a classes_pct distribution bar, not
 *  per-cell driest chips) but kept as a standalone tested utility for any
 *  future coordinate display. Mainland France is always north, so the
 *  latitude cardinal is fixed; the longitude cardinal reflects the sign
 *  instead of a bare minus, which reads as a bug to hydro/climate experts
 *  (western cells — Bretagne, Charente — are common in the driest list). */
export function formatLatLon(lat: number, lon: number, locale: FormatLocale = 'fr'): string {
  const lonCardinal = lon < 0 ? (locale === 'en' ? 'W' : 'O') : 'E'
  return `${lat.toFixed(1)}°N, ${Math.abs(lon).toFixed(1)}°${lonCardinal}`
}

/** Rounds the drought percentage to a single decimal for display (backend already rounds to 2). */
export function formatDroughtPct(pct: number): string {
  return pct.toFixed(1).replace('.0', '')
}

export interface SituationBannerData {
  pctSecheresse: string
  /** Year to show in "mois le plus sec depuis AAAA" — null when there's no historical comparison data. */
  driestSinceYear: number | null
}

/** Derives display-ready fields from the raw API summary — no i18n, no React.
 *  No `locale` param anymore: the banner now shows a classes_pct distribution
 *  bar instead of per-cell driest chips, so there's no lat/lon text to
 *  localise here (see formatLatLon for that, still used/tested standalone). */
export function buildSituationBannerData(summary: ClimatSituationSummary): SituationBannerData {
  return {
    pctSecheresse: formatDroughtPct(summary.pct_secheresse),
    driestSinceYear: summary.driest_since_year,
  }
}
