// Pure formatting helpers for the Climat "Situation" synthesis banner
// (fed by GET /observatory/climat/situation-summary). Kept separate from the
// component so the text-shaping logic is unit-testable without rendering React
// or initialising i18next.
import type { ClimatSituationSummary } from './observatory-types'

export interface SituationBannerChip {
  label: string
  latitude: number
  longitude: number
}

/** fr/en locale tag — plain string, no i18next dependency (this module stays
 *  React/i18n-free, see buildSituationBannerData below). */
export type FormatLocale = 'fr' | 'en'

/** "48.2°N, 1.7°E" (east) or "48.2°N, 1.7°O"/"1.7°W" (west, locale-aware) —
 *  coarse lat/lon label for a driest-cell chip (no reverse geocoding yet, per
 *  plan). Mainland France is always north, so the latitude cardinal is fixed;
 *  the longitude cardinal reflects the sign instead of a bare minus, which
 *  reads as a bug to hydro/climate experts (western cells — Bretagne,
 *  Charente — are common in the top-5 driest list). */
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
  chips: SituationBannerChip[]
}

/** Derives display-ready fields from the raw API summary — no i18n, no React;
 *  `locale` is a plain 'fr'|'en' tag the caller derives from i18next. */
export function buildSituationBannerData(
  summary: ClimatSituationSummary,
  locale: FormatLocale = 'fr',
): SituationBannerData {
  return {
    pctSecheresse: formatDroughtPct(summary.pct_secheresse),
    driestSinceYear: summary.driest_since_year,
    chips: summary.top5_cellules_seches.map((c) => ({
      label: formatLatLon(c.latitude, c.longitude, locale),
      latitude: c.latitude,
      longitude: c.longitude,
    })),
  }
}
