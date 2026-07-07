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

/** "48.2°N, 1.7°E" — coarse lat/lon label for a driest-cell chip (no reverse geocoding yet, per plan). */
export function formatLatLon(lat: number, lon: number): string {
  return `${lat.toFixed(1)}°N, ${lon.toFixed(1)}°E`
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

/** Derives display-ready fields from the raw API summary — no i18n, no React. */
export function buildSituationBannerData(summary: ClimatSituationSummary): SituationBannerData {
  return {
    pctSecheresse: formatDroughtPct(summary.pct_secheresse),
    driestSinceYear: summary.driest_since_year,
    chips: summary.top5_cellules_seches.map((c) => ({
      label: formatLatLon(c.latitude, c.longitude),
      latitude: c.latitude,
      longitude: c.longitude,
    })),
  }
}
