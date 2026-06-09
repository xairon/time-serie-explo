import type { SectorSituation, SituationClass } from '@/lib/observatory-types'

// Our 7-enum → BRGM hex (extracted from the real WFS indicateur_bsn 'color'/'class' fields).
// Classes 1–2 (driest) never appeared in BRGM artifacts → reds interpolated.
export const METEO_CLASS_COLORS: Record<string, string> = {
  EXTREMEMENT_BAS: '#d73027',
  TRES_BAS: '#e84a1a',
  BAS: '#f8930f',
  NORMAL: '#ffde1a',
  HAUT: '#60a3d6',
  TRES_HAUT: '#3071b0',
  EXTREMEMENT_HAUT: '#00408b',
  UNKNOWN: '#d9d9d9',
}

// BRGM's 7 label strings (used ONLY on /meteo; do NOT touch the Observatory CLASSIFICATION_LABELS).
export const METEO_CLASS_LABELS: Record<string, string> = {
  EXTREMEMENT_BAS: 'très bas',
  TRES_BAS: 'bas',
  BAS: 'modérément bas',
  NORMAL: 'autour de la moyenne',
  HAUT: 'modérément haut',
  TRES_HAUT: 'haut',
  EXTREMEMENT_HAUT: 'très haut',
  UNKNOWN: 'Sans nappe libre étendue / Absence de point de suivi',
}

export const METEO_TREND_LABELS: Record<string, string> = {
  hausse: 'en hausse',
  stable: 'stable',
  baisse: 'en baisse',
}

export function meteoClassColor(cls: SituationClass | null | undefined): string {
  return METEO_CLASS_COLORS[cls ?? 'UNKNOWN'] ?? METEO_CLASS_COLORS.UNKNOWN
}

/** Flat [sector_id(number), hex, ...] for a MapLibre `match` on ['get','sector_id']. */
export function meteoSectorColorPairs(sits: SectorSituation[]): (number | string)[] {
  const out: (number | string)[] = []
  for (const s of sits) {
    const hex = s.insufficient ? METEO_CLASS_COLORS.UNKNOWN : meteoClassColor(s.situation_class)
    out.push(Number(s.code), hex)
  }
  return out
}
