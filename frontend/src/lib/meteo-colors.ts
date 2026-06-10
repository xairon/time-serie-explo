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
  inconnu: 'Inconnu',
}

export function meteoClassColor(cls: SituationClass | null | undefined): string {
  return METEO_CLASS_COLORS[cls ?? 'UNKNOWN'] ?? METEO_CLASS_COLORS.UNKNOWN
}

// BRGM class index (0..7) -> our enum. 0 = no extensive free aquifer (grey).
export const BRGM_CLASS_TO_ENUM: Record<number, SituationClass | 'UNKNOWN'> = {
  0: 'UNKNOWN', 1: 'EXTREMEMENT_BAS', 2: 'TRES_BAS', 3: 'BAS',
  4: 'NORMAL', 5: 'HAUT', 6: 'TRES_HAUT', 7: 'EXTREMEMENT_HAUT',
}

// "En alerte" = the 3 driest classes.
export const METEO_CRITICAL_CLASSES: ReadonlySet<string> = new Set([
  'BAS', 'TRES_BAS', 'EXTREMEMENT_BAS',
])

export function isCriticalClass(cls: string | null | undefined): boolean {
  return cls != null && METEO_CRITICAL_CLASSES.has(cls)
}

// Severity order driest -> wettest for sorting (lower index = drier).
const _SEVERITY = ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT', 'UNKNOWN']
export function classSeverityRank(cls: string | null | undefined): number {
  const i = _SEVERITY.indexOf(cls ?? 'UNKNOWN')
  return i < 0 ? _SEVERITY.length : i
}

export interface AlertView { classEnum: string | null; trend: 'hausse' | 'stable' | 'baisse' | null }
export interface AlertSummary {
  dominantClass: string | null
  dominantClassLabel: string
  criticalCount: number
  withData: number
  trendSummary: string   // e.g. "majorité en baisse" / "stable" / "—"
}

/** Summarize a set of per-sector views for the national alert banner. */
export function summarizeAlert(views: AlertView[]): AlertSummary {
  const withDataViews = views.filter(v => v.classEnum && v.classEnum !== 'UNKNOWN')
  // dominant class = most frequent among withData
  const counts: Record<string, number> = {}
  for (const v of withDataViews) counts[v.classEnum as string] = (counts[v.classEnum as string] ?? 0) + 1
  let dominantClass: string | null = null, best = -1
  for (const [c, n] of Object.entries(counts)) if (n > best) { best = n; dominantClass = c }
  const criticalCount = withDataViews.filter(v => METEO_CRITICAL_CLASSES.has(v.classEnum as string)).length
  // trend summary
  const tc: Record<string, number> = { hausse: 0, stable: 0, baisse: 0 }
  for (const v of views) if (v.trend) tc[v.trend]++
  const topTrend = (Object.entries(tc).sort((a, b) => b[1] - a[1])[0] || ['', 0]) as [string, number]
  const trendSummary = topTrend[1] === 0 ? '—'
    : topTrend[0] === 'baisse' ? 'majorité en baisse'
    : topTrend[0] === 'hausse' ? 'majorité en hausse' : 'majoritairement stable'
  return {
    dominantClass,
    dominantClassLabel: dominantClass ? (METEO_CLASS_LABELS[dominantClass] ?? '') : '',
    criticalCount,
    withData: withDataViews.length,
    trendSummary,
  }
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
