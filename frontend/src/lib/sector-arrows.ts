import type { SituationClass, Trend } from '@/lib/observatory-types'
import { CLASSIFICATION_COLORS } from '@/lib/observatory-constants'

export const SECTOR_INSUFFICIENT_COLOR = '#d9d9d9'

/** BRGM tendancy_coord is "lat lon"; MapLibre wants [lon, lat]. */
export function parseTendancyCoord(raw: string | null | undefined): [number, number] | null {
  if (!raw) return null
  const parts = raw.trim().split(/\s+/).map(Number)
  if (parts.length !== 2 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1])) return null
  return [parts[1], parts[0]]
}

export function trendArrowGlyph(trend: Trend | null | undefined): string {
  if (trend === 'hausse') return '↑'
  if (trend === 'baisse') return '↓'
  if (trend === 'stable') return '→'
  return ''
}

export function sectorClassColor(cls: SituationClass | null): string {
  return cls ? (CLASSIFICATION_COLORS[cls] ?? SECTOR_INSUFFICIENT_COLOR) : SECTOR_INSUFFICIENT_COLOR
}
