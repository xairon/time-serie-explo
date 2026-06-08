import { CLASSIFICATION_COLORS } from './observatory-constants'
import type { SituationClass, Trend } from './observatory-types'

export const INSUFFICIENT_COLOR = '#374151'  // muted slate for "données insuffisantes"

export function classColor(cls: SituationClass | null): string {
  if (!cls) return INSUFFICIENT_COLOR
  return CLASSIFICATION_COLORS[cls] ?? INSUFFICIENT_COLOR
}

export function trendGlyph(trend: Trend | null): string {
  switch (trend) {
    case 'hausse': return '▲'
    case 'baisse': return '▼'
    case 'stable': return '▬'
    default: return '—'
  }
}
