import { describe, it, expect } from 'vitest'
import { parseTendancyCoord, trendArrowGlyph } from './sector-arrows'

describe('sector-arrows', () => {
  it('parses "lat lon" into [lon, lat]', () => {
    expect(parseTendancyCoord('50.13529085 3.04309184')).toEqual([3.04309184, 50.13529085])
  })
  it('returns null for invalid coord', () => {
    expect(parseTendancyCoord('')).toBeNull()
    expect(parseTendancyCoord('abc')).toBeNull()
  })
  it('maps trend to arrow glyph', () => {
    expect(trendArrowGlyph('hausse')).toBe('↑')
    expect(trendArrowGlyph('baisse')).toBe('↓')
    expect(trendArrowGlyph('stable')).toBe('→')
    expect(trendArrowGlyph(null)).toBe('')
  })
})
