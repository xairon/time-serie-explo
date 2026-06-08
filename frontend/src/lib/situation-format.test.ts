import { describe, it, expect } from 'vitest'
import { classColor, trendGlyph, INSUFFICIENT_COLOR } from './situation-format'

describe('situation-format', () => {
  it('maps a class to its palette color', () => {
    expect(classColor('BAS')).toBe('#f97316')
    expect(classColor('NORMAL')).toBe('#10b981')
  })
  it('falls back to insufficient grey for null', () => {
    expect(classColor(null)).toBe(INSUFFICIENT_COLOR)
  })
  it('maps trend to an arrow glyph', () => {
    expect(trendGlyph('hausse')).toBe('▲')
    expect(trendGlyph('baisse')).toBe('▼')
    expect(trendGlyph('stable')).toBe('▬')
    expect(trendGlyph(null)).toBe('—')
  })
})
