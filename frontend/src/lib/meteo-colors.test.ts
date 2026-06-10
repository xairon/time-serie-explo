import { describe, it, expect } from 'vitest'
import { METEO_CLASS_COLORS, METEO_CLASS_LABELS, METEO_TREND_LABELS, meteoClassColor } from './meteo-colors'

describe('meteo-colors', () => {
  it('maps confirmed BRGM hexes', () => {
    expect(METEO_CLASS_COLORS.BAS).toBe('#f8930f')
    expect(METEO_CLASS_COLORS.NORMAL).toBe('#ffde1a')
    expect(METEO_CLASS_COLORS.HAUT).toBe('#60a3d6')
    expect(METEO_CLASS_COLORS.TRES_HAUT).toBe('#3071b0')
    expect(METEO_CLASS_COLORS.EXTREMEMENT_HAUT).toBe('#00408b')
    expect(METEO_CLASS_COLORS.UNKNOWN).toBe('#d9d9d9')
  })
  it('uses BRGM labels', () => {
    expect(METEO_CLASS_LABELS.NORMAL).toBe('autour de la moyenne')
    expect(METEO_CLASS_LABELS.BAS).toBe('modérément bas')
    expect(METEO_TREND_LABELS.hausse).toBe('en hausse')
  })
  it('meteoClassColor falls back to UNKNOWN hex for null/undefined', () => {
    expect(meteoClassColor(null)).toBe('#d9d9d9')
    expect(meteoClassColor(undefined)).toBe('#d9d9d9')
    expect(meteoClassColor('NORMAL')).toBe('#ffde1a')
  })
})
