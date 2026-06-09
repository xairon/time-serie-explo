import { describe, it, expect } from 'vitest'
import { METEO_CLASS_COLORS, METEO_CLASS_LABELS, METEO_TREND_LABELS, meteoSectorColorPairs } from './meteo-colors'
import type { SectorSituation } from './observatory-types'

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
  it('meteoSectorColorPairs flattens to [sector_id, hex, ...] and greys insufficient/null', () => {
    const sits = [
      { code: '8', situation_class: 'NORMAL', insufficient: false },
      { code: '12', situation_class: null, insufficient: true },
    ] as unknown as SectorSituation[]
    const pairs = meteoSectorColorPairs(sits)
    expect(pairs).toEqual([8, '#ffde1a', 12, '#d9d9d9'])
  })
})
