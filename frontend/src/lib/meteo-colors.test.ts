import { describe, it, expect } from 'vitest'
import { METEO_CLASS_COLORS, METEO_CLASS_LABELS, METEO_TREND_LABELS, meteoSectorColorPairs, METEO_CRITICAL_CLASSES, isCriticalClass, classSeverityRank, summarizeAlert, BRGM_CLASS_TO_ENUM } from './meteo-colors'
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

describe('alert helpers', () => {
  it('critical = 3 driest classes', () => {
    expect(isCriticalClass('BAS')).toBe(true)
    expect(isCriticalClass('NORMAL')).toBe(false)
    expect(METEO_CRITICAL_CLASSES.size).toBe(3)
  })
  it('BRGM_CLASS_TO_ENUM maps 3->BAS, 4->NORMAL, 0->UNKNOWN', () => {
    expect(BRGM_CLASS_TO_ENUM[3]).toBe('BAS')
    expect(BRGM_CLASS_TO_ENUM[4]).toBe('NORMAL')
    expect(BRGM_CLASS_TO_ENUM[0]).toBe('UNKNOWN')
  })
  it('severity sorts driest first', () => {
    expect(classSeverityRank('EXTREMEMENT_BAS')).toBeLessThan(classSeverityRank('NORMAL'))
  })
  it('summarizeAlert counts critical, dominant, trend', () => {
    const s = summarizeAlert([
      { classEnum: 'BAS', trend: 'baisse' },
      { classEnum: 'BAS', trend: 'baisse' },
      { classEnum: 'NORMAL', trend: 'stable' },
      { classEnum: 'UNKNOWN', trend: null },
    ])
    expect(s.criticalCount).toBe(2)
    expect(s.dominantClass).toBe('BAS')
    expect(s.withData).toBe(3)
    expect(s.trendSummary).toBe('majorité en baisse')
  })
})
