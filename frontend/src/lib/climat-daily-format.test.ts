import { describe, it, expect } from 'vitest'
import { buildDailyBannerData, formatOneDecimal, HEAT_CELL_THRESHOLD_C } from './climat-daily-format'
import { PRECIP_DAILY_BOUNDS } from './climat-colors'
import type { ClimatDailyTempPoint } from './observatory-types'

function points(values: Array<number | null>): ClimatDailyTempPoint[] {
  return values.map((value, i) => ({ latitude: 45 + i * 0.1, longitude: 2, value }))
}

describe('formatOneDecimal', () => {
  it('formats with a comma decimal separator in French', () => {
    expect(formatOneDecimal(43.2, 'fr')).toBe('43,2')
  })

  it('formats with a period decimal separator in English', () => {
    expect(formatOneDecimal(43.2, 'en')).toBe('43.2')
  })

  it('always shows exactly one decimal', () => {
    expect(formatOneDecimal(43, 'fr')).toBe('43,0')
  })
})

describe('HEAT_CELL_THRESHOLD_C', () => {
  it('is the classic 35°C canicule marker', () => {
    expect(HEAT_CELL_THRESHOLD_C).toBe(35)
  })
})

describe('buildDailyBannerData', () => {
  it('computes the max value and the count of cells strictly above 35°C', () => {
    const data = buildDailyBannerData(points([32.1, 43.2, 35.0, 36.5, 28.9]), 'fr')
    expect(data).not.toBeNull()
    expect(data!.maxValueLabel).toBe('43,2')
    // 43.2 and 36.5 are > 35; 35.0 exactly is NOT counted.
    expect(data!.countAboveThreshold).toBe(2)
  })

  it('drops null cells before computing the max/count', () => {
    const data = buildDailyBannerData(points([null, 40.0, null]), 'fr')
    expect(data!.maxValueLabel).toBe('40,0')
    expect(data!.countAboveThreshold).toBe(1)
  })

  it('returns null when every cell is null (no data yet for this day)', () => {
    expect(buildDailyBannerData(points([null, null]), 'fr')).toBeNull()
  })

  it('returns null for an empty grid', () => {
    expect(buildDailyBannerData([], 'fr')).toBeNull()
  })

  it('locale-formats the max value in English', () => {
    const data = buildDailyBannerData(points([12.3, 43.2]), 'en')
    expect(data!.maxValueLabel).toBe('43.2')
  })

  it('counts zero cells above threshold when none qualify', () => {
    const data = buildDailyBannerData(points([10.0, 20.0, 34.9]), 'fr')
    expect(data!.countAboveThreshold).toBe(0)
  })
})

describe('buildDailyBannerData — pluie', () => {
  const TOP = PRECIP_DAILY_BOUNDS[PRECIP_DAILY_BOUNDS.length - 1] // 50

  it('compte les mailles au-dessus de la borne HAUTE de la légende', () => {
    // Le seuil n'est pas inventé : c'est la dernière borne de la légende, donc
    // le chiffre se vérifie à l'œil sur celle-ci (même principe que le %
    // sécheresse recalé sur une frontière de classe).
    const pts = [{ latitude: 1, longitude: 1, value: 98.8 }, { latitude: 1, longitude: 2, value: 50 },
                 { latitude: 1, longitude: 3, value: 49.9 }, { latitude: 1, longitude: 4, value: null }]
    const d = buildDailyBannerData(pts as any, 'fr', (v) => v >= TOP)
    expect(d?.maxValueLabel).toBe('98,8')
    expect(d?.countAboveThreshold).toBe(2) // 98.8 et 50 (>= inclusif, comme la classe « ≥ 50 »)
  })

  it('rend null quand aucune maille n’a de valeur', () => {
    const d = buildDailyBannerData([{ latitude: 1, longitude: 1, value: null }] as any, 'fr', () => true)
    expect(d).toBeNull()
  })
})
