import { describe, it, expect } from 'vitest'
import { buildDailyTempBannerData, formatTemperature, HEAT_CELL_THRESHOLD_C } from './climat-daily-temp-format'
import type { ClimatDailyTempPoint } from './observatory-types'

function points(values: Array<number | null>): ClimatDailyTempPoint[] {
  return values.map((value, i) => ({ latitude: 45 + i * 0.1, longitude: 2, value }))
}

describe('formatTemperature', () => {
  it('formats with a comma decimal separator in French', () => {
    expect(formatTemperature(43.2, 'fr')).toBe('43,2')
  })

  it('formats with a period decimal separator in English', () => {
    expect(formatTemperature(43.2, 'en')).toBe('43.2')
  })

  it('always shows exactly one decimal', () => {
    expect(formatTemperature(43, 'fr')).toBe('43,0')
  })
})

describe('HEAT_CELL_THRESHOLD_C', () => {
  it('is the classic 35°C canicule marker', () => {
    expect(HEAT_CELL_THRESHOLD_C).toBe(35)
  })
})

describe('buildDailyTempBannerData', () => {
  it('computes the max value and the count of cells strictly above 35°C', () => {
    const data = buildDailyTempBannerData(points([32.1, 43.2, 35.0, 36.5, 28.9]), 'fr')
    expect(data).not.toBeNull()
    expect(data!.maxValueLabel).toBe('43,2')
    // 43.2 and 36.5 are > 35; 35.0 exactly is NOT counted.
    expect(data!.countAboveThreshold).toBe(2)
  })

  it('drops null cells before computing the max/count', () => {
    const data = buildDailyTempBannerData(points([null, 40.0, null]), 'fr')
    expect(data!.maxValueLabel).toBe('40,0')
    expect(data!.countAboveThreshold).toBe(1)
  })

  it('returns null when every cell is null (no data yet for this day)', () => {
    expect(buildDailyTempBannerData(points([null, null]), 'fr')).toBeNull()
  })

  it('returns null for an empty grid', () => {
    expect(buildDailyTempBannerData([], 'fr')).toBeNull()
  })

  it('locale-formats the max value in English', () => {
    const data = buildDailyTempBannerData(points([12.3, 43.2]), 'en')
    expect(data!.maxValueLabel).toBe('43.2')
  })

  it('counts zero cells above threshold when none qualify', () => {
    const data = buildDailyTempBannerData(points([10.0, 20.0, 34.9]), 'fr')
    expect(data!.countAboveThreshold).toBe(0)
  })
})
