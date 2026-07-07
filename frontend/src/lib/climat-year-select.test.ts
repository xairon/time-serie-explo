import { describe, it, expect } from 'vitest'
import {
  DROUGHT_YEAR_PRESETS,
  MAX_COMPARE_YEARS,
  MIN_COMPARE_YEARS,
  defaultCompareYears,
  isValidYearSelection,
  toggleYear,
} from './climat-year-select'

describe('climat-year-select', () => {
  describe('isValidYearSelection', () => {
    it('rejects fewer than MIN_COMPARE_YEARS', () => {
      expect(MIN_COMPARE_YEARS).toBe(2)
      expect(isValidYearSelection([2003])).toBe(false)
    })

    it('accepts exactly MIN_COMPARE_YEARS', () => {
      expect(isValidYearSelection([1976, 2003])).toBe(true)
    })

    it('accepts exactly MAX_COMPARE_YEARS', () => {
      expect(isValidYearSelection([1976, 1989, 2003, 2018, 2022, 2026])).toBe(true)
    })

    it('rejects more than MAX_COMPARE_YEARS', () => {
      expect(isValidYearSelection([1976, 1989, 2003, 2018, 2022, 2025, 2026])).toBe(false)
    })
  })

  describe('defaultCompareYears', () => {
    it('includes all drought presets plus the current year, sorted', () => {
      const years = defaultCompareYears(2026)
      expect(years).toEqual([1976, 1989, 2003, 2022, 2026])
    })

    it('deduplicates when the current year coincides with a preset', () => {
      const years = defaultCompareYears(2022)
      expect(years).toEqual([1976, 1989, 2003, 2022])
    })

    it('always returns a valid (2-6) selection', () => {
      expect(isValidYearSelection(defaultCompareYears(2026))).toBe(true)
      expect(isValidYearSelection(defaultCompareYears(1980))).toBe(true)
    })

    it('caps at MAX_COMPARE_YEARS', () => {
      // Pathological case: nothing coincides, but the base set already fits (4 presets + 1).
      const years = defaultCompareYears(1999)
      expect(years.length).toBeLessThanOrEqual(MAX_COMPARE_YEARS)
    })
  })

  describe('toggleYear', () => {
    it('adds a year not yet selected, keeping the list sorted', () => {
      const result = toggleYear([1976, 2003], 1989)
      expect(result).toEqual([1976, 1989, 2003])
    })

    it('removes a year already selected, when above MIN_COMPARE_YEARS', () => {
      const result = toggleYear([1976, 1989, 2003], 1989)
      expect(result).toEqual([1976, 2003])
    })

    it('refuses to remove below MIN_COMPARE_YEARS and returns the same reference', () => {
      const selected = [1976, 2003]
      const result = toggleYear(selected, 2003)
      expect(result).toBe(selected)
      expect(result).toEqual([1976, 2003])
    })

    it('refuses to add past MAX_COMPARE_YEARS and returns the same reference', () => {
      const selected = [1976, 1989, 2003, 2018, 2022, 2026]
      const result = toggleYear(selected, 1997)
      expect(result).toBe(selected)
      expect(result.length).toBe(MAX_COMPARE_YEARS)
    })

    it('is a no-op re-toggle round trip when within bounds', () => {
      const start = [1976, 1989, 2003]
      const added = toggleYear(start, 2022)
      const removed = toggleYear(added, 2022)
      expect(removed).toEqual(start)
    })
  })

  it('exposes the expected famous-drought presets', () => {
    expect(DROUGHT_YEAR_PRESETS).toEqual([1976, 1989, 2003, 2022])
  })
})
