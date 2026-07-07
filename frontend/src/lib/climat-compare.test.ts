import { describe, it, expect } from 'vitest'
import { buildCumulSeries, buildNormalSeries } from './climat-compare'
import type { ClimatCompareYears } from './observatory-types'

const SAMPLE: ClimatCompareYears = {
  cell: { latitude: 47.4, longitude: 0.7 },
  years: {
    '1976': {
      cumul_mensuel: [
        { mois: 1, precipitation: 30, cumul: 30, cumul_normal: 40 },
        { mois: 2, precipitation: 20, cumul: 50, cumul_normal: 75 },
        { mois: 3, precipitation: 10, cumul: 60, cumul_normal: 110 },
      ],
      spi_3: [{ mois: 3, spi: -1.9 }],
    },
    '2003': {
      cumul_mensuel: [
        { mois: 1, precipitation: 25, cumul: 25, cumul_normal: 40 },
        { mois: 2, precipitation: 15, cumul: 40, cumul_normal: 75 },
      ],
      spi_3: [],
    },
    // 2026: no data yet (partial/current year) — absent from `years`.
  },
}

describe('buildCumulSeries', () => {
  it('returns one series per requested year, with fixed x = months 1..12', () => {
    const series = buildCumulSeries(SAMPLE, [1976, 2003])
    expect(series).toHaveLength(2)
    expect(series[0].year).toBe(1976)
    expect(series[0].x).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  })

  it('fills known months with the cumulative value', () => {
    const series = buildCumulSeries(SAMPLE, [1976])
    expect(series[0].y.slice(0, 3)).toEqual([30, 50, 60])
  })

  it('fills missing months with null (not 0, so the line breaks instead of dropping to zero)', () => {
    const series = buildCumulSeries(SAMPLE, [1976])
    expect(series[0].y.slice(3)).toEqual(Array(9).fill(null))
  })

  it('returns a null-filled series for a year absent from the response (e.g. still in progress)', () => {
    const series = buildCumulSeries(SAMPLE, [2026])
    expect(series).toHaveLength(1)
    expect(series[0].year).toBe(2026)
    expect(series[0].y.every((v) => v === null)).toBe(true)
  })

  it('returns an empty array when there is no data at all', () => {
    expect(buildCumulSeries(undefined, [1976, 2003])).toEqual([])
  })

  it('preserves the order of the requested years, independent of response key order', () => {
    const series = buildCumulSeries(SAMPLE, [2003, 1976])
    expect(series.map((s) => s.year)).toEqual([2003, 1976])
  })
})

describe('buildNormalSeries', () => {
  it('reads the normal off the year with the most complete monthly coverage', () => {
    // 1976 has 3 months, 2003 has 2 — 1976 should win.
    const normal = buildNormalSeries(SAMPLE, [1976, 2003])
    expect(normal.y.slice(0, 3)).toEqual([40, 75, 110])
  })

  it('falls back to null-filled when no selected year has data', () => {
    const normal = buildNormalSeries(SAMPLE, [2026])
    expect(normal.y.every((v) => v === null)).toBe(true)
  })

  it('returns null-filled when there is no data at all', () => {
    const normal = buildNormalSeries(undefined, [1976])
    expect(normal.x).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    expect(normal.y.every((v) => v === null)).toBe(true)
  })
})
