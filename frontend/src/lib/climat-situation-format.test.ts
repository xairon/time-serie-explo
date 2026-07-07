import { describe, it, expect } from 'vitest'
import { formatLatLon, formatDroughtPct, buildSituationBannerData } from './climat-situation-format'
import type { ClimatSituationSummary } from './observatory-types'

describe('formatLatLon', () => {
  it('formats a coordinate to one decimal with N/E suffixes (east, default fr locale)', () => {
    expect(formatLatLon(48.2345, 1.6789)).toBe('48.2°N, 1.7°E')
  })
  it('renders a west longitude with the French cardinal "O" (not a bare minus)', () => {
    expect(formatLatLon(45.0, -1.234)).toBe('45.0°N, 1.2°O')
  })
  it('renders a west longitude with the English cardinal "W"', () => {
    expect(formatLatLon(45.0, -1.234, 'en')).toBe('45.0°N, 1.2°W')
  })
  it('keeps "E" for an east longitude in English too', () => {
    expect(formatLatLon(48.2345, 1.6789, 'en')).toBe('48.2°N, 1.7°E')
  })
})

describe('formatDroughtPct', () => {
  it('drops a trailing .0', () => {
    expect(formatDroughtPct(42.0)).toBe('42')
  })
  it('keeps one decimal otherwise', () => {
    expect(formatDroughtPct(42.37)).toBe('42.4')
  })
  it('handles 0%', () => {
    expect(formatDroughtPct(0)).toBe('0')
  })
})

function makeSummary(overrides: Partial<ClimatSituationSummary> = {}): ClimatSituationSummary {
  return {
    month: '2026-07-01',
    window: 3,
    n_cells: 11496,
    classes_pct: { EXTREMEMENT_BAS: 5, TRES_BAS: 10, BAS: 15, NORMAL: 40, HAUT: 15, TRES_HAUT: 10, EXTREMEMENT_HAUT: 5 },
    pct_secheresse: 32.5,
    median_spi: -0.8,
    driest_since_year: 2003,
    is_driest_on_record: false,
    top5_cellules_seches: [
      { latitude: 43.6, longitude: 3.9, spi: -2.1 },
      { latitude: 44.1, longitude: 4.2, spi: -2.0 },
    ],
    available: true,
    ...overrides,
  }
}

describe('buildSituationBannerData', () => {
  it('derives display-ready fields from the raw summary', () => {
    const data = buildSituationBannerData(makeSummary())
    expect(data.pctSecheresse).toBe('32.5')
    expect(data.driestSinceYear).toBe(2003)
    expect(data.chips).toEqual([
      { label: '43.6°N, 3.9°E', latitude: 43.6, longitude: 3.9 },
      { label: '44.1°N, 4.2°E', latitude: 44.1, longitude: 4.2 },
    ])
  })

  it('surfaces a null driestSinceYear untouched (no historical comparison data)', () => {
    const data = buildSituationBannerData(makeSummary({ driest_since_year: null }))
    expect(data.driestSinceYear).toBeNull()
  })

  it('returns an empty chip list when the territory has no dry cells', () => {
    const data = buildSituationBannerData(makeSummary({ top5_cellules_seches: [] }))
    expect(data.chips).toEqual([])
  })

  it('formats a west-longitude chip with the locale cardinal (Bretagne, fr default)', () => {
    const data = buildSituationBannerData(
      makeSummary({ top5_cellules_seches: [{ latitude: 48.1, longitude: -2.8, spi: -2.3 }] }),
    )
    expect(data.chips).toEqual([{ label: '48.1°N, 2.8°O', latitude: 48.1, longitude: -2.8 }])
  })

  it('formats a west-longitude chip with the English cardinal when locale=en', () => {
    const data = buildSituationBannerData(
      makeSummary({ top5_cellules_seches: [{ latitude: 48.1, longitude: -2.8, spi: -2.3 }] }),
      'en',
    )
    expect(data.chips).toEqual([{ label: '48.1°N, 2.8°W', latitude: 48.1, longitude: -2.8 }])
  })
})
