import { describe, it, expect } from 'vitest'
import { sortEpisodes, findCurrentEpisode, findLastEntryWithSpi } from './climat-episodes'
import type { ClimatDroughtEpisode, ClimatPointSeriesEntry } from './observatory-types'

/** Minimal point-series entry — only the fields findLastEntryWithSpi reads. */
function entry(month: string, overrides: Partial<ClimatPointSeriesEntry> = {}): ClimatPointSeriesEntry {
  return {
    month,
    temperature_moyenne: null, temperature_min: null, temperature_max: null,
    precipitation_totale: null, etp_totale: null, bilan_hydrique: null, nb_jours: null,
    mois_complet: true, precipitation_normale: null, temperature_normale: null,
    spi_1: null, sti_1: null, spi_3: null, sti_3: null,
    spi_6: null, sti_6: null, spi_12: null, sti_12: null,
    ...overrides,
  }
}

const EPISODES: ClimatDroughtEpisode[] = [
  { debut: '1976-04-01', fin: '1976-08-01', duree_mois: 5, index_min: -2.1, deficit_cumule_mm: -180.4 },
  { debut: '2003-06-01', fin: '2003-09-01', duree_mois: 4, index_min: -1.9, deficit_cumule_mm: -120.0 },
  { debut: '2022-05-01', fin: '2022-09-01', duree_mois: 5, index_min: -2.4, deficit_cumule_mm: -200.7 },
]

describe('sortEpisodes', () => {
  it('sorts by duration descending, ties broken by start date ascending', () => {
    const sorted = sortEpisodes(EPISODES, 'duree_mois', 'desc')
    expect(sorted.map((e) => e.debut)).toEqual(['1976-04-01', '2022-05-01', '2003-06-01'])
  })

  it('sorts by duration ascending', () => {
    const sorted = sortEpisodes(EPISODES, 'duree_mois', 'asc')
    expect(sorted.map((e) => e.debut)).toEqual(['2003-06-01', '1976-04-01', '2022-05-01'])
  })

  it('sorts by start date descending', () => {
    const sorted = sortEpisodes(EPISODES, 'debut', 'desc')
    expect(sorted.map((e) => e.debut)).toEqual(['2022-05-01', '2003-06-01', '1976-04-01'])
  })

  it('sorts by start date ascending', () => {
    const sorted = sortEpisodes(EPISODES, 'debut', 'asc')
    expect(sorted.map((e) => e.debut)).toEqual(['1976-04-01', '2003-06-01', '2022-05-01'])
  })

  it('does not mutate the input array', () => {
    const copy = [...EPISODES]
    sortEpisodes(EPISODES, 'duree_mois', 'desc')
    expect(EPISODES).toEqual(copy)
  })
})

describe('findCurrentEpisode', () => {
  it('returns the episode ending on the last month when its SPI is below -1', () => {
    const current = findCurrentEpisode(EPISODES, '2022-09-01', -1.4)
    expect(current).toBe(EPISODES[2])
  })

  it('returns undefined when the last month is not in drought (spi >= -1)', () => {
    expect(findCurrentEpisode(EPISODES, '2022-09-01', -0.5)).toBeUndefined()
  })

  it('returns undefined when spi is exactly -1 (not strictly below the threshold)', () => {
    expect(findCurrentEpisode(EPISODES, '2022-09-01', -1)).toBeUndefined()
  })

  it('returns undefined when no episode ends on the last month', () => {
    expect(findCurrentEpisode(EPISODES, '2024-01-01', -1.5)).toBeUndefined()
  })

  it('returns undefined when lastMonth or lastMonthSpi is missing', () => {
    expect(findCurrentEpisode(EPISODES, undefined, -1.5)).toBeUndefined()
    expect(findCurrentEpisode(EPISODES, '2022-09-01', null)).toBeUndefined()
    expect(findCurrentEpisode(EPISODES, null, undefined)).toBeUndefined()
  })

  it('returns undefined for an empty episodes list', () => {
    expect(findCurrentEpisode([], '2022-09-01', -1.5)).toBeUndefined()
  })
})

describe('findLastEntryWithSpi', () => {
  it('returns the last entry when its spi_<window> is non-null', () => {
    const series = [entry('2026-04-01', { spi_3: -0.5 }), entry('2026-05-01', { spi_3: -1.6 })]
    expect(findLastEntryWithSpi(series, 3)?.month).toBe('2026-05-01')
  })

  it('scans backward past a trailing null spi_<window> (the partial current month)', () => {
    const series = [
      entry('2026-04-01', { spi_3: -1.8 }),
      entry('2026-05-01', { spi_3: -1.6 }),
      entry('2026-06-01', { spi_3: null }), // partial current month: no SPI yet
    ]
    expect(findLastEntryWithSpi(series, 3)?.month).toBe('2026-05-01')
  })

  it('reads the field matching the requested window, not a different one', () => {
    // spi_3 is null throughout (no 3-month episode yet); spi_6 has a real value.
    const series = [entry('2026-04-01', { spi_3: null, spi_6: -0.5 }), entry('2026-05-01', { spi_3: null, spi_6: -1.9 })]
    expect(findLastEntryWithSpi(series, 3)).toBeUndefined()
    expect(findLastEntryWithSpi(series, 6)?.month).toBe('2026-05-01')
  })

  it('returns undefined when every entry has a null spi_<window>', () => {
    const series = [entry('2026-04-01', { spi_3: null }), entry('2026-05-01', { spi_3: null })]
    expect(findLastEntryWithSpi(series, 3)).toBeUndefined()
  })

  it('returns undefined for an empty series', () => {
    expect(findLastEntryWithSpi([], 3)).toBeUndefined()
  })
})
