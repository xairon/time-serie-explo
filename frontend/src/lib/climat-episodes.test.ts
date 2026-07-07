import { describe, it, expect } from 'vitest'
import { sortEpisodes, findCurrentEpisode } from './climat-episodes'
import type { ClimatDroughtEpisode } from './observatory-types'

const EPISODES: ClimatDroughtEpisode[] = [
  { debut: '1976-04-01', fin: '1976-08-01', duree_mois: 5, spi_min: -2.1, deficit_cumule_mm: -180.4 },
  { debut: '2003-06-01', fin: '2003-09-01', duree_mois: 4, spi_min: -1.9, deficit_cumule_mm: -120.0 },
  { debut: '2022-05-01', fin: '2022-09-01', duree_mois: 5, spi_min: -2.4, deficit_cumule_mm: -200.7 },
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
