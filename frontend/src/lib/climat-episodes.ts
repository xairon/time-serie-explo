// Pure helpers for the Point-panel drought episodes table (Task B2, EpisodesTable.tsx).
// Kept separate from the component so the logic is unit-testable without rendering
// (mirrors the climat-situation-format.ts pattern used by SituationBanner).
import type { ClimatDroughtEpisode } from './observatory-types'

export type EpisodeSortKey = 'debut' | 'duree_mois'
export type SortDirection = 'asc' | 'desc'

/** Sort a *copy* of the episodes list by start date or duration. Ties on duration
 *  break by start date (ascending) so the table order stays deterministic. */
export function sortEpisodes(
  episodes: ClimatDroughtEpisode[],
  key: EpisodeSortKey,
  direction: SortDirection,
): ClimatDroughtEpisode[] {
  const sign = direction === 'asc' ? 1 : -1
  return [...episodes].sort((a, b) => {
    if (key === 'duree_mois') {
      if (a.duree_mois !== b.duree_mois) return (a.duree_mois - b.duree_mois) * sign
      return a.debut.localeCompare(b.debut) // tie-break stays ascending regardless of direction
    }
    return a.debut.localeCompare(b.debut) * sign
  })
}

/** The episode still in progress, if any: its `fin` must be the series' last
 *  available month AND that month's SPI must be below the -1 drought threshold
 *  (plan B2: "l'épisode en cours (si spi<−1 au dernier mois) mis en évidence").
 *  Returns undefined when the last month isn't in drought, or matches no episode
 *  (e.g. the episodes list was fetched at a different window than lastMonthSpi). */
export function findCurrentEpisode(
  episodes: ClimatDroughtEpisode[],
  lastMonth: string | null | undefined,
  lastMonthSpi: number | null | undefined,
): ClimatDroughtEpisode | undefined {
  if (lastMonth == null || lastMonthSpi == null || lastMonthSpi >= -1) return undefined
  return episodes.find((e) => e.fin === lastMonth)
}
