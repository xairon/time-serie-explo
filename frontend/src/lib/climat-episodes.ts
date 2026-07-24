// Pure helpers for the Point-panel drought episodes table (Task B2, EpisodesTable.tsx).
// Kept separate from the component so the logic is unit-testable without rendering.
import type { ClimatDroughtEpisode, ClimatPointSeriesEntry } from './observatory-types'

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

/** Last series entry with a non-null `<index>_<window>` value — mirrors the
 *  backward-scan pattern of `latestSpiPoint` in climate-cumuls.ts, but keyed on
 *  whichever index (SPI or SPEI) and window the caller is currently displaying
 *  (1/3/6/12), since point-series entries carry one `spi_<n>` and one `spei_<n>`
 *  field per window rather than a single `spi`/`spei` field.
 *
 *  The series' last entry is normally the partial current month, for which
 *  no SPI/STI/SPEI has been computed yet (null) — reading `series[length-1]`
 *  directly for the "épisode en cours" highlight would then always miss it in
 *  production. */
export function findLastEntryWithIndex(
  series: ClimatPointSeriesEntry[],
  window: number,
  index: 'spi' | 'spei' = 'spi',
): ClimatPointSeriesEntry | undefined {
  const key = `${index}_${window}` as keyof ClimatPointSeriesEntry
  for (let i = series.length - 1; i >= 0; i--) {
    if (series[i][key] != null) return series[i]
  }
  return undefined
}
