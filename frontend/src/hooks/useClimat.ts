// Climat module hooks (Lot 2) — TanStack Query wrappers over observatoryApi.climat.*
// Pattern mirrors the ERA5 hooks in useObservatory.ts (24h staleTime — the backend
// itself caches the same window server-side via get_cached, so re-fetching sooner
// on the client would just re-hit a warm Redis cache for nothing).
import { useCallback, useMemo } from 'react'
import { useQuery, useQueries } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { observatoryApi } from '@/lib/observatory-api'
import { MIN_COMPARE_YEARS } from '@/lib/climat-year-select'

const CLIMAT_STALE_TIME = 24 * 60 * 60 * 1000

// Matches the backend's DAILY_TEMP_RANGE_TTL (1h) — the daily-temp date window
// moves every night (J-7 ingestion), unlike the monthly /range which only moves
// once a month, so re-checking sooner than the server cache actually matters here.
const DAILY_TEMP_RANGE_STALE_TIME = 60 * 60 * 1000

/** Fixed window used for the Point-panel drought episodes table (Task B2) — the
 *  standard 3-month "meteorological drought" window, matching the backend default. */
export const EPISODES_WINDOW = 3

/** Month bounds for the Climat page — see ClimatRange for why the indices max
 *  and the raw-monthly max are reported separately (the old default month came
 *  from /era5/range, the daily grid, which lands on the partial current month
 *  and breaks SPI/STI: no indices are computed for it yet). */
export function useClimatRange(enabled = true) {
  return useQuery({
    queryKey: ['climat', 'range'],
    queryFn: () => observatoryApi.climat.range(),
    enabled,
    staleTime: CLIMAT_STALE_TIME,
  })
}

/** Per-cell monthly variable (temperature/precipitation/etp/bilan_hydrique) for one month. */
export function useClimatGridMonthly(month: string | undefined, variable: string, enabled: boolean) {
  return useQuery({
    queryKey: ['climat', 'grid-monthly', month, variable],
    queryFn: () => observatoryApi.climat.gridMonthly(month!, variable),
    enabled: enabled && !!month,
    staleTime: CLIMAT_STALE_TIME,
  })
}

/** Per-cell SPI or STI for one month/window. */
export function useClimatGridIndices(
  month: string | undefined,
  window: number,
  index: 'spi' | 'sti',
  enabled: boolean,
) {
  return useQuery({
    queryKey: ['climat', 'grid-indices', month, window, index],
    queryFn: () => observatoryApi.climat.gridIndices(month!, window, index),
    enabled: enabled && !!month,
    staleTime: CLIMAT_STALE_TIME,
  })
}

/** Date bounds for the daily-temperature layer (Tx/Tn/Tmoy) — coverage is
 *  partial (nightly J-7 ingestion + an independent 1950-2025 backfill), so the
 *  DayStepper bounds and default day always come from here, never assumed. */
export function useClimatDailyTempRange(enabled = true) {
  return useQuery({
    queryKey: ['climat', 'daily-temp-range'],
    queryFn: () => observatoryApi.climat.dailyTempRange(),
    enabled,
    staleTime: DAILY_TEMP_RANGE_STALE_TIME,
  })
}

/** Per-cell TRUE daily temperature statistic (Tx/Tn/Tmoy, 24 hourly steps) for one date. */
export function useClimatDailyTemp(
  day: string | undefined,
  variable: 'tmax' | 'tmin' | 'tmean',
  enabled: boolean,
) {
  return useQuery({
    queryKey: ['climat', 'daily-temp', day, variable],
    queryFn: () => observatoryApi.climat.dailyTemp(day!, variable),
    enabled: enabled && !!day,
    staleTime: CLIMAT_STALE_TIME,
  })
}

/** Territory-wide synthesis (7-class breakdown, % drought, driest-since-year, top-5 driest cells). */
export function useClimatSituationSummary(month: string | undefined, window: number, enabled: boolean) {
  return useQuery({
    queryKey: ['climat', 'situation-summary', month, window],
    queryFn: () => observatoryApi.climat.situationSummary(month!, window),
    enabled: enabled && !!month,
    staleTime: CLIMAT_STALE_TIME,
  })
}

export interface SelectedCell { lat: number; lon: number }

/** Selected-cell state for the Point/Zone panel (Task B2), round-tripped through the
 *  ?lat&lon URL query params so the view stays shareable — same lat/lon convention as
 *  ObservatoryPage's fly-to params, but kept live in the URL (not consumed once) so
 *  a direct link opens the panel already populated. Coordinates are rounded to 2
 *  decimals (finer than the 0.1° grid) — plenty of precision, short URLs. */
export function useSelectedCellParam() {
  const [searchParams, setSearchParams] = useSearchParams()
  const latParam = searchParams.get('lat')
  const lonParam = searchParams.get('lon')

  const selectedCell = useMemo<SelectedCell | null>(() => {
    if (latParam == null || lonParam == null) return null
    const lat = Number(latParam)
    const lon = Number(lonParam)
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null
    return { lat, lon }
  }, [latParam, lonParam])

  const selectCell = useCallback((lat: number, lon: number) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev)
      next.set('lat', lat.toFixed(2))
      next.set('lon', lon.toFixed(2))
      return next
    })
  }, [setSearchParams])

  const clearSelectedCell = useCallback(() => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev)
      next.delete('lat')
      next.delete('lon')
      return next
    })
  }, [setSearchParams])

  return { selectedCell, selectCell, clearSelectedCell }
}

/** Full monthly series (1950→présent) for the grid cell nearest lat/lon: monthly
 *  variables + calendar-month normal + SPI/STI for the 4 standard windows. */
export function useClimatPointSeries(lat: number | undefined, lon: number | undefined) {
  return useQuery({
    queryKey: ['climat', 'point-series', lat, lon],
    queryFn: () => observatoryApi.climat.pointSeries(lat!, lon!),
    enabled: lat != null && lon != null,
    staleTime: CLIMAT_STALE_TIME,
  })
}

/** Drought episodes (consecutive calendar months with SPI < -1) for the grid cell
 *  nearest lat/lon. `window` defaults to EPISODES_WINDOW (3 months) but threads the
 *  SPI/STI chart's window selector (ClimatIndexChart, Task B2/C3) so the episodes
 *  table follows whichever window the user is looking at — the backend endpoint
 *  accepts &window= for any of the 4 standard windows. */
export function useClimatPointEpisodes(lat: number | undefined, lon: number | undefined, window: number = EPISODES_WINDOW) {
  return useQuery({
    queryKey: ['climat', 'point-episodes', lat, lon, window],
    queryFn: () => observatoryApi.climat.pointEpisodes(lat!, lon!, window),
    enabled: lat != null && lon != null,
    staleTime: CLIMAT_STALE_TIME,
  })
}

/** Fixed SPI window used for the Comparaison mini-maps (Task B3) — matches the
 *  3-month series already returned inline by compare-years, so the mini-maps and
 *  the cumulative chart read the same drought signal. */
export const COMPARE_MINIMAP_WINDOW = 3

/** Comparaison view (Task B3): per-year cumulative rainfall (vs. normale) + 3-month
 *  SPI series for the grid cell nearest lat/lon. Gated on a valid 2-6 year selection —
 *  fetching for a single year (or none) would be a wasted round-trip the UI never
 *  renders anything meaningful for. */
export function useClimatCompareYears(lat: number | undefined, lon: number | undefined, years: number[]) {
  const sortedKey = useMemo(() => years.slice().sort((a, b) => a - b).join(','), [years])
  return useQuery({
    queryKey: ['climat', 'compare-years', lat, lon, sortedKey],
    queryFn: () => observatoryApi.climat.compareYears(lat!, lon!, years),
    enabled: lat != null && lon != null && years.length >= MIN_COMPARE_YEARS,
    staleTime: CLIMAT_STALE_TIME,
  })
}

/** Petits multiples SPI (Task B3): one grid-indices fetch per selected year for a
 *  single chosen calendar month (e.g. June across 1976/1989/2003/2022) — same query
 *  key shape as useClimatGridIndices, so a mini-map reuses the Situation map's cache
 *  when the two happen to target the same month/window. Only fetches the selected
 *  years (TanStack Query keys the cache per year, nothing else is refetched when the
 *  selection grows). */
export function useClimatCompareGridIndices(years: number[], calendarMonth: number, enabled: boolean) {
  const window = COMPARE_MINIMAP_WINDOW
  return useQueries({
    queries: years.map((year) => {
      const month = `${year}-${String(calendarMonth).padStart(2, '0')}`
      return {
        queryKey: ['climat', 'grid-indices', month, window, 'spi' as const],
        queryFn: () => observatoryApi.climat.gridIndices(month, window, 'spi'),
        enabled: enabled && years.length > 0,
        staleTime: CLIMAT_STALE_TIME,
      }
    }),
  })
}
