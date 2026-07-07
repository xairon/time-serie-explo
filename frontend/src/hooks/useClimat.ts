// Climat module hooks (Lot 2) — TanStack Query wrappers over observatoryApi.climat.*
// Pattern mirrors the ERA5 hooks in useObservatory.ts (24h staleTime — the backend
// itself caches the same window server-side via get_cached, so re-fetching sooner
// on the client would just re-hit a warm Redis cache for nothing).
import { useCallback, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { observatoryApi } from '@/lib/observatory-api'

const CLIMAT_STALE_TIME = 24 * 60 * 60 * 1000

/** Fixed window used for the Point-panel drought episodes table (Task B2) — the
 *  standard 3-month "meteorological drought" window, matching the backend default. */
export const EPISODES_WINDOW = 3

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
 *  nearest lat/lon, at the fixed EPISODES_WINDOW. */
export function useClimatPointEpisodes(lat: number | undefined, lon: number | undefined) {
  return useQuery({
    queryKey: ['climat', 'point-episodes', lat, lon, EPISODES_WINDOW],
    queryFn: () => observatoryApi.climat.pointEpisodes(lat!, lon!, EPISODES_WINDOW),
    enabled: lat != null && lon != null,
    staleTime: CLIMAT_STALE_TIME,
  })
}
