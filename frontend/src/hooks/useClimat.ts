// Climat module hooks (Lot 2) — TanStack Query wrappers over observatoryApi.climat.*
// Pattern mirrors the ERA5 hooks in useObservatory.ts (24h staleTime — the backend
// itself caches the same window server-side via get_cached, so re-fetching sooner
// on the client would just re-hit a warm Redis cache for nothing).
import { useQuery } from '@tanstack/react-query'
import { observatoryApi } from '@/lib/observatory-api'

const CLIMAT_STALE_TIME = 24 * 60 * 60 * 1000

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
