// Observatory hooks — combined from junondashboard hooks
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { useMemo, useCallback, useEffect, useRef } from 'react'
import { observatoryApi } from '@/lib/observatory-api'
import { situationApi } from '@/lib/situation-api'
import type {
  SPIDataPoint, SPLIDataPoint, SSFIDataPoint, WfsLayerId,
  ObsPastasSummary, ObsPastasTimeseriesPoint, ObsPastasSGIPoint, ObsPastasCoverage,
} from '@/lib/observatory-types'

// --- Station hooks ---

export function usePiezoStations(filters?: Record<string, string | string[] | undefined>) {
  return useQuery({
    queryKey: ['obs-stations', 'piezo', filters],
    queryFn: () => observatoryApi.piezo.stations(filters),
  })
}

export function useHydroStations(filters?: Record<string, string | string[] | undefined>) {
  return useQuery({
    queryKey: ['obs-stations', 'hydro', filters],
    queryFn: () => observatoryApi.hydro.stations(filters),
  })
}

export function usePiezoStationDetail(code: string) {
  return useQuery({
    queryKey: ['obs-station', 'piezo', code],
    queryFn: () => observatoryApi.piezo.detail(code),
    enabled: !!code,
  })
}

export function useHydroStationDetail(code: string) {
  return useQuery({
    queryKey: ['obs-station', 'hydro', code],
    queryFn: () => observatoryApi.hydro.detail(code),
    enabled: !!code,
  })
}

export function useStationsGeoJSON() {
  return useQuery({
    queryKey: ['obs-stations', 'geojson'],
    queryFn: () => observatoryApi.common.geojson(),
    staleTime: 3_600_000, // 1h
    gcTime: 3_600_000,
  })
}

export function usePiezoSiblings(code: string, level: 'nappe' | 'systeme') {
  return useQuery({
    queryKey: ['obs-siblings', 'piezo', code, level],
    queryFn: () => observatoryApi.piezo.siblings(code, level),
    enabled: !!code,
    staleTime: 3_600_000,
    retry: false,
  })
}

export function useHydroSiblings(code: string, level: 'site' | 'cours_eau' = 'site') {
  return useQuery({
    queryKey: ['obs-siblings', 'hydro', code, level],
    queryFn: () => observatoryApi.hydro.siblings(code, level),
    enabled: !!code,
    staleTime: 3_600_000,
    retry: false,
  })
}

// --- Timeseries hooks ---

export function usePiezoMonthly(code: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ['obs-timeseries', 'piezo', 'monthly', code],
    queryFn: () => observatoryApi.piezo.monthly(code),
    enabled: !!code && (options?.enabled ?? true),
    staleTime: 12 * 60 * 60 * 1000, // 12h
  })
}

export function useHydroMonthly(code: string, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ['obs-timeseries', 'hydro', 'monthly', code],
    queryFn: () => observatoryApi.hydro.monthly(code),
    enabled: !!code && (options?.enabled ?? true),
    staleTime: 12 * 60 * 60 * 1000,
  })
}

export function usePiezoYearly(code: string) {
  return useQuery({
    queryKey: ['obs-timeseries', 'piezo', 'yearly', code],
    queryFn: () => observatoryApi.piezo.yearly(code),
    enabled: !!code,
    staleTime: 24 * 60 * 60 * 1000,
  })
}

export function useHydroYearly(code: string) {
  return useQuery({
    queryKey: ['obs-timeseries', 'hydro', 'yearly', code],
    queryFn: () => observatoryApi.hydro.yearly(code),
    enabled: !!code,
    staleTime: 24 * 60 * 60 * 1000,
  })
}

export function usePiezoDaily(code: string, startDate?: string, endDate?: string, limit?: number) {
  return useQuery({
    queryKey: ['obs-timeseries', 'piezo', 'daily', code, startDate, endDate, limit],
    queryFn: () => observatoryApi.piezo.daily(code, {
      start_date: startDate,
      end_date: endDate,
      ...(limit ? { limit: String(limit) } : {}),
    }),
    enabled: !!code,
    staleTime: 6 * 60 * 60 * 1000,
  })
}

export function useHydroDaily(code: string, startDate?: string, endDate?: string, limit?: number) {
  return useQuery({
    queryKey: ['obs-timeseries', 'hydro', 'daily', code, startDate, endDate, limit],
    queryFn: () => observatoryApi.hydro.daily(code, {
      start_date: startDate,
      end_date: endDate,
      ...(limit ? { limit: String(limit) } : {}),
    }),
    enabled: !!code,
    staleTime: 6 * 60 * 60 * 1000,
  })
}

export function usePiezoSPLI(code: string) {
  return useQuery<SPLIDataPoint[]>({
    queryKey: ['obs-drought', 'piezo', 'spli', code],
    queryFn: () => observatoryApi.piezo.spli(code),
    enabled: !!code,
    staleTime: 24 * 60 * 60 * 1000,
  })
}

export function useHydroSSFI(code: string) {
  return useQuery<SSFIDataPoint[]>({
    queryKey: ['obs-drought', 'hydro', 'ssfi', code],
    queryFn: () => observatoryApi.hydro.ssfi(code),
    enabled: !!code,
    staleTime: 24 * 60 * 60 * 1000,
  })
}

export function useSPI(code: string, type: 'piezo' | 'hydro') {
  return useQuery<SPIDataPoint[]>({
    queryKey: ['obs-drought', type, 'spi', code],
    queryFn: () => type === 'piezo' ? observatoryApi.piezo.spi(code) : observatoryApi.hydro.spi(code),
    enabled: !!code,
    staleTime: 24 * 60 * 60 * 1000,
  })
}

// --- Pastas hooks (observatory-specific) ---

export function useObsPastasSummary(code: string) {
  return useQuery<ObsPastasSummary>({
    queryKey: ['obs-pastas', 'summary', code],
    queryFn: () => observatoryApi.pastas.summary(code),
    enabled: !!code,
    staleTime: 24 * 60 * 60 * 1000,
    retry: false,
  })
}

export function useObsPastasTimeseries(
  code: string,
  resolution: 'monthly' | 'daily' = 'monthly',
  start?: string,
  end?: string,
) {
  return useQuery<ObsPastasTimeseriesPoint[]>({
    queryKey: ['obs-pastas', 'timeseries', code, resolution, start, end],
    queryFn: () => observatoryApi.pastas.timeseries(code, { resolution, start, end }),
    enabled: !!code,
    staleTime: 6 * 60 * 60 * 1000,
    retry: false,
  })
}

export function useObsPastasSGI(code: string) {
  return useQuery<ObsPastasSGIPoint[]>({
    queryKey: ['obs-pastas', 'sgi', code],
    queryFn: () => observatoryApi.pastas.sgi(code),
    enabled: !!code,
    staleTime: 12 * 60 * 60 * 1000,
    retry: false,
  })
}

export function useObsPastasCoverage() {
  return useQuery<ObsPastasCoverage[]>({
    queryKey: ['obs-pastas', 'coverage'],
    queryFn: () => observatoryApi.pastas.coverage(),
    staleTime: 24 * 60 * 60 * 1000,
  })
}

// --- Filters hook ---

const STORAGE_KEY = 'obs_filters'

function saveToStorage(params: URLSearchParams) {
  const str = params.toString()
  if (str) sessionStorage.setItem(STORAGE_KEY, str)
  else sessionStorage.removeItem(STORAGE_KEY)
}

function restoreFromStorage(): URLSearchParams | null {
  const saved = sessionStorage.getItem(STORAGE_KEY)
  return saved ? new URLSearchParams(saved) : null
}

export interface ObsFilters {
  activeOnly?: boolean
  minObservations?: number
  lastMeasurementAfter?: string
  classification?: string[]
  codeDepartement?: string
  codeBdlisa?: string
  codeBassin?: string
  codeRegion?: string
  codeHer?: number
  showFiable?: boolean
  showIndicatif?: boolean
  showInsuffisant?: boolean
}

export function useObsFilters() {
  const [searchParams, setSearchParams] = useSearchParams()
  const restored = useRef(false)

  useEffect(() => {
    if (restored.current) return
    restored.current = true
    if (searchParams.toString() === '') {
      const saved = restoreFromStorage()
      if (saved && saved.toString() !== '') {
        setSearchParams(saved, { replace: true })
      }
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    saveToStorage(searchParams)
  }, [searchParams])

  const filters = useMemo<ObsFilters>(() => ({
    activeOnly: searchParams.get('active_only') === 'false' ? false : true,
    minObservations: searchParams.get('min_obs') ? Number(searchParams.get('min_obs')) : undefined,
    lastMeasurementAfter: searchParams.get('last_after') ?? undefined,
    classification: searchParams.getAll('classif').length > 0 ? searchParams.getAll('classif') : undefined,
    codeDepartement: searchParams.get('dept') ?? undefined,
    codeBdlisa: searchParams.get('bdlisa') ?? undefined,
    codeBassin: searchParams.get('bassin') ?? undefined,
    codeRegion: searchParams.get('region') ?? undefined,
    codeHer: searchParams.get('her') ? Number(searchParams.get('her')) : undefined,
    showFiable: searchParams.get('fiable') === 'false' ? false : true,
    showIndicatif: searchParams.get('indicatif') === 'true' ? true : false,
    showInsuffisant: searchParams.get('insuffisant') === 'true' ? true : false,
  }), [searchParams])

  const setFilter = useCallback((key: string, value: string | string[] | undefined) => {
    setSearchParams(prev => {
      const next = new URLSearchParams(prev)
      if (value === undefined) {
        next.delete(key)
      } else if (Array.isArray(value)) {
        next.delete(key)
        value.forEach(v => next.append(key, v))
      } else {
        next.set(key, value)
      }
      return next
    })
  }, [setSearchParams])

  const setFilters = useCallback((entries: Record<string, string | string[] | undefined>) => {
    setSearchParams(prev => {
      const next = new URLSearchParams(prev)
      for (const [key, value] of Object.entries(entries)) {
        if (value === undefined) {
          next.delete(key)
        } else if (Array.isArray(value)) {
          next.delete(key)
          value.forEach(v => next.append(key, v))
        } else {
          next.set(key, value)
        }
      }
      return next
    })
  }, [setSearchParams])

  const apiParams = useMemo(() => {
    const p: Record<string, string | string[] | undefined> = {}
    if (filters.minObservations) p.min_observations = String(filters.minObservations)
    if (filters.lastMeasurementAfter) p.last_measurement_after = filters.lastMeasurementAfter
    if (filters.classification) p.classification = filters.classification
    if (filters.codeDepartement) p.code_departement = filters.codeDepartement
    return p
  }, [filters])

  return { filters, setFilter, setFilters, apiParams }
}

// --- WFS Layer hook ---

export function useWfsLayer(layerId: WfsLayerId, enabled: boolean) {
  return useQuery({
    queryKey: ['obs-wfs', layerId],
    queryFn: () => observatoryApi.wfs.layer(layerId),
    enabled,
    staleTime: 24 * 60 * 60 * 1000,
    gcTime: 24 * 60 * 60 * 1000,
  })
}

// --- ERA5 hooks ---

export function useERA5Grid() {
  return useQuery({
    queryKey: ['obs-era5', 'grid'],
    queryFn: () => observatoryApi.era5.grid(),
    staleTime: 24 * 60 * 60 * 1000,
  })
}

export function useERA5Snapshot(date: string | undefined) {
  return useQuery({
    queryKey: ['obs-era5', 'snapshot', date],
    queryFn: () => observatoryApi.era5.snapshot(date!),
    enabled: !!date,
    staleTime: 24 * 60 * 60 * 1000,
  })
}

export function useERA5Dates() {
  return useQuery({
    queryKey: ['obs-era5', 'dates'],
    queryFn: () => observatoryApi.era5.dates(),
    staleTime: 24 * 60 * 60 * 1000,
  })
}

export function useERA5Monthly(month: string | undefined) {
  return useQuery({
    queryKey: ['obs-era5', 'monthly', month],
    queryFn: () => observatoryApi.era5.monthly(month!),
    enabled: !!month,
    staleTime: 24 * 60 * 60 * 1000,
  })
}

export function useERA5Range(enabled = true) {
  return useQuery({
    queryKey: ['obs-era5', 'range'],
    queryFn: () => observatoryApi.era5.range(),
    enabled,
    staleTime: 24 * 60 * 60 * 1000,
  })
}


export function useERA5Anomaly(variable: string, date: string | undefined, window: number, enabled: boolean) {
  return useQuery({
    queryKey: ['obs-era5', 'anomaly', variable, date, window],
    queryFn: () => observatoryApi.era5.anomaly(variable, date!, window),
    enabled: enabled && !!date,
    staleTime: 24 * 60 * 60 * 1000,
  })
}

export function useERA5Sti(date: string | undefined, window: number, enabled: boolean) {
  return useQuery({
    queryKey: ['obs-era5', 'sti', date, window],
    queryFn: () => observatoryApi.era5.sti(date!, window),
    enabled: enabled && !!date,
    staleTime: 24 * 60 * 60 * 1000,
  })
}

// --- Sector situation hooks ---

export function useSectorSituation(type: 'piezo' | 'hydro', enabled: boolean, network: 'all' | 'meteeau' = 'all') {
  return useQuery({ queryKey: ['obs-sectors', type, network], queryFn: () => situationApi.sectors(type, undefined, network), enabled, staleTime: 3_600_000 * 6 })
}

export function useSectorTimeline(type: 'piezo' | 'hydro', enabled: boolean, network: 'all' | 'meteeau' = 'all') {
  return useQuery({ queryKey: ['obs-sectors-timeline', type, network], queryFn: () => situationApi.sectorsTimeline(type, network), enabled, staleTime: 3_600_000 * 24 })
}

