// Observatory API — ported from junondashboard, adapted for /api/v1/observatory/ prefix
import { API_BASE } from './constants'
import type {
  PiezoStation, HydroStation, NationalStats,
  Alert, ERA5GridPoint,
  DailyPiezoMeasurement, DailyHydroMeasurement,
  MonthlyPiezoData, MonthlyHydroData,
  YearlyPiezoData, YearlyHydroData,
  StationPercentiles,
  StationGeoJSON, ClassificationTimeline,
  SPIDataPoint, SPLIDataPoint, SSFIDataPoint,
  HydroSiteSiblings,
  PiezoBdlisaSiblings,
  ObsPastasSummary, ObsPastasTimeseriesPoint, ObsPastasSGIPoint, ObsPastasCoverage,
} from './observatory-types'

export async function fetchJson<T>(path: string, params?: Record<string, string | string[] | undefined>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`, window.location.origin)
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v === undefined) return
      if (Array.isArray(v)) {
        v.forEach(val => url.searchParams.append(k, val))
      } else {
        url.searchParams.set(k, v)
      }
    })
  }
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 30_000)
  try {
    const res = await fetch(url.toString(), { signal: controller.signal })
    if (!res.ok) {
      let detail = ''
      try {
        const body = await res.json()
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      } catch { /* ignore parse errors */ }
      throw new Error(`API ${res.status}${detail ? `: ${detail}` : ''}`)
    }
    return await res.json()
  } finally {
    clearTimeout(timeoutId)
  }
}

export const observatoryApi = {
  piezo: {
    stations: (params?: Record<string, string | string[] | undefined>) =>
      fetchJson<PiezoStation[]>('/observatory/piezo/stations', params),
    detail: (code: string) => fetchJson<PiezoStation>(`/observatory/piezo/stations/${code}`),
    percentiles: (code: string) =>
      fetchJson<StationPercentiles>(`/observatory/piezo/stations/${encodeURIComponent(code)}/percentiles`),
    daily: (code: string, params?: Record<string, string | undefined>) =>
      fetchJson<DailyPiezoMeasurement[]>(`/observatory/piezo/stations/${code}/daily`, params),
    monthly: (code: string) => fetchJson<MonthlyPiezoData[]>(`/observatory/piezo/stations/${code}/monthly`),
    yearly: (code: string) => fetchJson<YearlyPiezoData[]>(`/observatory/piezo/stations/${code}/yearly`),
    spli: (code: string) => fetchJson<SPLIDataPoint[]>(`/observatory/piezo/stations/${code}/spli`),
    spi: (code: string) => fetchJson<SPIDataPoint[]>(`/observatory/piezo/stations/${code}/spi`),
    siblings: (code: string, level: 'nappe' | 'systeme' = 'nappe') =>
      fetchJson<PiezoBdlisaSiblings>(`/observatory/piezo/stations/${encodeURIComponent(code)}/siblings`, { level }),
  },
  hydro: {
    stations: (params?: Record<string, string | string[] | undefined>) =>
      fetchJson<HydroStation[]>('/observatory/hydro/stations', params),
    detail: (code: string) => fetchJson<HydroStation>(`/observatory/hydro/stations/${code}`),
    percentiles: (code: string) =>
      fetchJson<StationPercentiles>(`/observatory/hydro/stations/${encodeURIComponent(code)}/percentiles`),
    daily: (code: string, params?: Record<string, string | undefined>) =>
      fetchJson<DailyHydroMeasurement[]>(`/observatory/hydro/stations/${code}/daily`, params),
    monthly: (code: string) => fetchJson<MonthlyHydroData[]>(`/observatory/hydro/stations/${code}/monthly`),
    yearly: (code: string) => fetchJson<YearlyHydroData[]>(`/observatory/hydro/stations/${code}/yearly`),
    ssfi: (code: string) => fetchJson<SSFIDataPoint[]>(`/observatory/hydro/stations/${code}/ssfi`),
    spi: (code: string) => fetchJson<SPIDataPoint[]>(`/observatory/hydro/stations/${code}/spi`),
    siblings: (code: string, level: 'site' | 'cours_eau' = 'site') =>
      fetchJson<HydroSiteSiblings>(`/observatory/hydro/stations/${encodeURIComponent(code)}/siblings`, { level }),
  },
  common: {
    geojson: (stationType?: 'piezo' | 'hydro' | 'all') =>
      fetchJson<StationGeoJSON>('/observatory/stations/geojson', stationType ? { type: stationType } : undefined),
    alerts: (params?: Record<string, string | string[] | undefined>) =>
      fetchJson<Alert[]>('/observatory/alerts', params),
    statsNational: () => fetchJson<NationalStats>('/observatory/stats/national'),
    classificationTimeline: () => fetchJson<ClassificationTimeline>('/observatory/classifications/timeline'),
  },
  era5: {
    grid: () => fetchJson<ERA5GridPoint[]>('/observatory/era5/grid'),
    snapshot: (date: string) => fetchJson<ERA5GridPoint[]>('/observatory/era5/snapshot', { date }),
    dates: () => fetchJson<string[]>('/observatory/era5/dates'),
    monthly: (month: string) => fetchJson<ERA5GridPoint[]>('/observatory/era5/monthly', { month }),
  },
  wfs: {
    layer: (layerId: string, bbox?: string) =>
      fetchJson<any>(`/observatory/wfs/${layerId}`, bbox ? { bbox } : undefined),
  },
  bdlisa: {
    entity: (code: string) =>
      fetchJson<{ polygon: GeoJSON.MultiPolygon; code: string; denomination: string | null; nature: string | null }>(`/observatory/bdlisa/entity`, { code }),
  },
  pastas: {
    summary: (code: string) =>
      fetchJson<ObsPastasSummary>(`/observatory/pastas/stations/${code}/summary`),
    timeseries: (code: string, params?: Record<string, string | undefined>) =>
      fetchJson<ObsPastasTimeseriesPoint[]>(`/observatory/pastas/stations/${code}/timeseries`, params),
    sgi: (code: string) =>
      fetchJson<ObsPastasSGIPoint[]>(`/observatory/pastas/stations/${code}/sgi`),
    coverage: () =>
      fetchJson<ObsPastasCoverage[]>('/observatory/pastas/coverage'),
  },
}
