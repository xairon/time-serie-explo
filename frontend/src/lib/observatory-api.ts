// Observatory API — ported from junondashboard, adapted for /api/v1/observatory/ prefix
import { API_BASE } from './constants'
import type {
  PiezoStation, HydroStation, NationalStats,
  Alert, ERA5GridPoint, ERA5Range, ERA5StiPoint, ERA5SpiPoint,
  DailyPiezoMeasurement, DailyHydroMeasurement,
  MonthlyPiezoData, MonthlyHydroData,
  YearlyPiezoData, YearlyHydroData,
  StationPercentiles,
  StationGeoJSON, ClassificationTimeline,
  SPIDataPoint, SPLIDataPoint, SSFIDataPoint,
  HydroSiteSiblings,
  PiezoBdlisaSiblings,
  ObsPastasSummary, ObsPastasTimeseriesPoint, ObsPastasSGIPoint, ObsPastasCoverage,
  ClimatMonthlyPoint, ClimatIndexPoint, ClimatSituationSummary, ClimatPointSeries, ClimatDroughtEpisode,
  ClimatCompareYears, ClimatRange,
} from './observatory-types'

export const EXPORT_COLUMN_GROUPS = ['identity', 'values', 'meteo', 'index', 'provenance'] as const
export type ExportColumnGroup = (typeof EXPORT_COLUMN_GROUPS)[number]

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

export interface ExportRange {
  start_date?: string
  end_date?: string
}

function exportQuery(range?: ExportRange, groups?: string[]): string {
  const qs = new URLSearchParams()
  if (range?.start_date) qs.set('start_date', range.start_date)
  if (range?.end_date) qs.set('end_date', range.end_date)
  if (groups && groups.length > 0 && groups.length < EXPORT_COLUMN_GROUPS.length) {
    qs.set('groups', groups.join(','))
  }
  const s = qs.toString()
  return s ? `?${s}` : ''
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
    spi: (code: string, window: number = 1) =>
      fetchJson<SPIDataPoint[]>(`/observatory/piezo/stations/${code}/spi`, { window: String(window) }),
    siblings: (code: string, level: 'nappe' | 'systeme' = 'nappe') =>
      fetchJson<PiezoBdlisaSiblings>(`/observatory/piezo/stations/${encodeURIComponent(code)}/siblings`, { level }),
    exportUrl: (code: string, range?: ExportRange, groups?: string[]) =>
      `${API_BASE}/observatory/piezo/stations/${encodeURIComponent(code)}/export.csv${exportQuery(range, groups)}`,
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
    spi: (code: string, window: number = 1) =>
      fetchJson<SPIDataPoint[]>(`/observatory/hydro/stations/${code}/spi`, { window: String(window) }),
    siblings: (code: string, level: 'site' | 'cours_eau' = 'site') =>
      fetchJson<HydroSiteSiblings>(`/observatory/hydro/stations/${encodeURIComponent(code)}/siblings`, { level }),
    exportUrl: (code: string, range?: ExportRange, groups?: string[]) =>
      `${API_BASE}/observatory/hydro/stations/${encodeURIComponent(code)}/export.csv${exportQuery(range, groups)}`,
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
    // /grid returns only the raw grid-point coordinates (era5_latitude/longitude),
    // NOT the full ERA5GridPoint shape — mistyping it made .latitude read undefined.
    grid: () => fetchJson<{ era5_latitude: number; era5_longitude: number }[]>('/observatory/era5/grid'),
    snapshot: (date: string) => fetchJson<ERA5GridPoint[]>('/observatory/era5/snapshot', { date }),
    dates: () => fetchJson<string[]>('/observatory/era5/dates'),
    monthly: (month: string) => fetchJson<ERA5GridPoint[]>('/observatory/era5/monthly', { month }),
    range: () => fetchJson<ERA5Range>('/observatory/era5/range'),
    sti: (date: string, window: number) =>
      fetchJson<ERA5StiPoint[]>('/observatory/era5/sti', { date, window: String(window) }),
    spi: (date: string, window: number) =>
      fetchJson<ERA5SpiPoint[]>('/observatory/era5/spi', { date, window: String(window) }),
  },
  // Climat module (Lot 2) — thin read layer over the BRGM gold ERA5 grid marts
  // (fct_era5_monthly_grid / fct_era5_climatology_grid / fct_era5_indices_grid).
  // No stats are recomputed server-side; see api/routers/observatory_climat.py.
  climat: {
    // Month bounds, split by data availability (indices vs. raw monthly) — see
    // ClimatRange. Drives ClimatPage's default month + MonthStepper bounds.
    range: () => fetchJson<ClimatRange>('/observatory/climat/range'),
    gridMonthly: (month: string, variable: string) =>
      fetchJson<ClimatMonthlyPoint[]>('/observatory/climat/grid-monthly', { month, variable }),
    gridIndices: (month: string, window: number, index: 'spi' | 'sti') =>
      fetchJson<ClimatIndexPoint[]>('/observatory/climat/grid-indices', { month, window: String(window), index }),
    situationSummary: (month: string, window: number) =>
      fetchJson<ClimatSituationSummary>('/observatory/climat/situation-summary', { month, window: String(window) }),
    // Point/Zone view (Task B2) — full history + drought episodes for the grid cell
    // nearest to lat/lon (rounded to 0.1° server-side).
    pointSeries: (lat: number, lon: number) =>
      fetchJson<ClimatPointSeries>('/observatory/climat/point-series', { lat: String(lat), lon: String(lon) }),
    pointEpisodes: (lat: number, lon: number, window: number) =>
      fetchJson<ClimatDroughtEpisode[]>('/observatory/climat/point-episodes', {
        lat: String(lat), lon: String(lon), window: String(window),
      }),
    // Comparaison view (Task B3) — per-year cumulative rainfall + 3-month SPI series
    // for the grid cell nearest lat/lon, over the requested years (2-15 accepted
    // server-side; the front bounds the multi-select to 2-6, see climat-year-select.ts).
    compareYears: (lat: number, lon: number, years: number[]) =>
      fetchJson<ClimatCompareYears>('/observatory/climat/compare-years', {
        lat: String(lat), lon: String(lon), years: years.join(','),
      }),
    // Direct download link (no fetch-into-memory) — the browser streams the CSV response.
    exportPointUrl: (lat: number, lon: number) =>
      `${API_BASE}/observatory/climat/export-point.csv?lat=${encodeURIComponent(String(lat))}&lon=${encodeURIComponent(String(lon))}`,
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
