// Observatory types — ported from junondashboard

// Station types
export interface PiezoStation {
  code_bss: string
  bss_id?: string
  latitude: number | null
  longitude: number | null
  nom_commune: string | null
  code_departement: string | null
  nom_departement: string | null
  nb_mesures_total: number | null
  derniere_mesure: string | null
  niveau_derniere_annee: number | null
  codes_bdlisa: string | null
  reference_flag: string | null
  index_class_bounds: number[] | null
}

export interface HydroStation {
  code_station: string
  code_site: string | null
  libelle_station: string | null
  libelle_site: string | null
  code_cours_eau: string | null
  nom_cours_eau: string | null
  latitude_station: number | null
  longitude_station: number | null
  code_departement: string | null
  nom_departement: string | null
  grandeur_hydro_principale: 'Q' | 'H' | null
  nb_jours_total: number | null
  derniere_mesure: string | null
  resultat_min_global: number | null
  resultat_max_global: number | null
  reference_flag: string | null
  index_class_bounds: number[] | null
}

// Timeseries types
export interface DailyPiezoMeasurement {
  date_mesure: string
  niveau_nappe_eau: number | null
  profondeur_nappe: number | null
  qualification: string | null
}

export interface DailyHydroMeasurement {
  date_obs_elab: string
  resultat_obs_elab: number | null
  grandeur_hydro_elab: string | null
}

export interface MonthlyPiezoData {
  mois: string
  niveau_moyen: number | null
  niveau_min: number | null
  niveau_max: number | null
  nb_jours_mesures: number | null
  precipitation_totale: number | null
  temperature_moyenne: number | null
  evaporation_moyenne: number | null
}

export interface MonthlyHydroData {
  mois: string
  resultat_moyen: number | null
  resultat_min: number | null
  resultat_max: number | null
  nb_jours_mesures: number | null
}

export interface YearlyPiezoData {
  annee: number
  niveau_moyen_annuel: number | null
  niveau_min_annuel: number | null
  niveau_max_annuel: number | null
  amplitude_annuelle: number | null
  nb_jours_mesures_annuel: number | null
  classification_niveau_annuel: string | null
  precipitation_totale_annuelle: number | null
  bilan_hydrique_annuel: number | null
  percentile_niveau_historique: number | null
  niveau_moy_mobile_5ans: number | null
  temperature_moyenne_annuelle: number | null
}

export interface YearlyHydroData {
  annee: number
  resultat_moyen_annuel: number | null
  resultat_min_annuel: number | null
  resultat_max_annuel: number | null
  nb_jours_mesures_annuel: number | null
  classification_resultat_annuel: string | null
  percentile_resultat_historique: number | null
  resultat_moy_mobile_5ans: number | null
  temperature_moyenne_annuelle: number | null
  precipitation_totale_annuelle: number | null
}

// Sibling/grouping types
export interface HydroSiblingStation {
  code_station: string
  libelle_station: string | null
  grandeur_hydro_principale: string | null
  classification: string | null
  derniere_mesure: string | null
}

export interface HydroSiteSiblings {
  code_site: string | null
  libelle_site: string | null
  nom_cours_eau: string | null
  nb_stations: number
  level: string
  siblings: HydroSiblingStation[]
}

export interface PiezoBdlisaSibling {
  code_bss: string
  nom_commune: string | null
  codes_bdlisa: string | null
  classification: string | null
}

export interface PiezoBdlisaSiblings {
  level: string
  code_bdlisa: string | null
  non_rattachee: boolean
  nb_stations: number
  siblings: PiezoBdlisaSibling[]
}

// Drought index types
export interface SPIDataPoint {
  mois: string
  value: number | null
  spi: number | null
  classification: string
}

export interface SPLIDataPoint {
  mois: string
  value: number | null
  spli: number | null
  classification: string
}

export interface SSFIDataPoint {
  mois: string
  value: number | null
  ssfi: number | null
  classification: string
}

// Stats types
export interface NationalStats {
  total_piezo: number
  piezo_extremement_bas: number
  piezo_tres_bas: number
  piezo_bas: number
  piezo_normal: number
  piezo_haut: number
  piezo_tres_haut: number
  piezo_extremement_haut: number
  piezo_no_class: number
  total_hydro: number
  hydro_extremement_bas: number
  hydro_tres_bas: number
  hydro_bas: number
  hydro_normal: number
  hydro_haut: number
  hydro_tres_haut: number
  hydro_extremement_haut: number
}

// Alert types
export interface Alert {
  code: string
  type: 'piezo' | 'hydro'
  latitude: number | null
  longitude: number | null
  commune: string | null
  code_departement: string | null
  departement: string | null
  classification: string | null
  derniere_mesure: string | null
  alerte_depuis_annee: number | null
  nb_annees_consecutives: number | null
}

// ERA5 types
export interface ERA5GridPoint {
  latitude: number
  longitude: number
  temperature_2m: number | null
  total_precipitation: number | null
  potential_evaporation: number | null
}

export interface ERA5Range {
  min_date: string
  max_date: string
}

export interface ERA5AnomalyPoint {
  latitude: number
  longitude: number
  /** Generic anomaly field returned by /anomaly?variable=... (°C for temperature, % for precipitation). */
  anomaly?: number | null
}

export interface ERA5StiPoint {
  latitude: number
  longitude: number
  /** Standardized Temperature Index z-score (null when reference or observations are insufficient). */
  sti: number | null
  /** McKee 7-class string or 'UNKNOWN'. */
  index_class: string | null
}

export interface ERA5SpiPoint {
  latitude: number
  longitude: number
  /** Standardized Precipitation Index z-score (McKee 1993, gamma). */
  spi: number | null
  /** McKee 7-class string or 'UNKNOWN'. */
  index_class: string | null
}

// --- Climat module types (api/routers/observatory_climat.py — reads gold ERA5 grid marts) ---

/** One grid cell from GET /observatory/climat/grid-monthly (temperature/precip/etp/bilan_hydrique). */
export interface ClimatMonthlyPoint {
  latitude: number
  longitude: number
  value: number | null
  mois_complet: boolean | null
}

/** One grid cell from GET /observatory/climat/grid-indices?index=spi|sti — only the requested index key is present. */
export interface ClimatIndexPoint {
  latitude: number
  longitude: number
  spi?: number | null
  sti?: number | null
  /** McKee 7-class string (server always returns a non-null index for cells present in the response). */
  index_class: string
}

/** Territory-wide aggregate from GET /observatory/climat/situation-summary. */
export interface ClimatSituationSummary {
  month: string
  window: number
  n_cells: number
  classes_pct: Record<string, number>
  pct_secheresse: number
  median_spi: number | null
  driest_since_year: number | null
  is_driest_on_record: boolean
  top5_cellules_seches: Array<{ latitude: number; longitude: number; spi: number }>
}

/** One monthly row from GET /observatory/climat/point-series (Task B2) — merges the
 *  monthly grid variables with the calendar-month normal (window-1 climatology) and
 *  SPI/STI for the 4 standard windows (1/3/6/12 months). */
export interface ClimatPointSeriesEntry {
  month: string
  temperature_moyenne: number | null
  temperature_min: number | null
  temperature_max: number | null
  precipitation_totale: number | null
  etp_totale: number | null
  bilan_hydrique: number | null
  nb_jours: number | null
  mois_complet: boolean | null
  precipitation_normale: number | null
  temperature_normale: number | null
  spi_1: number | null
  sti_1: number | null
  spi_3: number | null
  sti_3: number | null
  spi_6: number | null
  sti_6: number | null
  spi_12: number | null
  sti_12: number | null
}

/** GET /observatory/climat/point-series response — the full history for the nearest cell. */
export interface ClimatPointSeries {
  cell: { latitude: number; longitude: number }
  series: ClimatPointSeriesEntry[]
}

/** One drought episode from GET /observatory/climat/point-episodes — a run of
 *  consecutive calendar months with SPI < -1 at the requested window. */
export interface ClimatDroughtEpisode {
  debut: string
  fin: string
  duree_mois: number
  spi_min: number
  deficit_cumule_mm: number
}

/** One month of the running cumulative sum within a compare-years year (Task B3). */
export interface ClimatCompareMonthCumul {
  mois: number
  precipitation: number
  cumul: number
  cumul_normal: number
}

/** One month of the fixed-window-3 SPI series within a compare-years year (Task B3). */
export interface ClimatCompareMonthSpi {
  mois: number
  spi: number | null
}

/** Per-year payload of GET /observatory/climat/compare-years. */
export interface ClimatCompareYearData {
  cumul_mensuel: ClimatCompareMonthCumul[]
  spi_3: ClimatCompareMonthSpi[]
}

/** GET /observatory/climat/compare-years response (Task B3 — Vue Comparaison):
 *  per requested year, the monthly running rainfall cumulative sum (vs. the calendar-month
 *  normal) and the 3-month SPI series, for the grid cell nearest lat/lon. */
export interface ClimatCompareYears {
  cell: { latitude: number; longitude: number }
  years: Record<string, ClimatCompareYearData>
}

export interface StationPercentiles {
  p10: number | null
  p25: number | null
  p75: number | null
  p90: number | null
}

// GeoJSON station types (endpoint /stations/geojson)
export interface StationGeoJSONProperties {
  code: string
  type: 'piezo' | 'hydro'
  classification: string | null
  commune: string | null
  departement: string | null
  code_departement: string | null
  codes_bdlisa?: string | null    // piezo only
  code_district?: string | null   // hydro only
  code_site?: string | null       // hydro only
  derniere_mesure: string | null
  nb_observations: number | null
  fiabilite?: 'fiable' | 'indicatif' | 'insuffisant'
}

export interface StationGeoJSONFeature {
  type: 'Feature'
  geometry: { type: 'Point'; coordinates: [number, number] }
  properties: StationGeoJSONProperties
}

export interface StationGeoJSON {
  type: 'FeatureCollection'
  features: StationGeoJSONFeature[]
}

// Timeline
export interface ClassificationTimeline {
  periods: string[]  // ['2005-01', '2005-02', ...]
  stations: Record<string, number[]>  // code -> array of classification indices per period
}

// Classification codes for timeline: 0=EXTREMEMENT_BAS...6=EXTREMEMENT_HAUT, 7=UNKNOWN
export const TIMELINE_CLASSIFICATIONS = [
  'EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT', 'UNKNOWN',
] as const

// Classification
export type Classification = 'EXTREMEMENT_BAS' | 'TRES_BAS' | 'BAS' | 'NORMAL' | 'HAUT' | 'TRES_HAUT' | 'EXTREMEMENT_HAUT' | 'UNKNOWN'
export type TrendClassification = 'HAUSSE_FORTE' | 'HAUSSE_SIGNIFICATIVE' | 'STABLE' | 'BAISSE_SIGNIFICATIVE' | 'BAISSE_FORTE'

// Chart tooltip style
export const CHART_TOOLTIP_STYLE = {
  backgroundColor: '#111827',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 8,
  fontSize: 12,
} as const

// WFS Layer types
export type WfsLayerId =
  | 'region-hydro' | 'secteur-hydro' | 'sous-secteur-hydro' | 'zone-hydro'
  | 'cours-eau-1' | 'cours-eau-2' | 'plan-eau'
  | 'masse-eau-riv'

export interface WfsLayerConfig {
  id: WfsLayerId
  label: string
  group: 'sandre' | 'carthage' | 'hydroeco' | 'admin'
  minZoom: number
  geometryType: 'polygon' | 'line'
  color: string
  tooltipFields: string[]
}

// PASTAS model types (observatory-specific — separate from existing Pastas types)
export interface ObsPastasSummary {
  code_bss: string
  evp: number | null
  nash: number | null
  kge: number | null
  rmse: number | null
  r2: number | null
  tmax_days: number | null
  cutoff_95_days: number | null
  gain: number | null
  mean_response_time: number | null
  block_response: number[] | null
  autocorr_time: number | null
  recession_constant: number | null
  recovery_constant: number | null
  parde_seasonality: number | null
  avg_seasonal_fluctuation: number | null
  colwell_constancy: number | null
  duration_curve_slope: number | null
  baselevel_index: number | null
  series_start: string | null
  series_end: string | null
  series_length_days: number | null
  n_observations: number | null
  fitted_at: string | null
  pastas_version: string | null
}

export interface ObsPastasTimeseriesPoint {
  date: string
  simulated: number | null
  observed: number | null
  residuals: number | null
  recharge_contribution: number | null
  wb_recharge: number | null
  wb_actual_evaporation: number | null
  wb_surface_runoff: number | null
  wb_effective_precip: number | null
}

export interface ObsPastasSGIPoint {
  date: string
  sgi: number | null
  classification: string
}

export interface ObsPastasCoverage {
  code_bss: string
  evp: number | null
  nash: number | null
  tmax_days: number | null
}

export type SituationClass =
  | 'EXTREMEMENT_BAS' | 'TRES_BAS' | 'BAS' | 'NORMAL'
  | 'HAUT' | 'TRES_HAUT' | 'EXTREMEMENT_HAUT'
export type Trend = 'hausse' | 'stable' | 'baisse'

export interface Outlook {
  horizon_months: number
  situation_class: SituationClass | null
  trend: Trend | null
  confidence: number | null
  coverage_pct: number | null
}

export interface TerritorySituation {
  level: 'region' | 'department'
  code: string
  name: string
  type: 'piezo' | 'hydro'
  situation_class: SituationClass | null
  trend: Trend | null
  pct_below_normal: number | null
  n_eligible: number
  n_provisoire: number
  distribution: Record<string, number>
  insufficient: boolean
  outlook: Outlook | null
}

export interface SectorSituation {
  level: 'sector'
  code: string
  name: string
  type: 'piezo' | 'hydro'
  situation_class: SituationClass | null
  trend: Trend | null
  pct_below_normal: number | null
  n_eligible: number
  n_provisoire: number
  insufficient: boolean
  tendancy_coord: string | null
}

export interface SectorTimeline {
  periods: string[]
  sectors: Record<string, number[]>
  trends: Record<string, number[]>
}

