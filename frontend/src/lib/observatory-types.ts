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
  classification_derniere_annee: string | null
  niveau_derniere_annee: number | null
  codes_bdlisa: string | null
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
  classification_resultat_dern_annee: string | null
  resultat_min_global: number | null
  resultat_max_global: number | null
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
  code_site: string
  libelle_site: string | null
  nom_cours_eau: string | null
  nb_stations: number
  siblings: HydroSiblingStation[]
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
