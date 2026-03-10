// Health
export interface HealthStatus {
  status: string
  db: string
  redis: string
  gpu: { available: boolean; device?: string; memory_total_gb?: number }
}

// Datasets
export interface DatasetSummary {
  id: string
  name: string
  source_file: string
  stations: string[]
  target_variable: string
  covariates: string[]
  date_range: string[]
  n_rows: number
  created_at: string
  station_column: string | null
}

export interface DatasetPreview {
  columns: string[]
  rows: Record<string, unknown>[]
  total_rows: number
}

export interface DatasetProfile {
  columns: Record<string, Record<string, unknown>>
  shape: number[]
  dtypes: Record<string, string>
  missing: Record<string, number>
  correlation: Record<string, Record<string, number>> | null
  timeseries_data: {
    dates: string[]
    series: Record<string, (number | null)[]>
  } | null
}

// Station metadata (from dim_piezo_stations)
export interface StationInfo {
  code_bss: string
  nom_commune: string | null
  code_departement: string | null
  nom_departement: string | null
  codes_bdlisa: string | null
  altitude_station: number | null
  latitude: number | null
  longitude: number | null
  premiere_mesure: string | null
  derniere_mesure: string | null
  nb_mesures_total: number | null
  niveau_moyen_global: number | null
  amplitude_totale: number | null
  tendance_classification: string | null
  niveau_alerte: string | null
  classification_derniere_annee: string | null
  qualite_tendance: string | null
}

// Models - matches backend ModelSummary schema
export interface ModelSummary {
  model_id: string
  model_name: string
  model_type: string
  stations: string[]
  primary_station: string | null
  created_at: string
  metrics: Record<string, number>
  data_source: string | null
}

export interface ModelDetail extends ModelSummary {
  run_id: string
  hyperparams: Record<string, unknown>
  preprocessing_config: Record<string, unknown>
  display_name: string
}

// Training - matches backend TrainingRequest schema
export interface TrainingConfig {
  dataset_id: string
  model_name: string
  hyperparams: Record<string, unknown>
  train_ratio: number
  val_ratio: number
  n_epochs: number | null
  early_stopping: boolean
  early_stopping_patience: number
  station_name?: string
  use_covariates: boolean
  loss_function: string
}

export interface TrainingMetrics {
  current_epoch: number
  total_epochs: number
  train_loss: number | null
  val_loss: number | null
  best_val_loss: number | null
  train_losses?: (number | null)[]
  val_losses?: (number | null)[]
  status?: string
  elapsed_seconds?: number
  eta_seconds?: number | null
}

export interface TrainingResult {
  task_id: string
  status: string
  metrics: Record<string, number> | null
  metrics_sliding: Record<string, number> | null
  model_name: string | null
  station: string | null
  error: string | null
}

// Forecasting - backend returns predictions as list of {time, column: value} dicts
export interface ForecastTimePoint {
  time: string
  [key: string]: unknown
}

export interface ForecastResultRaw {
  predictions: ForecastTimePoint[]
  target: ForecastTimePoint[]
  metrics: Record<string, number>
  horizon: number | null
  predictions_onestep: ForecastTimePoint[] | null
  metrics_onestep: Record<string, number> | null
  predictions_exact: ForecastTimePoint[] | null
  metrics_exact: Record<string, number> | null
  forecast_windows: ForecastTimePoint[][] | null
}

// Transformed forecast result for UI consumption
export interface ForecastResult {
  dates: string[]
  predictions: (number | null)[]
  actuals: (number | null)[]
  metrics: Record<string, number>
  confidence_low: (number | null)[]
  confidence_high: (number | null)[]
  predictions_onestep: (number | null)[] | null
  metrics_onestep: Record<string, number> | null
  predictions_exact: (number | null)[] | null
  metrics_exact: Record<string, number> | null
}

// Model test set info (for sliding window UI)
export interface ModelTestInfo {
  test_dates: string[]
  test_values: (number | null)[]
  test_length: number
  input_chunk_length: number
  output_chunk_length: number
  valid_start_idx: number
  valid_end_idx: number
  target_column: string
}

// Counterfactual - matches backend CFResult schema
export interface CounterfactualResult {
  task_id: string
  status: string
  result: {
    method: string
    original: number[]
    counterfactual: number[]
    dates: string[]
    theta: Record<string, number>
    metrics: Record<string, number>
    convergence?: number[]
    best_trial?: Record<string, unknown>
  } | null
  error: string | null
}

// Available model architectures - matches backend AvailableModel schema
export interface AvailableModel {
  name: string
  is_torch: boolean
  description: string
  category: string
  default_hyperparams: Record<string, unknown>
}

// Explainability
export interface ExplainResult {
  method: string
  success: boolean
  feature_importance: Record<string, number | null> | null
  feature_names: string[] | null
  temporal_importance: number[] | null
  attention_weights: number[][] | null
  shap_values: number[][] | null
  gradient_attributions: number[][] | null
  encoder_importance: Record<string, number> | null
  decoder_importance: Record<string, number> | null
  model_type: string | null
  error_message: string | null
}

// Lag Importance (autocorrelation)
export interface LagImportanceResult {
  lags: number[]
  autocorrelations: number[]
  partial_autocorrelations: number[] | null
  significant_lags: number[] | null
}

// Residual Analysis
export interface ResidualAnalysisResult {
  mean_error: number
  std_error: number
  skewness: number | null
  kurtosis: number | null
  normality_pvalue: number | null
  acf_lag1: number | null
  bias_status: string
  residuals: number[] | null
  dates: string[] | null
}

// Seasonality Detection
export interface SeasonalityResult {
  detected_periods: number[]
  period_strengths: Record<string, number> | null
  decomposition: { trend: number[]; seasonal: number[]; residual: number[] } | null
  decomposition_dates: string[] | null
  variance_trend: number | null
  variance_seasonal: number | null
  variance_residual: number | null
}

// IPS Reference
export interface IPSReference {
  model_id: string
  window: number
  ref_stats: Record<string, unknown>
  mu_target: number | null
  sigma_target: number | null
  n_years: number | null
  validation: Record<string, unknown> | null
}

// IPS Bounds (per-month class bounds in physical units)
export interface IPSBoundsRow {
  month_start: string
  month_end: string
  month: number
  mu: number
  sigma: number
  [key: string]: unknown // {cls}_lower, {cls}_upper
}

export interface IPSBoundsResponse {
  bounds: IPSBoundsRow[]
  classes: Record<string, string>  // key -> label FR
  colors: Record<string, string>   // key -> hex color
}

// Typed inner result from CF generation
export interface CFInnerResult {
  method: string
  original: number[]
  counterfactual: number[]
  dates: string[]
  theta: Record<string, number>
  metrics: Record<string, number | boolean | string>
  convergence?: number[]
  best_trial?: Record<string, unknown>
}
