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

// Models
export interface ModelSummary {
  id: string
  name: string
  model_type: string
  station: string | null
  metrics: Record<string, number>
  created_at: string
  mlflow_run_id: string | null
}

// Training
export interface TrainingConfig {
  dataset_id: string
  model_type: string
  hyperparams: Record<string, unknown>
  train_split: number
  val_split: number
  max_epochs: number
  early_stopping_patience: number
  station: string | null
}

export interface TrainingMetrics {
  epoch: number
  total_epochs: number
  train_loss: number | null
  val_loss: number | null
  best_val_loss: number | null
}

// Forecasting
export interface ForecastResult {
  dates: string[]
  predictions: (number | null)[]
  actuals: (number | null)[]
  metrics: Record<string, number>
  confidence_low: (number | null)[]
  confidence_high: (number | null)[]
}

// Counterfactual
export interface CounterfactualResult {
  original: number[]
  counterfactual: number[]
  dates: string[]
  theta: Record<string, number>
  metrics: Record<string, number>
}

// Available model architectures
export interface AvailableModel {
  name: string
  description: string
  category: string
  default_hyperparams: Record<string, unknown>
}
