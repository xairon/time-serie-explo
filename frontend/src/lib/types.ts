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
  source: 'csv' | 'db'
  stations: string[]
  target_variable: string
  covariates: string[]
  date_range: [string, string]
  n_rows: number
  created_at: string
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
