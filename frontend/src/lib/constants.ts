export const API_BASE = '/api/v1'

export const MODEL_COLORS: Record<string, string> = {
  NBEATS: '#06b6d4',
  TFT: '#6366f1',
  TCN: '#10b981',
  LSTM: '#f59e0b',
  GRU: '#ef4444',
  Transformer: '#8b5cf6',
}

export const METRIC_LABELS: Record<string, string> = {
  MAE: 'MAE',
  RMSE: 'RMSE',
  sMAPE: 'sMAPE',
  WAPE: 'WAPE',
  NRMSE: 'NRMSE',
  Dir_Acc: 'Dir. Accuracy',
  NSE: 'Nash-Sutcliffe',
  KGE: 'Kling-Gupta',
}
