import { useMutation } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { ExplainResult } from '@/lib/types'

export function useForecast() {
  return useMutation({
    mutationFn: (body: { model_id: string; horizon?: number; dataset_id?: string }) =>
      api.forecasting.run(body),
  })
}

export function useForecastSingle() {
  return useMutation({
    mutationFn: (body: { model_id: string; start_date?: string; use_covariates?: boolean }) =>
      api.forecasting.single(body),
  })
}

export function useForecastRolling() {
  return useMutation({
    mutationFn: (body: { model_id: string; start_date: string; forecast_horizon: number; stride?: number }) =>
      api.forecasting.rolling(body),
  })
}

export function useForecastComparison() {
  return useMutation({
    mutationFn: (body: { model_id: string; start_date: string; forecast_horizon: number }) =>
      api.forecasting.comparison(body),
  })
}

export function useForecastGlobal() {
  return useMutation({
    mutationFn: (body: { model_id: string; use_covariates?: boolean }) =>
      api.forecasting.global(body),
  })
}

export function useFeatureImportance() {
  return useMutation({
    mutationFn: (modelId: string) => api.explainability.featureImportance(modelId),
  })
}

export function useShapAnalysis() {
  return useMutation({
    mutationFn: (body: { model_id: string; n_samples?: number }) =>
      api.explainability.shap(body),
  })
}

export function useAttentionAnalysis() {
  return useMutation({
    mutationFn: (body: { model_id: string }) =>
      api.explainability.attention(body),
  })
}

export function useGradientAnalysis() {
  return useMutation({
    mutationFn: (body: { model_id: string; method?: string }) =>
      api.explainability.gradients(body),
  })
}
