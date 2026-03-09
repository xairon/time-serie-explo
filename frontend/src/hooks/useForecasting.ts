import { useMutation } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useForecast() {
  return useMutation({
    mutationFn: (body: { model_id: string; horizon: number; dataset_id: string }) =>
      api.forecasting.run(body),
  })
}

export function useFeatureImportance() {
  return useMutation({
    mutationFn: (modelId: string) => api.explainability.featureImportance(modelId),
  })
}
