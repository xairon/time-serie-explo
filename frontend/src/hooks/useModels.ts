import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: api.models.list,
  })
}

export function useModelDetail(modelId: string | null) {
  return useQuery({
    queryKey: ['model-detail', modelId],
    queryFn: () => api.models.get(modelId!),
    enabled: !!modelId,
  })
}

export function useAvailableModels() {
  return useQuery({
    queryKey: ['available-models'],
    queryFn: api.models.available,
  })
}
