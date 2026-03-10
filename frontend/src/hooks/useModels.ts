import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
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

export function useModelTestInfo(modelId: string | null) {
  return useQuery({
    queryKey: ['model-test-info', modelId],
    queryFn: () => api.models.testInfo(modelId!),
    enabled: !!modelId,
  })
}

export function useDeleteModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.models.delete(id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['models'] })
    },
  })
}
