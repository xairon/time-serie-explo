import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: api.models.list,
  })
}

export function useAvailableModels() {
  return useQuery({
    queryKey: ['available-models'],
    queryFn: api.training.availableModels,
  })
}
