import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useDatasets() {
  return useQuery({
    queryKey: ['datasets'],
    queryFn: api.datasets.list,
  })
}

export function useDataset(id: string | null) {
  return useQuery({
    queryKey: ['datasets', id],
    queryFn: () => api.datasets.get(id!),
    enabled: !!id,
  })
}
