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

export function useDatasetPreview(id: string | null, n: number = 100) {
  return useQuery({
    queryKey: ['dataset-preview', id, n],
    queryFn: () => api.datasets.preview(id!, n),
    enabled: !!id,
  })
}

export function useDatasetProfile(id: string | null) {
  return useQuery({
    queryKey: ['dataset-profile', id],
    queryFn: () => api.datasets.profile(id!),
    enabled: !!id,
  })
}
