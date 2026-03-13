import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useStationEmbeddings(domain: string) {
  return useQuery({
    queryKey: ['latent-space', 'stations', domain],
    queryFn: () => api.latentSpace.stations(domain),
    staleTime: 5 * 60 * 1000,
    enabled: !!domain,
  })
}

export function useSimilarStations(domain: string, stationId: string | null) {
  return useQuery({
    queryKey: ['latent-space', 'similar', domain, stationId],
    queryFn: () => api.latentSpace.similar(domain, stationId!, 10),
    staleTime: 5 * 60 * 1000,
    enabled: !!stationId,
  })
}

export function useComputeUMAP() {
  return useMutation({
    mutationFn: (body: Record<string, unknown>) => api.latentSpace.compute(body),
  })
}

export function useClusterProfiling(domain: string, hideUnclassified: boolean) {
  return useQuery({
    queryKey: ['latent-space', 'profiling', domain, hideUnclassified],
    queryFn: () => api.latentSpace.profiling(domain, hideUnclassified),
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    enabled: !!domain,
  })
}
