import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useRecomputePca() {
  return useMutation({
    mutationFn: (body: { domain: string; space: string; variance_threshold: number }) =>
      api.latentSpace.recomputePca(body),
  })
}

export function useRecomputeViz() {
  return useMutation({
    mutationFn: (body: { domain: string; space: string; n_neighbors: number; min_dist: number }) =>
      api.latentSpace.recomputeViz(body),
  })
}

export function useRecomputeClustering() {
  return useMutation({
    mutationFn: (body: { domain: string; space: string; min_cluster_size?: number; min_samples?: number }) =>
      api.latentSpace.recomputeClustering(body),
  })
}

export function useAutoTune() {
  return useMutation({
    mutationFn: (body: { domain: string; space: string }) =>
      api.latentSpace.autoTune(body),
  })
}

export function useCachedCompute(domain: string, space: string = 'multi') {
  return useQuery({
    queryKey: ['latent-space', 'cached', domain, space],
    queryFn: () => api.latentSpace.cached(domain, space),
    staleTime: 10 * 60 * 1000,
    retry: false,  // 404 = no cache, don't retry
    enabled: !!domain,
  })
}

export function useStationEmbeddings(domain: string, space: string = 'multi') {
  return useQuery({
    queryKey: ['latent-space', 'stations', domain, space],
    queryFn: () => api.latentSpace.stations(domain, space),
    staleTime: 5 * 60 * 1000,
    enabled: !!domain,
  })
}

export function useSimilarStations(domain: string, stationId: string | null, space: string = 'multi') {
  return useQuery({
    queryKey: ['latent-space', 'similar', domain, stationId, space],
    queryFn: () => api.latentSpace.similar(domain, stationId!, 10, space),
    staleTime: 5 * 60 * 1000,
    enabled: !!stationId,
  })
}

export function useStationWindows(domain: string, stationId: string | null, space: string = 'multi') {
  return useQuery({
    queryKey: ['latent-space', 'station-windows', domain, stationId, space],
    queryFn: () => api.latentSpace.stationWindows(domain, stationId!, space),
    staleTime: 5 * 60 * 1000,
    enabled: !!stationId,
  })
}

export function useClusterProfiling(domain: string, space: string = 'multi', hideUnclassified: boolean = false) {
  return useQuery({
    queryKey: ['latent-space', 'profiling', domain, space, hideUnclassified],
    queryFn: () => api.latentSpace.profiling(domain, hideUnclassified, space),
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    enabled: !!domain,
  })
}
