import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useStationEmbeddings(domain: string, space: string = 'multi') {
  return useQuery({
    queryKey: ['latent-space', 'stations', domain, space],
    queryFn: () => api.latentSpace.stations(domain, space),
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

export function useClusteringRuns(domain: string, space: string = 'multi') {
  return useQuery({
    queryKey: ['latent-space', 'clustering-runs', domain, space],
    queryFn: () => api.latentSpace.clusteringRuns(domain, space),
    staleTime: 5 * 60 * 1000,
    enabled: !!domain,
  })
}

export function useClusteringRun(runId: number | null) {
  return useQuery({
    queryKey: ['latent-space', 'clustering-run', runId],
    queryFn: () => api.latentSpace.clusteringRun(runId!),
    staleTime: 5 * 60 * 1000,
    enabled: runId != null,
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
