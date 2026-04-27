import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useAdaptiveBounds(runId: string | null, tFinalDays: number | null) {
  return useQuery({
    queryKey: ['pastas', 'adaptive-bounds', runId, tFinalDays],
    queryFn: () => api.pastas.adaptiveBounds(runId!, tFinalDays ?? undefined),
    enabled: !!runId,
    staleTime: 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
    retry: false,
  })
}
