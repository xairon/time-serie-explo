import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useScenarioPresets(params?: { aquifer_family?: string; tmin?: string; tmax?: string }) {
  return useQuery({
    queryKey: ['pastas', 'scenario-presets', params?.aquifer_family, params?.tmin, params?.tmax],
    queryFn: () => api.pastas.scenarioPresets(params),
    staleTime: 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  })
}
