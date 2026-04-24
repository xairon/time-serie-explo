import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useSavedScenarios(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'saved-scenarios', runId],
    queryFn: () => api.pastas.savedScenarios(runId!),
    enabled: !!runId,
  })
}

export function useSaveScenario() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ runId, body }: {
      runId: string
      body: { name: string; description?: string; modifications: Array<Record<string, unknown>>; tmin?: string; tmax?: string }
    }) => api.pastas.saveScenario(runId, body),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ['pastas', 'saved-scenarios', variables.runId] })
    },
  })
}

export function useDeleteScenario() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ runId, name }: { runId: string; name: string }) =>
      api.pastas.deleteScenario(runId, name),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ['pastas', 'saved-scenarios', variables.runId] })
    },
  })
}
