import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function usePastasOptions() {
  return useQuery({
    queryKey: ['pastas', 'options'],
    queryFn: () => api.pastas.options(),
    staleTime: 60 * 60 * 1000,
  })
}

export function usePastasFit() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.pastas.fit,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pastas', 'models'] })
    },
  })
}

export function usePastasModels(codeBss?: string) {
  return useQuery({
    queryKey: ['pastas', 'models', codeBss],
    queryFn: () => api.pastas.models(codeBss),
  })
}

export function usePastasModel(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'model', runId],
    queryFn: () => api.pastas.model(runId!),
    enabled: !!runId,
  })
}

export function usePastasDeleteModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.pastas.deleteModel,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pastas', 'models'] })
    },
  })
}

export function usePastasSimulate() {
  return useMutation({
    mutationFn: api.pastas.simulate,
  })
}
