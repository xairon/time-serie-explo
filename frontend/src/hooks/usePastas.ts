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

export function usePastasPreview(codeBss: string | null) {
  return useQuery({
    queryKey: ['pastas', 'preview', codeBss],
    queryFn: () => api.pastas.preview(codeBss!),
    enabled: !!codeBss,
    staleTime: 10 * 60 * 1000,
  })
}

export function usePastasDiagnostics(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'diagnostics', runId],
    queryFn: () => api.pastas.diagnostics(runId!),
    enabled: !!runId,
  })
}

export function usePastasSignatures(runId: string | null) {
  return useQuery({
    queryKey: ['pastas', 'signatures', runId],
    queryFn: () => api.pastas.signatures(runId!),
    enabled: !!runId,
  })
}

export function usePastasStationInfo(codeBss: string | null) {
  return useQuery({
    queryKey: ['pastas', 'station-info', codeBss],
    queryFn: () => api.pastas.stationInfo(codeBss!),
    enabled: !!codeBss,
    staleTime: 60 * 60 * 1000,
  })
}

export function usePastasCompare() {
  return useMutation({
    mutationFn: (runIds: string[]) => api.pastas.compare(runIds),
  })
}

export function usePastasSiblings(codeBss: string | null) {
  return useQuery({
    queryKey: ['pastas', 'siblings', codeBss],
    queryFn: () => api.pastas.siblings(codeBss!),
    enabled: !!codeBss,
    staleTime: 30 * 60 * 1000,
  })
}

export function usePastasDiagnose(codeBss: string | null) {
  return useQuery({
    queryKey: ['pastas', 'diagnose', codeBss],
    queryFn: () => api.pastas.diagnose(codeBss!),
    enabled: !!codeBss,
    staleTime: 10 * 60 * 1000,
  })
}

export function usePastasAutoFit() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.pastas.autoFit,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pastas', 'models'] })
    },
  })
}

export function usePastasCompareAI() {
  return useMutation({
    mutationFn: api.pastas.compareAI,
  })
}
