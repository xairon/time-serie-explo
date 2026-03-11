import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useCounterfactualRun() {
  return useMutation({
    mutationFn: (body: {
      model_id: string
      method?: string
      target_ips_classes?: Record<string, string>
      start_idx?: number
      lambda_prox?: number
      n_iter?: number
      lr?: number
      cc_rate?: number
      n_trials?: number
      num_distractors?: number
      tau?: number
    }) => api.counterfactual.run(body),
  })
}

export function useIPSReference(modelId: string | null, window: number = 3) {
  return useQuery({
    queryKey: ['ips-reference', modelId, window],
    queryFn: () => api.counterfactual.ipsReference(modelId!, window),
    enabled: !!modelId,
  })
}

export function useIPSBounds(modelId: string | null, window: number = 1) {
  return useQuery({
    queryKey: ['ips-bounds', modelId, window],
    queryFn: () => api.counterfactual.ipsBounds(modelId!, window),
    enabled: !!modelId,
  })
}
