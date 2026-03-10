import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useCounterfactualRun() {
  return useMutation({
    mutationFn: (body: {
      model_id: string
      method?: string
      target_ips_class?: string
      from_ips_class?: string
      to_ips_class?: string
      modifications?: Record<string, number>
      lambda_prox?: number
      n_iter?: number
      lr?: number
      cc_rate?: number
      device?: string
      n_trials?: number
      seed?: number
      k_sigma?: number
      lambda_smooth?: number
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
