import { useMutation } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function useCounterfactualRun() {
  return useMutation({
    mutationFn: (body: {
      model_id: string
      dataset_id: string
      modifications: Record<string, number>
    }) => api.counterfactual.run(body),
  })
}
