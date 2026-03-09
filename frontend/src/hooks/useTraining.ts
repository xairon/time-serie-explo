import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { TrainingConfig } from '@/lib/types'

export function useStartTraining() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (config: TrainingConfig) => api.training.start(config),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['models'] })
    },
  })
}

export function useStopTraining() {
  return useMutation({
    mutationFn: (taskId: string) => api.training.stop(taskId),
  })
}
