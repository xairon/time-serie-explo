import { useMutation, useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '@/lib/api'

interface PumpingStage {
  stage: string
  pct: number
  message: string
}

interface PumpingDetectionState {
  taskId: string | null
  stages: PumpingStage[]
  currentStage: PumpingStage | null
  partialResults: Record<string, unknown>
  status: 'idle' | 'running' | 'done' | 'error' | 'cancelled'
  error: string | null
}

export function usePumpingDetection() {
  const [state, setState] = useState<PumpingDetectionState>({
    taskId: null,
    stages: [],
    currentStage: null,
    partialResults: {},
    status: 'idle',
    error: null,
  })
  const esRef = useRef<EventSource | null>(null)

  const analyzeMutation = useMutation({
    mutationFn: (body: { dataset_id: string; config?: Record<string, unknown> }) =>
      api.pumpingDetection.analyze(body),
    onSuccess: (data) => {
      setState(prev => ({ ...prev, taskId: data.task_id, status: 'running', stages: [] }))
    },
  })

  useEffect(() => {
    if (!state.taskId || state.status !== 'running') return

    const es = api.pumpingDetection.stream(state.taskId)
    esRef.current = es

    es.addEventListener('progress', (e: MessageEvent) => {
      const data = JSON.parse(e.data) as PumpingStage
      setState(prev => ({
        ...prev,
        currentStage: data,
        stages: [...prev.stages, data],
      }))
    })

    es.addEventListener('metrics', (e: MessageEvent) => {
      const data = JSON.parse(e.data) as { stage: string; partial_result: unknown }
      setState(prev => ({
        ...prev,
        partialResults: { ...prev.partialResults, [data.stage]: data.partial_result },
      }))
    })

    es.addEventListener('error', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data) as { recoverable?: boolean; error_message?: string }
        if (!data.recoverable) {
          setState(prev => ({ ...prev, status: 'error', error: data.error_message ?? 'Erreur inconnue' }))
        }
      } catch {
        setState(prev => ({ ...prev, status: 'error', error: 'Connexion SSE perdue' }))
      }
    })

    es.addEventListener('done', () => {
      setState(prev => ({ ...prev, status: 'done' }))
      es.close()
    })

    return () => { es.close(); esRef.current = null }
  }, [state.taskId, state.status])

  const cancel = useCallback(() => {
    if (state.taskId) {
      void api.pumpingDetection.cancel(state.taskId)
      setState(prev => ({ ...prev, status: 'cancelled' }))
      esRef.current?.close()
    }
  }, [state.taskId])

  const reset = useCallback(() => {
    esRef.current?.close()
    setState({ taskId: null, stages: [], currentStage: null, partialResults: {}, status: 'idle', error: null })
  }, [])

  return {
    analyze: analyzeMutation.mutate,
    cancel,
    reset,
    ...state,
    isAnalyzing: analyzeMutation.isPending || state.status === 'running',
  }
}

export function usePumpingResults(taskId: string | null) {
  return useQuery({
    queryKey: ['pumping-results', taskId],
    queryFn: () => api.pumpingDetection.results(taskId!),
    enabled: !!taskId,
  })
}

export function useBNPEContext(lat: number | null, lon: number | null) {
  return useQuery({
    queryKey: ['bnpe-context', lat, lon],
    queryFn: () => api.pumpingDetection.bnpeContext(lat!, lon!),
    enabled: lat != null && lon != null,
  })
}
