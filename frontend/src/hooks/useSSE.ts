import { useCallback, useEffect, useRef, useState } from 'react'

interface SSEState<T> {
  data: T | null
  status: 'idle' | 'connected' | 'done' | 'error'
  error: string | null
}

export function useSSE<T>(url: string | null): SSEState<T> {
  const [state, setState] = useState<SSEState<T>>({
    data: null,
    status: 'idle',
    error: null,
  })
  const esRef = useRef<EventSource | null>(null)

  const close = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!url) {
      setState({ data: null, status: 'idle', error: null })
      return
    }

    close()
    setState((prev) => ({ ...prev, status: 'idle', error: null }))

    const es = new EventSource(url)
    esRef.current = es

    es.onopen = () => {
      setState((prev) => ({ ...prev, status: 'connected' }))
    }

    es.addEventListener('metrics', (e: MessageEvent) => {
      try {
        const parsed = JSON.parse(e.data as string) as T
        setState({ data: parsed, status: 'connected', error: null })
      } catch {
        /* ignore parse errors */
      }
    })

    es.addEventListener('done', () => {
      setState((prev) => ({ ...prev, status: 'done' }))
      es.close()
      esRef.current = null
    })

    es.onerror = () => {
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: 'SSE connection lost',
      }))
      es.close()
      esRef.current = null
    }

    return close
  }, [url, close])

  return state
}
