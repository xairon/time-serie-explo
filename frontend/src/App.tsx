import { useState, Component, type ReactNode, type ErrorInfo } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// Simple class-based ErrorBoundary
class ErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div role="alert" style={{ padding: 32, textAlign: 'center', color: '#f87171' }}>
          <h1 style={{ fontSize: 24, marginBottom: 8 }}>Une erreur est survenue</h1>
          <p style={{ color: '#9ca3af', marginBottom: 16 }}>
            {this.state.error?.message ?? 'Erreur inconnue'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              padding: '8px 16px',
              borderRadius: 8,
              border: '1px solid rgba(255,255,255,0.1)',
              background: '#1f2937',
              color: '#e5e7eb',
              cursor: 'pointer',
            }}
          >
            Réessayer
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

export default function App() {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 5 * 60 * 1000,
            gcTime: 30 * 60 * 1000,
            refetchOnWindowFocus: false,
            retry: 1,
          },
        },
      }),
  )

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <div className="min-h-screen bg-bg-primary text-text-primary">
          <p className="p-8 text-text-secondary">Junon Time-Series Explorer — routes not configured yet.</p>
        </div>
      </QueryClientProvider>
    </ErrorBoundary>
  )
}
