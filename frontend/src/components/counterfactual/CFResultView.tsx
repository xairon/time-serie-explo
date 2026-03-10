import { CFOverlayPlot } from '@/components/charts/CFOverlayPlot'
import { METRIC_LABELS } from '@/lib/constants'
import type { CounterfactualResult } from '@/lib/types'

interface CFResultViewProps {
  result: CounterfactualResult | null
  isLoading: boolean
  className?: string
}

export function CFResultView({ result, isLoading, className = '' }: CFResultViewProps) {
  if (isLoading) {
    return (
      <div className={`bg-bg-card rounded-xl border border-white/5 p-6 ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-bg-hover rounded w-1/3" />
          <div className="h-[300px] bg-bg-hover rounded-lg" />
          <p className="text-xs text-text-secondary text-center">
            {result?.status === 'pending' ? 'En attente de traitement...' : 'Génération en cours...'}
          </p>
        </div>
      </div>
    )
  }

  if (result?.error) {
    return (
      <div className={`bg-bg-card rounded-xl border border-accent-red/20 p-6 ${className}`}>
        <p className="text-sm text-accent-red">Erreur : {result.error}</p>
      </div>
    )
  }

  if (!result || !result.result) {
    return (
      <div
        className={`bg-bg-card rounded-xl border border-white/5 p-6 flex items-center justify-center ${className}`}
      >
        <p className="text-text-secondary text-sm">
          Configurez et lancez une analyse contrefactuelle pour voir les résultats.
        </p>
      </div>
    )
  }

  const inner = result.result

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="bg-bg-card rounded-xl border border-white/5 p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs text-text-secondary uppercase">
            Méthode : {inner.method}
          </h4>
        </div>
        <CFOverlayPlot result={result} className="h-[350px]" />
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {Object.entries(inner.metrics).map(([key, val]) => (
          <div key={key} className="bg-bg-card rounded-lg p-3 border border-white/5 text-center">
            <p className="text-[10px] text-text-secondary uppercase">
              {METRIC_LABELS[key] ?? key}
            </p>
            <p className="text-base font-bold text-text-primary">{val.toFixed(4)}</p>
          </div>
        ))}
      </div>

      {/* Theta values */}
      {inner.theta && Object.keys(inner.theta).length > 0 && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-4">
          <h4 className="text-xs text-text-secondary uppercase mb-2">Paramètres Theta</h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(inner.theta).map(([key, val]) => (
              <span
                key={key}
                className="text-xs px-2 py-1 rounded-lg bg-accent-indigo/10 border border-accent-indigo/20 text-accent-indigo"
              >
                {key}: {val.toFixed(3)}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
