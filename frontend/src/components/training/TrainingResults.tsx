import { useState, useMemo } from 'react'
import { MetricsRadar } from '@/components/charts/MetricsRadar'
import { METRIC_LABELS } from '@/lib/constants'

interface TrainingResultsProps {
  metrics: Record<string, number>
  metricsSliding?: Record<string, number> | null
  mlflowRunId?: string | null
  className?: string
}

/** Filter out system/* and sliding/* prefixed metrics from display */
function filterDisplayMetrics(metrics: Record<string, number>): Record<string, number> {
  const filtered: Record<string, number> = {}
  for (const [key, val] of Object.entries(metrics)) {
    if (key.startsWith('system/') || key.startsWith('sliding/') || key.startsWith('system_')) {
      continue
    }
    filtered[key] = val
  }
  return filtered
}

export function TrainingResults({ metrics, metricsSliding, mlflowRunId, className = '' }: TrainingResultsProps) {
  const [showSliding, setShowSliding] = useState(false)

  const displayMetrics = useMemo(() => filterDisplayMetrics(metrics), [metrics])
  const displaySlidingMetrics = useMemo(
    () => (metricsSliding ? filterDisplayMetrics(metricsSliding) : null),
    [metricsSliding],
  )

  const activeMetrics = showSliding && displaySlidingMetrics ? displaySlidingMetrics : displayMetrics

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-text-primary">Resultats</h3>
        {displaySlidingMetrics && Object.keys(displaySlidingMetrics).length > 0 && (
          <div className="flex items-center gap-1 bg-bg-hover rounded-lg p-0.5">
            <button
              type="button"
              onClick={() => setShowSliding(false)}
              className={`px-3 py-1 rounded-md text-xs transition-colors ${
                !showSliding
                  ? 'bg-accent-cyan/20 text-accent-cyan font-medium'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              Fenetre unique
            </button>
            <button
              type="button"
              onClick={() => setShowSliding(true)}
              className={`px-3 py-1 rounded-md text-xs transition-colors ${
                showSliding
                  ? 'bg-accent-cyan/20 text-accent-cyan font-medium'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              Fenetre glissante
            </button>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {Object.entries(activeMetrics).map(([key, val]) => (
          <div key={key} className="bg-bg-hover rounded-lg p-3 text-center">
            <p className="text-[10px] text-text-secondary uppercase">
              {METRIC_LABELS[key] ?? key}
            </p>
            <p className="text-base font-bold text-text-primary">{val != null ? val.toFixed(4) : '—'}</p>
          </div>
        ))}
      </div>

      <MetricsRadar metrics={activeMetrics} className="h-[300px]" />

      {mlflowRunId && (
        <a
          href={`/mlflow/#/experiments/0/runs/${mlflowRunId}`}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-block text-xs text-accent-cyan hover:underline"
        >
          Voir dans MLflow
        </a>
      )}
    </div>
  )
}
