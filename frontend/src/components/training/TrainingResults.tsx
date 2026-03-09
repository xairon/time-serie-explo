import { MetricsRadar } from '@/components/charts/MetricsRadar'
import { METRIC_LABELS } from '@/lib/constants'

interface TrainingResultsProps {
  metrics: Record<string, number>
  mlflowRunId?: string | null
  className?: string
}

export function TrainingResults({ metrics, mlflowRunId, className = '' }: TrainingResultsProps) {
  return (
    <div className={`space-y-4 ${className}`}>
      <h3 className="text-sm font-semibold text-text-primary">Résultats</h3>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {Object.entries(metrics).map(([key, val]) => (
          <div key={key} className="bg-bg-hover rounded-lg p-3 text-center">
            <p className="text-[10px] text-text-secondary uppercase">
              {METRIC_LABELS[key] ?? key}
            </p>
            <p className="text-base font-bold text-text-primary">{val.toFixed(4)}</p>
          </div>
        ))}
      </div>

      <MetricsRadar metrics={metrics} className="h-[300px]" />

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
