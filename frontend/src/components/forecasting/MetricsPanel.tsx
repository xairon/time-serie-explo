import { METRIC_LABELS } from '@/lib/constants'

interface MetricsPanelProps {
  metrics: Record<string, number>
  className?: string
}

// Thresholds: lower is better for error metrics, higher is better for efficiency metrics
const isGood = (key: string, value: number): boolean | null => {
  const lower = ['MAE', 'RMSE', 'sMAPE', 'WAPE', 'NRMSE']
  const higher = ['NSE', 'KGE', 'Dir_Acc']
  if (lower.includes(key)) return value < 0.1
  if (higher.includes(key)) return value > 0.7
  return null
}

export function MetricsPanel({ metrics, className = '' }: MetricsPanelProps) {
  return (
    <div className={`space-y-2 ${className}`}>
      <h4 className="text-sm font-semibold text-text-primary">Métriques</h4>
      <div className="grid grid-cols-2 gap-2">
        {Object.entries(metrics).map(([key, val]) => {
          const good = isGood(key, val)
          const colorClass =
            good === true
              ? 'text-accent-green'
              : good === false
                ? 'text-accent-red'
                : 'text-text-primary'

          return (
            <div
              key={key}
              className="bg-bg-card rounded-lg p-3 border border-white/5"
            >
              <p className="text-[10px] text-text-secondary uppercase mb-1">
                {METRIC_LABELS[key] ?? key}
              </p>
              <p className={`text-lg font-bold ${colorClass}`}>{val != null ? val.toFixed(4) : '—'}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
