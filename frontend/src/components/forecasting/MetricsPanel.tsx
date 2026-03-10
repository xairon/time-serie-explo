import { useMemo } from 'react'
import { METRIC_LABELS } from '@/lib/constants'

interface MetricsPanelProps {
  metrics: Record<string, number>
  actuals?: (number | null)[]
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

/** Compute IQR from actual values (ignoring nulls) */
function computeIQR(values: (number | null)[]): number | null {
  const sorted = values.filter((v): v is number => v != null).sort((a, b) => a - b)
  if (sorted.length < 4) return null
  const q25Idx = Math.floor(sorted.length * 0.25)
  const q75Idx = Math.floor(sorted.length * 0.75)
  const iqr = sorted[q75Idx] - sorted[q25Idx]
  return iqr > 0 ? iqr : null
}

export function MetricsPanel({ metrics, actuals, className = '' }: MetricsPanelProps) {
  const iqr = useMemo(() => (actuals ? computeIQR(actuals) : null), [actuals])

  const mae = metrics['MAE'] ?? metrics['mae']

  return (
    <div className={`space-y-2 ${className}`}>
      <h4 className="text-sm font-semibold text-text-primary">Metriques</h4>
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

      {/* Relative error context */}
      {iqr != null && mae != null && (
        <div className="bg-bg-card rounded-lg border border-white/5 p-3">
          <p className="text-xs text-text-secondary">
            MAE ≈ <span className="text-text-primary font-medium">{((mae / iqr) * 100).toFixed(1)}%</span> de l&apos;echelle (IQR = {iqr.toFixed(4)})
          </p>
        </div>
      )}
    </div>
  )
}
