import { LossPlot } from '@/components/charts/LossPlot'
import type { TrainingMetrics } from '@/lib/types'

interface TrainingMonitorProps {
  metrics: TrainingMetrics | null
  trainLossHistory: number[]
  valLossHistory: number[]
  status: 'idle' | 'connected' | 'done' | 'error'
  error: string | null
  onCancel: () => void
}

export function TrainingMonitor({
  metrics,
  trainLossHistory,
  valLossHistory,
  status,
  error,
  onCancel,
}: TrainingMonitorProps) {
  const progress = metrics
    ? Math.round((metrics.current_epoch / metrics.total_epochs) * 100)
    : 0

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-text-primary">Moniteur d'entraînement</h3>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              status === 'connected'
                ? 'bg-accent-green animate-pulse'
                : status === 'done'
                  ? 'bg-accent-cyan'
                  : status === 'error'
                    ? 'bg-accent-red'
                    : 'bg-text-secondary'
            }`}
          />
          <span className="text-xs text-text-secondary">
            {status === 'connected'
              ? 'En cours'
              : status === 'done'
                ? 'Terminé'
                : status === 'error'
                  ? 'Erreur'
                  : 'En attente'}
          </span>
        </div>
      </div>

      {/* Progress bar */}
      {metrics && (
        <div>
          <div className="flex items-center justify-between text-xs text-text-secondary mb-1">
            <span>
              Epoch {metrics.current_epoch} / {metrics.total_epochs}
            </span>
            <span>{progress}%</span>
          </div>
          <div className="h-2 bg-bg-hover rounded-full overflow-hidden">
            <div
              className="h-full bg-accent-cyan rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Metric cards */}
      {metrics && (
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-bg-hover rounded-lg p-3 text-center">
            <p className="text-[10px] text-text-secondary uppercase">Train Loss</p>
            <p className="text-lg font-bold text-text-primary">
              {metrics.train_loss?.toFixed(5) ?? '—'}
            </p>
          </div>
          <div className="bg-bg-hover rounded-lg p-3 text-center">
            <p className="text-[10px] text-text-secondary uppercase">Val Loss</p>
            <p className="text-lg font-bold text-text-primary">
              {metrics.val_loss?.toFixed(5) ?? '—'}
            </p>
          </div>
          <div className="bg-bg-hover rounded-lg p-3 text-center">
            <p className="text-[10px] text-text-secondary uppercase">Best Val</p>
            <p className="text-lg font-bold text-accent-green">
              {metrics.best_val_loss?.toFixed(5) ?? '—'}
            </p>
          </div>
        </div>
      )}

      {/* Loss plot */}
      {trainLossHistory.length > 0 && (
        <LossPlot
          trainLoss={trainLossHistory}
          valLoss={valLossHistory}
          className="h-[250px]"
        />
      )}

      {/* Error */}
      {error && (
        <p className="text-xs text-accent-red bg-accent-red/10 p-2 rounded-lg">{error}</p>
      )}

      {/* Cancel */}
      {status === 'connected' && (
        <button
          onClick={onCancel}
          className="w-full bg-bg-hover text-text-primary px-4 py-2 rounded-lg border border-white/10 hover:bg-accent-red/10 hover:text-accent-red transition-colors text-sm"
        >
          Annuler
        </button>
      )}
    </div>
  )
}
