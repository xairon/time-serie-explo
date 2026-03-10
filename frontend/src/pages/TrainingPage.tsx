import { useState, useCallback } from 'react'
import { API_BASE } from '@/lib/constants'
import { api } from '@/lib/api'
import { useSSE } from '@/hooks/useSSE'
import { useStartTraining, useStopTraining } from '@/hooks/useTraining'
import { useModels } from '@/hooks/useModels'
import { ModelConfigForm } from '@/components/training/ModelConfigForm'
import { TrainingMonitor } from '@/components/training/TrainingMonitor'
import { TrainingResults } from '@/components/training/TrainingResults'
import type { TrainingConfig, TrainingMetrics } from '@/lib/types'
import { METRIC_LABELS } from '@/lib/constants'

export default function TrainingPage() {
  const [taskId, setTaskId] = useState<string | null>(null)
  const [sseUrl, setSseUrl] = useState<string | null>(null)
  const [trainLossHistory, setTrainLossHistory] = useState<number[]>([])
  const [valLossHistory, setValLossHistory] = useState<number[]>([])
  const [finalMetrics, setFinalMetrics] = useState<Record<string, number> | null>(null)

  const startMutation = useStartTraining()
  const stopMutation = useStopTraining()
  const { data: models } = useModels()

  const sse = useSSE<TrainingMetrics>(sseUrl)

  // Track loss history when SSE data updates
  const lastEpoch = sse.data?.current_epoch ?? 0
  if (sse.data) {
    if (
      sse.data.train_loss !== null &&
      trainLossHistory.length < lastEpoch
    ) {
      setTrainLossHistory((prev) => [...prev, sse.data!.train_loss!])
    }
    if (
      sse.data.val_loss !== null &&
      valLossHistory.length < lastEpoch
    ) {
      setValLossHistory((prev) => [...prev, sse.data!.val_loss!])
    }
  }

  // Detect training done
  if (sse.status === 'done' && !finalMetrics && taskId) {
    void fetch(`${API_BASE}/training/${taskId}/status`)
      .then((r) => r.json())
      .then((data: { metrics?: Record<string, number> }) => {
        if (data.metrics) setFinalMetrics(data.metrics)
      })
      .catch(() => {
        /* ignore */
      })
  }

  const handleStart = useCallback(
    (config: TrainingConfig) => {
      setTrainLossHistory([])
      setValLossHistory([])
      setFinalMetrics(null)
      startMutation.mutate(config, {
        onSuccess: (data) => {
          setTaskId(data.task_id)
          setSseUrl(`${API_BASE}/training/${data.task_id}/stream`)
        },
      })
    },
    [startMutation],
  )

  const handleCancel = useCallback(() => {
    if (taskId) {
      stopMutation.mutate(taskId)
      setSseUrl(null)
    }
  }, [taskId, stopMutation])

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-1">Entraînement</h1>
        <p className="text-sm text-text-secondary">
          Configurer et lancer l'entraînement de modèles de prévision
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Config form */}
        <div className="bg-bg-card rounded-xl border border-white/5 p-5">
          <ModelConfigForm
            onSubmit={handleStart}
            isPending={startMutation.isPending}
          />
          {startMutation.isError && (
            <p className="mt-3 text-xs text-accent-red">
              Erreur : {(startMutation.error as Error).message}
            </p>
          )}
        </div>

        {/* Right: Monitor */}
        <div className="bg-bg-card rounded-xl border border-white/5 p-5">
          {sseUrl || sse.status !== 'idle' ? (
            <TrainingMonitor
              metrics={sse.data}
              trainLossHistory={trainLossHistory}
              valLossHistory={valLossHistory}
              status={sse.status}
              error={sse.error}
              onCancel={handleCancel}
            />
          ) : (
            <div className="flex items-center justify-center h-full min-h-[300px]">
              <p className="text-sm text-text-secondary">
                Configurez le modèle et lancez l'entraînement pour voir le moniteur en temps réel.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      {finalMetrics && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-5">
          <TrainingResults metrics={finalMetrics} />
        </div>
      )}

      {/* History */}
      {models && models.length > 0 && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-5">
          <h3 className="text-sm font-semibold text-text-primary mb-3">
            Historique des entraînements
          </h3>
          <div className="overflow-x-auto rounded-lg border border-white/5">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg-hover">
                  {['Nom', 'Architecture', 'Station', 'Date', ...Object.keys(METRIC_LABELS).slice(0, 4), ''].map(
                    (h) => (
                      <th
                        key={h}
                        className="px-3 py-2 text-left text-xs font-medium text-text-secondary uppercase tracking-wide whitespace-nowrap"
                      >
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.model_id} className="border-t border-white/5 hover:bg-bg-hover/50">
                    <td className="px-3 py-1.5 text-text-primary">{m.model_name}</td>
                    <td className="px-3 py-1.5 text-text-primary">{m.model_type}</td>
                    <td className="px-3 py-1.5 text-text-secondary">{m.primary_station ?? '—'}</td>
                    <td className="px-3 py-1.5 text-text-secondary text-xs">
                      {new Date(m.created_at).toLocaleDateString('fr-FR')}
                    </td>
                    {Object.keys(METRIC_LABELS)
                      .slice(0, 4)
                      .map((key) => (
                        <td key={key} className="px-3 py-1.5 text-text-primary text-xs">
                          {m.metrics[key]?.toFixed(4) ?? '—'}
                        </td>
                      ))}
                    <td className="px-3 py-1.5">
                      <a
                        href={api.models.downloadUrl(m.model_id)}
                        className="text-xs text-accent-cyan hover:underline"
                        download
                      >
                        Télécharger
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
