import { useState, useCallback, useEffect, useRef } from 'react'
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

type TrainingPhase = 'idle' | 'preparing' | 'training' | 'completed' | 'error'

export default function TrainingPage() {
  const [taskId, setTaskId] = useState<string | null>(null)
  const [sseUrl, setSseUrl] = useState<string | null>(null)
  const [trainLossHistory, setTrainLossHistory] = useState<number[]>([])
  const [valLossHistory, setValLossHistory] = useState<number[]>([])
  const [finalMetrics, setFinalMetrics] = useState<Record<string, number> | null>(null)

  const [logs, setLogs] = useState<string[]>([])
  const [phase, setPhase] = useState<TrainingPhase>('idle')
  const logsEndRef = useRef<HTMLDivElement>(null)

  const startMutation = useStartTraining()
  const stopMutation = useStopTraining()
  const { data: models } = useModels()

  const sse = useSSE<TrainingMetrics>(sseUrl)

  // Derive phase from SSE state
  useEffect(() => {
    if (sse.status === 'error') {
      setPhase('error')
    } else if (sse.status === 'done') {
      setPhase('completed')
    } else if (sse.status === 'connected' && sse.data && sse.data.current_epoch > 0) {
      setPhase('training')
    } else if (sse.status === 'connected') {
      setPhase('preparing')
    }
  }, [sse.status, sse.data])

  // Add log entries from SSE metrics updates
  useEffect(() => {
    if (!sse.data) return
    const { current_epoch, total_epochs, train_loss, val_loss, best_val_loss, status } = sse.data
    const timestamp = new Date().toLocaleTimeString('fr-FR')
    let msg = `[${timestamp}] `
    if (status) {
      msg += status
    } else if (current_epoch > 0) {
      msg += `Epoch ${current_epoch}/${total_epochs}`
      if (train_loss !== null) msg += ` | train_loss: ${train_loss.toFixed(5)}`
      if (val_loss !== null) msg += ` | val_loss: ${val_loss.toFixed(5)}`
      if (best_val_loss !== null) msg += ` | best: ${best_val_loss.toFixed(5)}`
    } else {
      msg += 'Preparation des donnees...'
    }
    setLogs((prev) => [...prev, msg])
  }, [sse.data?.current_epoch, sse.data?.status])

  // Log SSE state changes
  useEffect(() => {
    const timestamp = new Date().toLocaleTimeString('fr-FR')
    if (sse.status === 'connected') {
      setLogs((prev) => [...prev, `[${timestamp}] Connexion SSE etablie`])
    } else if (sse.status === 'done') {
      setLogs((prev) => [...prev, `[${timestamp}] Entrainement termine`])
    } else if (sse.status === 'error') {
      setLogs((prev) => [...prev, `[${timestamp}] Erreur: ${sse.error ?? 'connexion perdue'}`])
    }
  }, [sse.status])

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

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
      setLogs([])
      setPhase('preparing')
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

      {/* Phase indicator & Logs */}
      {phase !== 'idle' && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-5 space-y-4">
          {/* Phase steps */}
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-text-primary mr-4">Phase :</h3>
            {(['preparing', 'training', 'completed'] as const).map((p, i) => {
              const labels = { preparing: 'Preparation', training: 'Entrainement', completed: 'Termine' }
              const isActive = phase === p
              const isPast = (phase === 'training' && p === 'preparing') ||
                (phase === 'completed' && (p === 'preparing' || p === 'training'))
              return (
                <div key={p} className="flex items-center gap-2">
                  {i > 0 && <div className={`w-8 h-px ${isPast ? 'bg-accent-cyan' : 'bg-white/10'}`} />}
                  <div className="flex items-center gap-1.5">
                    <div
                      className={`w-3 h-3 rounded-full border-2 ${
                        isActive
                          ? 'border-accent-cyan bg-accent-cyan/30 animate-pulse'
                          : isPast
                            ? 'border-accent-cyan bg-accent-cyan'
                            : phase === 'error' && p !== 'completed'
                              ? 'border-accent-red bg-accent-red/30'
                              : 'border-white/20 bg-transparent'
                      }`}
                    />
                    <span className={`text-xs ${isActive ? 'text-accent-cyan font-medium' : isPast ? 'text-text-primary' : 'text-text-secondary'}`}>
                      {labels[p]}
                    </span>
                  </div>
                </div>
              )
            })}
            {phase === 'error' && (
              <div className="flex items-center gap-1.5 ml-4">
                <div className="w-3 h-3 rounded-full border-2 border-accent-red bg-accent-red/30" />
                <span className="text-xs text-accent-red font-medium">Erreur</span>
              </div>
            )}
          </div>

          {/* Scrollable logs */}
          <div>
            <h4 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">Logs</h4>
            <div className="bg-bg-primary rounded-lg border border-white/5 p-3 max-h-48 overflow-y-auto font-mono text-[11px] text-text-secondary space-y-0.5">
              {logs.length === 0 ? (
                <p className="text-text-secondary/50">En attente des logs...</p>
              ) : (
                logs.map((log, i) => (
                  <div key={i} className={`${log.includes('Erreur') || log.includes('error') ? 'text-accent-red' : log.includes('termine') || log.includes('Termine') ? 'text-accent-green' : ''}`}>
                    {log}
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>
      )}

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
