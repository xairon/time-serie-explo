import { useState, useEffect, useCallback } from 'react'
import { Download } from 'lucide-react'
import { CFConfigForm } from '@/components/counterfactual/CFConfigForm'
import type { CFFormConfig } from '@/components/counterfactual/CFConfigForm'
import { CFResultView } from '@/components/counterfactual/CFResultView'
import { IPSPanel } from '@/components/counterfactual/IPSPanel'
import { PastasPanel } from '@/components/counterfactual/PastasPanel'
import { RadarPlot } from '@/components/charts/RadarPlot'
import { ConvergencePlot } from '@/components/counterfactual/ConvergencePlot'
import { useCounterfactualRun } from '@/hooks/useCounterfactual'
import { api } from '@/lib/api'
import type { CounterfactualResult } from '@/lib/types'

type RightTab = 'ips' | 'radar' | 'pastas' | 'convergence'

export default function CounterfactualPage() {
  const [rightTab, setRightTab] = useState<RightTab>('ips')
  const [modelId, setModelId] = useState<string | null>(null)
  const cfMutation = useCounterfactualRun()

  // The mutation returns {task_id, status, result, error} - initially result is null
  const [result, setResult] = useState<CounterfactualResult | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamError, setStreamError] = useState<string | null>(null)

  // After mutation succeeds with a task_id, connect to SSE to get the final result
  useEffect(() => {
    if (!cfMutation.data?.task_id) return
    if (cfMutation.data.status === 'done' && cfMutation.data.result) {
      // Synchronous result - no need to stream
      setResult(cfMutation.data)
      return
    }

    const taskId = cfMutation.data.task_id
    setIsStreaming(true)
    setStreamError(null)
    setResult({ task_id: taskId, status: 'pending', result: null, error: null })

    const es = api.counterfactual.stream(taskId)

    es.addEventListener('progress', (event) => {
      try {
        const data = JSON.parse(event.data)
        setResult((prev) => prev ? { ...prev, status: 'running', progress: data } : prev)
      } catch { /* ignore parse errors */ }
    })

    es.addEventListener('done', (event) => {
      try {
        const data = JSON.parse(event.data) as { status: string; error?: string; result?: CounterfactualResult['result'] }
        setResult({
          task_id: taskId,
          status: data.status,
          result: data.result ?? null,
          error: data.error ?? null,
        })
      } catch { /* ignore parse errors */ }
      setIsStreaming(false)
      es.close()
    })

    es.addEventListener('error', (event) => {
      try {
        const data = JSON.parse((event as MessageEvent).data) as { error?: string }
        setStreamError(data.error ?? 'Erreur inconnue')
        setResult((prev) =>
          prev ? { ...prev, status: 'error', error: data.error ?? 'Erreur inconnue' } : null,
        )
      } catch {
        setStreamError('Connexion au serveur perdue')
      }
      setIsStreaming(false)
      es.close()
    })

    es.onerror = () => {
      // EventSource auto-reconnects, but if it fails completely:
      if (es.readyState === EventSource.CLOSED) {
        setStreamError('Connexion au serveur perdue')
        setIsStreaming(false)
      }
    }

    return () => {
      es.close()
      setIsStreaming(false)
    }
  }, [cfMutation.data])

  const handleSubmit = useCallback(
    (config: CFFormConfig) => {
      setModelId(config.model_id)
      setResult(null)
      setStreamError(null)
      cfMutation.mutate({
        model_id: config.model_id,
        method: config.method,
        target_ips_class: config.target_ips_class,
        from_ips_class: config.from_ips_class,
        to_ips_class: config.to_ips_class,
        modifications: config.modifications,
        lambda_prox: config.lambda_prox,
        n_iter: config.n_iter,
        lr: config.lr,
        n_trials: config.n_trials,
        k_sigma: config.k_sigma,
        lambda_smooth: config.lambda_smooth,
        cc_rate: config.cc_rate,
        device: config.device,
        seed: config.seed,
      })
    },
    [cfMutation],
  )

  const handleExport = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `counterfactual_${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const isLoading = cfMutation.isPending || isStreaming
  const innerResult = result?.result ?? null
  const hasConvergence = innerResult?.convergence && innerResult.convergence.length > 0

  const rightTabs: { key: RightTab; label: string }[] = [
    { key: 'ips', label: 'IPS' },
    { key: 'radar', label: 'Radar' },
    { key: 'convergence', label: 'Convergence' },
    { key: 'pastas', label: 'Pastas' },
  ]

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Analyse contrefactuelle</h1>
          <p className="text-sm text-text-secondary">
            Simulation de scenarios alternatifs pour comprendre l'impact des covariables
          </p>
        </div>
        {result?.result && (
          <button
            onClick={handleExport}
            className="flex items-center gap-2 bg-bg-hover text-text-primary px-4 py-2 rounded-lg border border-white/10 hover:bg-bg-hover/80 transition-colors text-sm"
          >
            <Download className="w-4 h-4" />
            Exporter JSON
          </button>
        )}
      </div>

      {(cfMutation.isError || streamError || result?.error) && (
        <div className="bg-accent-red/10 border border-accent-red/20 rounded-xl p-4 flex items-center justify-between">
          <p className="text-sm text-accent-red">
            Erreur : {streamError ?? result?.error ?? (cfMutation.error as Error)?.message}
          </p>
          <button
            onClick={() => {
              cfMutation.reset()
              setStreamError(null)
              setResult(null)
            }}
            className="text-xs text-accent-cyan hover:underline"
          >
            Fermer
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Config form */}
        <div className="lg:col-span-3">
          <div className="bg-bg-card rounded-xl border border-white/5 p-5">
            <CFConfigForm onSubmit={handleSubmit} isPending={isLoading} />
          </div>
        </div>

        {/* Center: Results */}
        <div className="lg:col-span-5">
          <CFResultView
            result={result}
            isLoading={isLoading}
          />
        </div>

        {/* Right: Panels */}
        <div className="lg:col-span-4">
          <div className="bg-bg-card rounded-xl border border-white/5 p-5">
            {/* Right tab bar */}
            <div className="flex border-b border-white/10 mb-4">
              {rightTabs.map((t) => (
                <button
                  key={t.key}
                  onClick={() => setRightTab(t.key)}
                  className={`px-3 py-1.5 text-xs transition-colors ${
                    rightTab === t.key
                      ? 'border-b-2 border-accent-cyan text-accent-cyan'
                      : 'text-text-secondary hover:text-text-primary'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {rightTab === 'ips' && <IPSPanel modelId={modelId} />}

            {rightTab === 'radar' && (
              innerResult?.theta && Object.keys(innerResult.theta).length > 0 ? (
                <RadarPlot theta={innerResult.theta} className="h-[350px]" />
              ) : (
                <p className="text-xs text-text-secondary italic text-center py-8">
                  Les parametres theta apparaitront ici apres la generation du contrefactuel.
                </p>
              )
            )}

            {rightTab === 'convergence' && (
              hasConvergence ? (
                <ConvergencePlot lossHistory={innerResult!.convergence!} className="h-[350px]" />
              ) : (
                <p className="text-xs text-text-secondary italic text-center py-8">
                  La courbe de convergence apparaitra ici apres la generation du contrefactuel.
                </p>
              )
            )}

            {rightTab === 'pastas' && <PastasPanel />}
          </div>
        </div>
      </div>
    </div>
  )
}
