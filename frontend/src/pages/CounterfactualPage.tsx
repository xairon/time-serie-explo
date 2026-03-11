import { useState, useEffect, useCallback, useMemo } from 'react'
import { Download } from 'lucide-react'
import { ModelSelector } from '@/components/forecasting/ModelSelector'
import { CFConfigForm } from '@/components/counterfactual/CFConfigForm'
import type { CFFormData } from '@/components/counterfactual/CFConfigForm'
import { CFResultView } from '@/components/counterfactual/CFResultView'
import { TestSetOverview } from '@/components/charts/TestSetOverview'
import IPSMonthlyGrid from '@/components/counterfactual/IPSMonthlyGrid'
import { useCounterfactualRun, useIPSBounds, useIPSReference } from '@/hooks/useCounterfactual'
import { useModelTestInfo } from '@/hooks/useModels'
import { api } from '@/lib/api'
import type { CounterfactualResult } from '@/lib/types'

export default function CounterfactualPage() {
  // Core state
  const [modelId, setModelId] = useState<string>('')
  const [startIdx, setStartIdx] = useState(0)
  const [ipsWindow, setIpsWindow] = useState(1)

  // Fetch model test set info
  const { data: testInfo } = useModelTestInfo(modelId || null)
  const { data: ipsBoundsData } = useIPSBounds(modelId || null, ipsWindow)
  const { data: ipsRef } = useIPSReference(modelId || null, ipsWindow)

  // CF generation
  const cfMutation = useCounterfactualRun()
  const [result, setResult] = useState<CounterfactualResult | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamError, setStreamError] = useState<string | null>(null)

  // Reset state on model change
  useEffect(() => {
    setStartIdx(0)
    setResult(null)
    setStreamError(null)
  }, [modelId])

  // Set default start_idx to middle of valid range
  useEffect(() => {
    if (testInfo && startIdx === 0) {
      const mid = Math.floor((testInfo.valid_start_idx + testInfo.valid_end_idx) / 2)
      setStartIdx(mid)
    }
  }, [testInfo, startIdx])

  // Compute window dates from startIdx + testInfo
  const windowInfo = useMemo(() => {
    if (!testInfo) return null
    const L = testInfo.input_chunk_length
    const H = testInfo.output_chunk_length
    const dates = testInfo.test_dates

    const predStartDate = dates[startIdx] ?? ''
    const predEndIdx = Math.min(startIdx + H - 1, dates.length - 1)
    const predEndDate = dates[predEndIdx] ?? ''

    // Context = L days before predStart in the full dataset
    // Since test_dates starts from the test set, context may extend before test set
    const contextStartIdx = Math.max(0, startIdx - L)
    const contextStartDate = dates[contextStartIdx] ?? ''
    const contextEndDate = dates[Math.max(0, startIdx - 1)] ?? predStartDate

    return {
      contextStart: contextStartDate,
      contextEnd: contextEndDate,
      predStart: predStartDate,
      predEnd: predEndDate,
      predStartIdx: startIdx,
      predEndIdx,
      L,
      H,
    }
  }, [testInfo, startIdx])

  // Extract current prediction window dates/values for IPS classification
  const currentWindowData = useMemo(() => {
    if (!testInfo || !windowInfo) return null
    const dates = testInfo.test_dates.slice(windowInfo.predStartIdx, windowInfo.predEndIdx + 1)
    const values = testInfo.test_values.slice(windowInfo.predStartIdx, windowInfo.predEndIdx + 1)
    return { dates, values: values.filter((v): v is number => v !== null) }
  }, [testInfo, windowInfo])

  // SSE streaming for CF result
  useEffect(() => {
    if (!cfMutation.data?.task_id) return
    if (cfMutation.data.status === 'done' && cfMutation.data.result) {
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
      } catch { /* ignore */ }
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
      } catch { /* ignore */ }
      setIsStreaming(false)
      es.close()
    })

    es.addEventListener('error', (event) => {
      try {
        const data = JSON.parse((event as MessageEvent).data) as { error?: string }
        setStreamError(data.error ?? 'Erreur inconnue')
        setResult((prev) => prev ? { ...prev, status: 'error', error: data.error ?? 'Erreur inconnue' } : null)
      } catch {
        setStreamError('Connexion au serveur perdue')
      }
      setIsStreaming(false)
      es.close()
    })

    es.onerror = () => {
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
    (config: CFFormData) => {
      setResult(null)
      setStreamError(null)
      cfMutation.mutate({
        model_id: config.model_id,
        method: config.method,
        target_ips_class: config.target_ips_class,
        from_ips_class: config.from_ips_class,
        to_ips_class: config.to_ips_class,
        start_idx: config.start_idx,
        lambda_prox: config.lambda_prox,
        n_iter: config.n_iter,
        lr: config.lr,
        n_trials: config.n_trials,
        k_sigma: config.k_sigma,
        lambda_smooth: config.lambda_smooth,
        cc_rate: config.cc_rate,
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

  // Build ref_stats as Record<string, [number, number]> for IPSMonthlyGrid
  const refStatsForGrid = useMemo(() => {
    if (!ipsRef?.ref_stats) return {}
    const out: Record<string, [number, number]> = {}
    for (const [k, v] of Object.entries(ipsRef.ref_stats)) {
      if (Array.isArray(v) && v.length >= 2) {
        out[k] = [v[0] as number, v[1] as number]
      }
    }
    return out
  }, [ipsRef])

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
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

      {/* Error banner */}
      {(cfMutation.isError || streamError || result?.error) && (
        <div className="bg-accent-red/10 border border-accent-red/20 rounded-xl p-4 flex items-center justify-between">
          <p className="text-sm text-accent-red">
            Erreur : {streamError ?? result?.error ?? (cfMutation.error as Error)?.message}
          </p>
          <button
            onClick={() => { cfMutation.reset(); setStreamError(null); setResult(null) }}
            className="text-xs text-accent-cyan hover:underline"
          >
            Fermer
          </button>
        </div>
      )}

      {/* Model selector */}
      <div className="bg-bg-card rounded-xl border border-white/5 p-5">
        <ModelSelector value={modelId} onChange={setModelId} />
      </div>

      {/* Test set overview with IPS bands + window slider */}
      {modelId && testInfo && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-text-primary">Set de test — selection de la fenetre</h3>
            {windowInfo && (
              <p className="text-xs text-text-secondary">
                Contexte: {windowInfo.contextStart} → {windowInfo.contextEnd} ({windowInfo.L}j)
                {' | '}
                Prediction: {windowInfo.predStart} → {windowInfo.predEnd} ({windowInfo.H}j)
              </p>
            )}
          </div>

          {/* Test set chart with window highlight */}
          <TestSetOverview
            testDates={testInfo.test_dates}
            testValues={testInfo.test_values}
            sliderIdx={startIdx}
            inputChunkLength={testInfo.input_chunk_length}
            outputChunkLength={testInfo.output_chunk_length}
            className="h-[300px]"
          />

          {/* Window slider */}
          <div className="flex items-center gap-4">
            <span className="text-xs text-text-secondary shrink-0">Position</span>
            <input
              type="range"
              min={testInfo.valid_start_idx}
              max={testInfo.valid_end_idx}
              value={startIdx}
              onChange={(e) => setStartIdx(Number(e.target.value))}
              className="flex-1 accent-accent-cyan h-1.5"
            />
            <span className="text-xs text-text-primary font-mono w-12 text-right">{startIdx}</span>
          </div>

          {/* IPS classification of current window (observed data) */}
          {currentWindowData && currentWindowData.dates.length > 0 && Object.keys(refStatsForGrid).length > 0 && (
            <IPSMonthlyGrid
              predDates={currentWindowData.dates}
              predValues={currentWindowData.values}
              gtValues={currentWindowData.values}
              refStats={refStatsForGrid}
              ipsLabels={ipsBoundsData?.classes ?? {}}
              ipsColors={ipsBoundsData?.colors ?? {}}
              label="Classification IPS de la fenetre selectionnee"
            />
          )}
        </div>
      )}

      {/* Main content: Config sidebar + Results */}
      {modelId && testInfo && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Config */}
          <div className="lg:col-span-3">
            <div className="bg-bg-card rounded-xl border border-white/5 p-5">
              <h3 className="text-sm font-semibold text-text-primary mb-4">Configuration</h3>
              <CFConfigForm
                modelId={modelId}
                startIdx={startIdx}
                ipsWindow={ipsWindow}
                onIpsWindowChange={setIpsWindow}
                onSubmit={handleSubmit}
                isPending={isLoading}
              />
            </div>
          </div>

          {/* Right: Results */}
          <div className="lg:col-span-9 space-y-4">
            <CFResultView
              result={result}
              isLoading={isLoading}
            />

            {/* IPS Monthly Grid — prediction */}
            {innerResult && Object.keys(refStatsForGrid).length > 0 && (
              <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                <h4 className="text-sm font-semibold text-text-primary mb-3">Classification IPS mensuelle</h4>
                <IPSMonthlyGrid
                  predDates={innerResult.dates}
                  predValues={innerResult.counterfactual}
                  gtValues={innerResult.original}
                  refStats={refStatsForGrid}
                  ipsLabels={ipsBoundsData?.classes ?? {}}
                  ipsColors={ipsBoundsData?.colors ?? {}}
                  label="Contrefactuel vs Original"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* No model selected placeholder */}
      {!modelId && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-12 text-center">
          <p className="text-text-secondary text-sm">
            Selectionnez un modele entraine pour commencer l'analyse contrefactuelle.
          </p>
        </div>
      )}
    </div>
  )
}
