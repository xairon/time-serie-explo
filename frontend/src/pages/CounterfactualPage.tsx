import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { Download, ShieldCheck } from 'lucide-react'
import { ModelSelector } from '@/components/forecasting/ModelSelector'
import { CFTargetSelector } from '@/components/counterfactual/CFTargetSelector'
import type { CFTargetData } from '@/components/counterfactual/CFTargetSelector'
import { CFResultView } from '@/components/counterfactual/CFResultView'
import { TestSetOverview } from '@/components/charts/TestSetOverview'
import IPSMonthlyGrid from '@/components/counterfactual/IPSMonthlyGrid'
import { useIPSBounds, useIPSReference } from '@/hooks/useCounterfactual'
import { useForecastSingle } from '@/hooks/useForecasting'
import { useModelTestInfo } from '@/hooks/useModels'
import { api } from '@/lib/api'
import { computeMonthlyIps, computeQualityGate, type MonthIps, type QualityVerdict } from '@/lib/ips'
import type { CounterfactualResult, PastasValidationResult } from '@/lib/types'

const METHODS = ['physcf', 'comet'] as const
const METHOD_LABELS: Record<string, string> = { physcf: 'PhysCF', comet: 'COMET' }

/** Start SSE stream for a CF task, with generation guard to prevent stale overwrites */
function startCFStream(
  taskId: string,
  method: string,
  setResults: React.Dispatch<React.SetStateAction<Record<string, CounterfactualResult | null>>>,
  setStreaming: React.Dispatch<React.SetStateAction<Record<string, boolean>>>,
  generation: number,
  generationRef: React.MutableRefObject<number>,
): EventSource {
  setStreaming((prev) => ({ ...prev, [method]: true }))
  setResults((prev) => ({
    ...prev,
    [method]: { task_id: taskId, status: 'pending', result: null, error: null },
  }))

  const es = api.counterfactual.stream(taskId)

  es.addEventListener('progress', (event) => {
    if (generationRef.current !== generation) { es.close(); return }
    try {
      const data = JSON.parse(event.data)
      setResults((prev) => {
        const cur = prev[method]
        return cur ? { ...prev, [method]: { ...cur, status: 'running', progress: data } } : prev
      })
    } catch { /* ignore */ }
  })

  es.addEventListener('done', (event) => {
    if (generationRef.current !== generation) { es.close(); return }
    try {
      const data = JSON.parse(event.data) as { status: string; error?: string; result?: CounterfactualResult['result'] }
      setResults((prev) => ({
        ...prev,
        [method]: { task_id: taskId, status: data.status, result: data.result ?? null, error: data.error ?? null },
      }))
    } catch { /* ignore */ }
    setStreaming((prev) => ({ ...prev, [method]: false }))
    es.close()
  })

  es.addEventListener('error', (event) => {
    if (generationRef.current !== generation) { es.close(); return }
    try {
      const data = JSON.parse((event as MessageEvent).data) as { error?: string }
      setResults((prev) => ({
        ...prev,
        [method]: { task_id: taskId, status: 'error', result: null, error: data.error ?? 'Erreur inconnue' },
      }))
    } catch {
      setResults((prev) => ({
        ...prev,
        [method]: { task_id: taskId, status: 'error', result: null, error: 'Connexion au serveur perdue' },
      }))
    }
    setStreaming((prev) => ({ ...prev, [method]: false }))
    es.close()
  })

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) {
      setStreaming((prev) => ({ ...prev, [method]: false }))
    }
  }

  return es
}

export default function CounterfactualPage() {
  // Core state
  const [modelId, setModelId] = useState<string>('')
  const [startIdx, setStartIdx] = useState(0)
  const [sliderDraft, setSliderDraft] = useState(0)

  // Fetch model test set info (IPS always monthly = window 1)
  const { data: testInfo } = useModelTestInfo(modelId || null)
  const { data: ipsBoundsData } = useIPSBounds(modelId || null, 1)
  const { data: ipsRef } = useIPSReference(modelId || null, 1)

  // Forecast single window
  const forecastMutation = useForecastSingle()

  // CF generation — dual method
  const [results, setResults] = useState<Record<string, CounterfactualResult | null>>({})
  const [streaming, setStreaming] = useState<Record<string, boolean>>({})
  const [activeTab, setActiveTab] = useState<string>('physcf')
  const eventSourcesRef = useRef<EventSource[]>([])
  const generationRef = useRef(0)

  // Pastas validation
  const [pastasResults, setPastasResults] = useState<Record<string, PastasValidationResult | null>>({})
  const [pastasLoading, setPastasLoading] = useState<Record<string, boolean>>({})

  const isLoading = METHODS.some((m) => streaming[m])
  const hasAnyResult = METHODS.some((m) => results[m]?.result)

  // Reset state on model change
  useEffect(() => {
    setStartIdx(0)
    setSliderDraft(0)
    setResults({})
    setStreaming({})
    setPastasResults({})
    setPastasLoading({})
  }, [modelId])

  // Set default start_idx to middle of valid range
  useEffect(() => {
    if (testInfo && startIdx === 0) {
      const mid = Math.floor((testInfo.valid_start_idx + testInfo.valid_end_idx) / 2)
      setStartIdx(mid)
      setSliderDraft(mid)
    }
  }, [testInfo, startIdx])

  // Fetch forecast for current window position
  useEffect(() => {
    if (modelId && testInfo && testInfo.test_dates[startIdx]) {
      forecastMutation.mutate({ model_id: modelId, start_date: testInfo.test_dates[startIdx] })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [startIdx, modelId, testInfo])

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      eventSourcesRef.current.forEach((es) => es.close())
    }
  }, [])

  // Compute window dates from startIdx + testInfo
  const windowInfo = useMemo(() => {
    if (!testInfo) return null
    const L = testInfo.input_chunk_length
    const H = testInfo.output_chunk_length
    const dates = testInfo.test_dates

    const predStartDate = dates[startIdx] ?? ''
    const predEndIdx = Math.min(startIdx + H - 1, dates.length - 1)
    const predEndDate = dates[predEndIdx] ?? ''
    const contextStartIdx = Math.max(0, startIdx - L)
    const contextStartDate = dates[contextStartIdx] ?? ''
    const contextEndDate = dates[Math.max(0, startIdx - 1)] ?? predStartDate

    return { contextStart: contextStartDate, contextEnd: contextEndDate, predStart: predStartDate, predEnd: predEndDate, predStartIdx: startIdx, predEndIdx, L, H }
  }, [testInfo, startIdx])

  const currentWindowData = useMemo(() => {
    if (!testInfo || !windowInfo) return null
    const dates = testInfo.test_dates.slice(windowInfo.predStartIdx, windowInfo.predEndIdx + 1)
    const values = testInfo.test_values.slice(windowInfo.predStartIdx, windowInfo.predEndIdx + 1)
    return { dates, values: values.filter((v): v is number => v !== null) }
  }, [testInfo, windowInfo])

  const windowPredValues = useMemo(() => {
    if (!currentWindowData || !forecastMutation.data) return null
    const forecastDates = forecastMutation.data.dates
    const forecastPreds = forecastMutation.data.predictions
    const dateSet = new Set(currentWindowData.dates)
    const aligned: number[] = []
    for (let i = 0; i < forecastDates.length; i++) {
      if (dateSet.has(forecastDates[i]) && forecastPreds[i] !== null) {
        aligned.push(forecastPreds[i] as number)
      }
    }
    return aligned.length > 0 ? aligned : null
  }, [currentWindowData, forecastMutation.data])

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

  // IPS classification per year-month (replaces old per-month-number grouping)
  const gtIps = useMemo<MonthIps[]>(() => {
    if (!currentWindowData || Object.keys(refStatsForGrid).length === 0) return []
    return computeMonthlyIps(currentWindowData.dates, currentWindowData.values, refStatsForGrid)
  }, [currentWindowData, refStatsForGrid])

  const predIps = useMemo<MonthIps[]>(() => {
    if (!currentWindowData || !windowPredValues || Object.keys(refStatsForGrid).length === 0) return []
    return computeMonthlyIps(currentWindowData.dates, windowPredValues, refStatsForGrid)
  }, [currentWindowData, windowPredValues, refStatsForGrid])

  // Quality gate: concordance between GT and model predictions
  const qualityGate = useMemo(() => {
    if (gtIps.length === 0 || predIps.length === 0) return null
    return computeQualityGate(gtIps, predIps)
  }, [gtIps, predIps])

  const verdict: QualityVerdict = qualityGate?.verdict ?? 'not_qualified'

  // Submit: fire both PhysCF and COMET in parallel
  const handleSubmit = useCallback(
    async (config: CFTargetData) => {
      // Increment generation to prevent stale SSE overwrites
      generationRef.current += 1
      const thisGeneration = generationRef.current

      // Close any existing streams
      eventSourcesRef.current.forEach((es) => es.close())
      eventSourcesRef.current = []
      setResults({})
      setStreaming({})
      setActiveTab('physcf')

      const baseBody = {
        model_id: config.model_id,
        target_ips_classes: config.target_ips_classes,
        start_idx: config.start_idx,
        lambda_prox: config.lambda_prox,
        n_iter: config.n_iter,
        lr: config.lr,
        cc_rate: config.cc_rate,
        k_sigma: config.k_sigma,
        lambda_smooth: config.lambda_smooth,
      }

      for (const method of METHODS) {
        try {
          const resp = await api.counterfactual.run({ ...baseBody, method })
          if (generationRef.current !== thisGeneration) return
          if (resp.task_id) {
            if (resp.status === 'done' && resp.result) {
              setResults((prev) => ({ ...prev, [method]: resp }))
            } else {
              const es = startCFStream(resp.task_id, method, setResults, setStreaming, thisGeneration, generationRef)
              eventSourcesRef.current.push(es)
            }
          }
        } catch (err) {
          if (generationRef.current !== thisGeneration) return
          setResults((prev) => ({
            ...prev,
            [method]: { task_id: '', status: 'error', result: null, error: (err as Error).message },
          }))
        }
      }
    },
    [],
  )

  const handleExport = () => {
    const allResults = Object.fromEntries(
      METHODS.filter((m) => results[m]?.result).map((m) => [m, results[m]])
    )
    if (Object.keys(allResults).length === 0) return
    const blob = new Blob([JSON.stringify(allResults, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `counterfactual_${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Pastas validation for a specific method
  const handlePastasValidate = useCallback(async (method: string) => {
    const r = results[method]
    if (!r?.task_id || !modelId) return
    setPastasLoading((prev) => ({ ...prev, [method]: true }))
    try {
      const resp = await api.counterfactual.pastasValidate({
        model_id: modelId,
        cf_task_id: r.task_id,
      })
      setPastasResults((prev) => ({ ...prev, [method]: resp }))
    } catch (err) {
      setPastasResults((prev) => ({
        ...prev,
        [method]: { model_id: modelId, cf_task_id: r.task_id, gamma: 1.5, accepted: false, rmse_cf: 0, rmse_0: 0, epsilon: 0, status: 'error', message: (err as Error).message },
      }))
    }
    setPastasLoading((prev) => ({ ...prev, [method]: false }))
  }, [results, modelId])

  // Active tab result
  const activeResult = results[activeTab] ?? null
  const activeInner = activeResult?.result ?? null
  const activePastas = pastasResults[activeTab] ?? null

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
        {hasAnyResult && (
          <button
            onClick={handleExport}
            className="flex items-center gap-2 bg-bg-hover text-text-primary px-4 py-2 rounded-lg border border-white/10 hover:bg-bg-hover/80 transition-colors text-sm"
          >
            <Download className="w-4 h-4" />
            Exporter JSON
          </button>
        )}
      </div>

      {/* Model selector */}
      <div className="bg-bg-card rounded-xl border border-white/5 p-5">
        <ModelSelector value={modelId} onChange={setModelId} />
      </div>

      {/* Test set overview with window slider */}
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

          <TestSetOverview
            testDates={testInfo.test_dates}
            testValues={testInfo.test_values}
            sliderIdx={startIdx}
            inputChunkLength={testInfo.input_chunk_length}
            outputChunkLength={testInfo.output_chunk_length}
            windowResult={forecastMutation.data}
            className="h-[300px]"
          />

          <div className="flex items-center gap-4">
            <span className="text-xs text-text-secondary shrink-0">Position</span>
            <input
              type="range"
              min={testInfo.valid_start_idx}
              max={testInfo.valid_end_idx}
              value={sliderDraft}
              onChange={(e) => setSliderDraft(Number(e.target.value))}
              onMouseUp={() => setStartIdx(sliderDraft)}
              onTouchEnd={() => setStartIdx(sliderDraft)}
              onKeyUp={() => setStartIdx(sliderDraft)}
              className="flex-1 accent-accent-cyan h-1.5"
            />
            <span className="text-xs text-text-primary font-mono w-12 text-right">{sliderDraft}</span>
          </div>

          {currentWindowData && currentWindowData.dates.length > 0 && Object.keys(refStatsForGrid).length > 0 && (
            <IPSMonthlyGrid
              predDates={currentWindowData.dates}
              predValues={windowPredValues ?? currentWindowData.values}
              gtValues={currentWindowData.values}
              refStats={refStatsForGrid}
              ipsLabels={ipsBoundsData?.classes ?? {}}
              ipsColors={ipsBoundsData?.colors ?? {}}
              label="Classification IPS de la fenetre selectionnee"
              concordance={qualityGate?.months}
            />
          )}

          {/* Quality gate verdict banner */}
          {qualityGate && (
            <div className={`rounded-lg p-3 flex items-center gap-2 text-sm ${
              verdict === 'qualified'
                ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400'
                : verdict === 'partial'
                  ? 'bg-amber-500/10 border border-amber-500/20 text-amber-400'
                  : 'bg-red-500/10 border border-red-500/20 text-red-400'
            }`}>
              {verdict === 'qualified' && 'Le modele reproduit fidelement l\'observe sur cette fenetre'}
              {verdict === 'partial' && 'Concordance partielle — certains mois divergent'}
              {verdict === 'not_qualified' && 'Le modele ne reproduit pas l\'observe — contrefactuelle non fiable'}
            </div>
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
              <CFTargetSelector
                modelId={modelId}
                startIdx={startIdx}
                gtIps={gtIps}
                verdict={verdict}
                isForecastLoading={forecastMutation.isPending}
                onSubmit={handleSubmit}
                isPending={isLoading}
              />
            </div>
          </div>

          {/* Right: Results with method tabs */}
          <div className="lg:col-span-9 space-y-4">
            {/* Method tabs */}
            {(isLoading || hasAnyResult) && (
              <div className="flex border-b border-white/10">
                {METHODS.map((m) => {
                  const r = results[m]
                  const done = r?.result != null
                  const running = streaming[m]
                  const error = r?.error != null
                  return (
                    <button
                      key={m}
                      onClick={() => setActiveTab(m)}
                      className={`px-4 py-2 text-sm transition-colors relative ${
                        activeTab === m
                          ? 'border-b-2 border-accent-cyan text-accent-cyan'
                          : 'text-text-secondary hover:text-text-primary'
                      }`}
                    >
                      {METHOD_LABELS[m]}
                      {running && (
                        <span className="ml-2 inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                      )}
                      {done && !running && (
                        <span className="ml-2 inline-block w-2 h-2 rounded-full bg-emerald-400" />
                      )}
                      {error && !running && (
                        <span className="ml-2 inline-block w-2 h-2 rounded-full bg-red-400" />
                      )}
                    </button>
                  )
                })}
              </div>
            )}

            <CFResultView
              result={activeResult}
              isLoading={!!streaming[activeTab]}
            />

            {/* IPS Monthly Grid for active result */}
            {activeInner && Object.keys(refStatsForGrid).length > 0 && (
              <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                <h4 className="text-sm font-semibold text-text-primary mb-3">
                  Classification IPS mensuelle — {METHOD_LABELS[activeTab]}
                </h4>
                <IPSMonthlyGrid
                  predDates={activeInner.dates}
                  predValues={activeInner.counterfactual}
                  gtValues={activeInner.original}
                  refStats={refStatsForGrid}
                  ipsLabels={ipsBoundsData?.classes ?? {}}
                  ipsColors={ipsBoundsData?.colors ?? {}}
                  label="Contrefactuel vs Original"
                />
              </div>
            )}

            {/* Pastas dual validation */}
            {activeInner && (
              <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4" />
                    Validation Pastas (TFN)
                  </h4>
                  {!activePastas && (
                    <button
                      onClick={() => handlePastasValidate(activeTab)}
                      disabled={!!pastasLoading[activeTab]}
                      className="text-xs bg-accent-indigo/20 text-accent-indigo px-3 py-1.5 rounded-lg hover:bg-accent-indigo/30 disabled:opacity-50 transition-colors"
                    >
                      {pastasLoading[activeTab] ? 'Validation...' : 'Lancer la validation'}
                    </button>
                  )}
                </div>

                {pastasLoading[activeTab] && (
                  <div className="animate-pulse flex gap-3">
                    <div className="h-3 bg-bg-hover rounded w-1/3" />
                    <div className="h-3 bg-bg-hover rounded w-1/4" />
                  </div>
                )}

                {activePastas && activePastas.status === 'error' && (
                  <p className="text-sm text-accent-red">{activePastas.message ?? 'Erreur de validation'}</p>
                )}

                {activePastas && activePastas.status !== 'error' && (
                  <div className="space-y-2">
                    <div className={`rounded-lg p-3 flex items-center gap-3 ${
                      activePastas.accepted
                        ? 'bg-emerald-500/10 border border-emerald-500/20'
                        : 'bg-amber-500/10 border border-amber-500/20'
                    }`}>
                      <span className={`text-sm font-semibold ${activePastas.accepted ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {activePastas.accepted ? 'Valide' : 'Rejete'}
                      </span>
                      <span className="text-xs text-text-secondary">
                        RMSE CF = {activePastas.rmse_cf.toFixed(4)} | Seuil = {activePastas.epsilon.toFixed(4)} (γ={activePastas.gamma})
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div className="bg-bg-hover/30 rounded p-2">
                        <p className="text-[10px] text-text-secondary uppercase">RMSE baseline</p>
                        <p className="text-sm font-mono text-text-primary">{activePastas.rmse_0.toFixed(4)}</p>
                      </div>
                      <div className="bg-bg-hover/30 rounded p-2">
                        <p className="text-[10px] text-text-secondary uppercase">RMSE CF</p>
                        <p className="text-sm font-mono text-text-primary">{activePastas.rmse_cf.toFixed(4)}</p>
                      </div>
                      <div className="bg-bg-hover/30 rounded p-2">
                        <p className="text-[10px] text-text-secondary uppercase">Tolerance ε</p>
                        <p className="text-sm font-mono text-text-primary">{activePastas.epsilon.toFixed(4)}</p>
                      </div>
                    </div>
                  </div>
                )}

                {!activePastas && !pastasLoading[activeTab] && (
                  <p className="text-[10px] text-text-secondary/50">
                    Validation independante par modele hydrologique Pastas (Transfer Function Noise).
                    Compare les predictions CF du TFT avec celles d'un modele physique independant.
                  </p>
                )}
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
