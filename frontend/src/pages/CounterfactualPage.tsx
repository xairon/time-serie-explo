import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { Download } from 'lucide-react'
import { ModelSelector } from '@/components/forecasting/ModelSelector'
import { CFTargetSelector } from '@/components/counterfactual/CFTargetSelector'
import type { CFTargetData } from '@/components/counterfactual/CFTargetSelector'
import { CFComparisonView } from '@/components/counterfactual/CFComparisonView'
import { TestSetOverview } from '@/components/charts/TestSetOverview'
import { useIPSReference } from '@/hooks/useCounterfactual'
import { useForecastSingle } from '@/hooks/useForecasting'
import { useModelTestInfo } from '@/hooks/useModels'
import { api } from '@/lib/api'
import { computeMonthlyIps, computeQualityGate, type MonthIps, type QualityVerdict } from '@/lib/ips'
import type { CounterfactualResult } from '@/lib/types'

const METHODS = ['physcf', 'comte'] as const

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
  const [modelId, setModelId] = useState<string>('')
  const [startIdx, setStartIdx] = useState(0)
  const [sliderDraft, setSliderDraft] = useState(0)

  const { data: testInfo } = useModelTestInfo(modelId || null)
  const { data: ipsRef } = useIPSReference(modelId || null, 1)

  const forecastMutation = useForecastSingle()

  const [results, setResults] = useState<Record<string, CounterfactualResult | null>>({})
  const [streaming, setStreaming] = useState<Record<string, boolean>>({})
  const eventSourcesRef = useRef<EventSource[]>([])
  const generationRef = useRef(0)

  const isLoading = METHODS.some((m) => streaming[m])
  const hasAnyResult = METHODS.some((m) => results[m]?.result || results[m]?.error)

  useEffect(() => {
    setStartIdx(0)
    setSliderDraft(0)
    setResults({})
    setStreaming({})
  }, [modelId])

  useEffect(() => {
    if (testInfo && startIdx === 0) {
      const mid = Math.floor((testInfo.valid_start_idx + testInfo.valid_end_idx) / 2)
      setStartIdx(mid)
      setSliderDraft(mid)
    }
  }, [testInfo, startIdx])

  useEffect(() => {
    if (modelId && testInfo && testInfo.test_dates[startIdx]) {
      forecastMutation.mutate({ model_id: modelId, start_date: testInfo.test_dates[startIdx] })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [startIdx, modelId, testInfo])

  useEffect(() => {
    return () => { eventSourcesRef.current.forEach((es) => es.close()) }
  }, [])

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
    const dateSet = new Set(currentWindowData.dates.map(d => d.slice(0, 10)))
    const aligned: number[] = []
    for (let i = 0; i < forecastDates.length; i++) {
      if (dateSet.has(forecastDates[i]?.slice(0, 10)) && forecastPreds[i] !== null) {
        aligned.push(forecastPreds[i] as number)
      }
    }
    return aligned.length > 0 ? aligned : null
  }, [currentWindowData, forecastMutation.data])

  const refStatsForGrid = useMemo(() => {
    if (!ipsRef?.ref_stats) return {}
    const out: Record<string, [number, number]> = {}
    for (const [k, v] of Object.entries(ipsRef.ref_stats)) {
      if (Array.isArray(v) && v.length >= 2) out[k] = [v[0] as number, v[1] as number]
    }
    return out
  }, [ipsRef])

  const gtIps = useMemo<MonthIps[]>(() => {
    if (!currentWindowData || Object.keys(refStatsForGrid).length === 0) return []
    return computeMonthlyIps(currentWindowData.dates, currentWindowData.values, refStatsForGrid)
  }, [currentWindowData, refStatsForGrid])

  const predIps = useMemo<MonthIps[]>(() => {
    if (!currentWindowData || !windowPredValues || Object.keys(refStatsForGrid).length === 0) return []
    return computeMonthlyIps(currentWindowData.dates, windowPredValues, refStatsForGrid)
  }, [currentWindowData, windowPredValues, refStatsForGrid])

  const qualityGate = useMemo(() => {
    if (gtIps.length === 0 || predIps.length === 0) return null
    return computeQualityGate(gtIps, predIps)
  }, [gtIps, predIps])

  const verdict: QualityVerdict = qualityGate?.verdict ?? 'not_qualified'

  const handleSubmit = useCallback(
    async (config: CFTargetData) => {
      generationRef.current += 1
      const thisGeneration = generationRef.current
      eventSourcesRef.current.forEach((es) => es.close())
      eventSourcesRef.current = []
      setResults({})
      setStreaming({})

      const baseBody = {
        model_id: config.model_id,
        target_ips_classes: config.target_ips_classes,
        start_idx: config.start_idx,
        lambda_prox: config.lambda_prox,
        n_iter: config.n_iter,
        lr: config.lr,
        cc_rate: config.cc_rate,
        num_distractors: config.num_distractors,
        tau: config.tau,
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

      {/* Main content: Config sidebar + Comparison results */}
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

          {/* Right: Unified comparison view */}
          <div className="lg:col-span-9">
            <CFComparisonView
              results={results}
              streaming={streaming}
              modelId={modelId}
              gtDates={currentWindowData?.dates}
              gtValues={currentWindowData?.values}
            />
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
