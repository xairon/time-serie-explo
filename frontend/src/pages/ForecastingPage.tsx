import { useState, useMemo, useCallback } from 'react'
import { ChevronDown, ChevronRight, Download } from 'lucide-react'
import { ModelSelector } from '@/components/forecasting/ModelSelector'
import { ForecastView } from '@/components/forecasting/ForecastView'
import { MetricsPanel } from '@/components/forecasting/MetricsPanel'
import { ExplainabilityPanel } from '@/components/forecasting/ExplainabilityPanel'
import {
  useForecastSingle,
  useForecastRolling,
  useForecastComparison,
  useForecastGlobal,
} from '@/hooks/useForecasting'
import { useDatasets } from '@/hooks/useDatasets'
import { useModelDetail } from '@/hooks/useModels'

type ForecastMode = 'single' | 'rolling' | 'comparison' | 'global'

export default function ForecastingPage() {
  const [modelId, setModelId] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [horizon, setHorizon] = useState(30)
  const [startDate, setStartDate] = useState('')
  const [stride, setStride] = useState(1)
  const [mode, setMode] = useState<ForecastMode>('single')
  const [hyperparamsOpen, setHyperparamsOpen] = useState(false)

  const { data: datasets } = useDatasets()
  const { data: modelDetail } = useModelDetail(modelId || null)

  const singleMutation = useForecastSingle()
  const rollingMutation = useForecastRolling()
  const comparisonMutation = useForecastComparison()
  const globalMutation = useForecastGlobal()

  const modes: { key: ForecastMode; label: string; description: string }[] = [
    { key: 'single', label: 'Unique', description: 'Prevision autoregressive simple' },
    { key: 'rolling', label: 'Glissant', description: 'Fenetres glissantes avec stride configurable' },
    { key: 'comparison', label: 'Comparaison', description: 'Autoregressif vs one-step' },
    { key: 'global', label: 'Global', description: 'Prevision sur tout le dataset' },
  ]

  // The active mutation depends on the mode
  const activeMutation = useMemo(() => {
    switch (mode) {
      case 'single':
        return singleMutation
      case 'rolling':
        return rollingMutation
      case 'comparison':
        return comparisonMutation
      case 'global':
        return globalMutation
    }
  }, [mode, singleMutation, rollingMutation, comparisonMutation, globalMutation])

  const needsStartDate = mode === 'rolling' || mode === 'comparison'
  const needsHorizon = mode === 'single' || mode === 'rolling' || mode === 'comparison'
  const needsDataset = mode === 'single'
  const needsStride = mode === 'rolling'

  // Extract chunk lengths from model hyperparams
  const inputChunkLength = useMemo(() => {
    const hp = modelDetail?.hyperparams
    if (!hp) return undefined
    const val = hp['input_chunk_length'] ?? hp['input_length']
    return typeof val === 'number' ? val : undefined
  }, [modelDetail])

  const outputChunkLength = useMemo(() => {
    const hp = modelDetail?.hyperparams
    if (!hp) return undefined
    const val = hp['output_chunk_length'] ?? hp['output_length']
    return typeof val === 'number' ? val : undefined
  }, [modelDetail])

  // Test set date range for slider (from model detail or result)
  const testDateRange = useMemo(() => {
    const hp = modelDetail?.hyperparams
    if (!hp) return null
    const testStart = hp['test_start_date'] as string | undefined
    const testEnd = hp['test_end_date'] as string | undefined
    if (testStart && testEnd) return { min: testStart, max: testEnd }
    return null
  }, [modelDetail])

  // Dataset split sizes from hyperparams
  const datasetSplits = useMemo(() => {
    const hp = modelDetail?.hyperparams
    if (!hp) return null
    const trainSize = hp['train_size'] ?? hp['n_train']
    const valSize = hp['val_size'] ?? hp['n_val']
    const testSize = hp['test_size'] ?? hp['n_test']
    if (trainSize != null || valSize != null || testSize != null) {
      return {
        train: trainSize as number | undefined,
        val: valSize as number | undefined,
        test: testSize as number | undefined,
      }
    }
    return null
  }, [modelDetail])

  const canRun = (): boolean => {
    if (!modelId) return false
    if (needsDataset && !datasetId) return false
    if (needsStartDate && !startDate) return false
    return true
  }

  const handleRun = () => {
    if (!canRun()) return

    switch (mode) {
      case 'single':
        singleMutation.mutate({
          model_id: modelId,
          ...(datasetId ? { dataset_id: datasetId } : {}),
          ...(horizon ? { horizon } : {}),
          ...(startDate ? { start_date: startDate } : {}),
        })
        break
      case 'rolling':
        rollingMutation.mutate({
          model_id: modelId,
          start_date: startDate,
          forecast_horizon: horizon,
          stride,
        })
        break
      case 'comparison':
        comparisonMutation.mutate({
          model_id: modelId,
          start_date: startDate,
          forecast_horizon: horizon,
        })
        break
      case 'global':
        globalMutation.mutate({
          model_id: modelId,
        })
        break
    }
  }

  const result = activeMutation.data ?? null
  const isPending = activeMutation.isPending
  const isError = activeMutation.isError
  const error = activeMutation.error

  // Prediction window text
  const predictionWindowText = useMemo(() => {
    if (!result || result.dates.length === 0) return null
    const start = result.dates[0]
    const end = result.dates[result.dates.length - 1]
    const days = result.dates.length
    return `Prediction : ${new Date(start).toLocaleDateString('fr-FR')} → ${new Date(end).toLocaleDateString('fr-FR')} (${days}j)`
  }, [result])

  // CSV export
  const handleDownloadCSV = useCallback(() => {
    if (!result) return
    const lines = ['date,actual,predicted']
    for (let i = 0; i < result.dates.length; i++) {
      const date = result.dates[i]
      const actual = result.actuals[i] != null ? String(result.actuals[i]) : ''
      const predicted = result.predictions[i] != null ? String(result.predictions[i]) : ''
      lines.push(`${date},${actual},${predicted}`)
    }
    const csv = lines.join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    const modelName = modelDetail?.model_name ?? 'model'
    const dateStr = new Date().toISOString().slice(0, 10)
    a.href = url
    a.download = `forecast_${modelName}_${dateStr}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [result, modelDetail])

  // Filtered hyperparams (exclude internal/split keys for the collapsible section)
  const displayHyperparams = useMemo(() => {
    if (!modelDetail?.hyperparams) return null
    const skipKeys = new Set([
      'train_size', 'val_size', 'test_size', 'n_train', 'n_val', 'n_test',
      'test_start_date', 'test_end_date',
    ])
    const entries = Object.entries(modelDetail.hyperparams).filter(
      ([key]) => !skipKeys.has(key),
    )
    return entries.length > 0 ? entries : null
  }, [modelDetail])

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-1">Prevision</h1>
        <p className="text-sm text-text-secondary">
          Prediction piezometrique avec les modeles entraines
        </p>
      </div>

      {/* Mode tabs */}
      <div className="bg-bg-card rounded-xl border border-white/5 p-5">
        <div className="flex border-b border-white/10 mb-4">
          {modes.map((m) => (
            <button
              key={m.key}
              onClick={() => setMode(m.key)}
              className={`px-4 py-2 text-xs transition-colors ${
                mode === m.key
                  ? 'border-b-2 border-accent-cyan text-accent-cyan'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
              title={m.description}
            >
              {m.label}
            </button>
          ))}
        </div>

        <p className="text-xs text-text-secondary mb-4">
          {modes.find((m) => m.key === mode)?.description}
        </p>

        {/* Controls - adapt to mode */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div className="md:col-span-1">
            <ModelSelector value={modelId} onChange={setModelId} />
          </div>

          {needsDataset && (
            <div>
              <label className="block text-xs text-text-secondary mb-1">Dataset</label>
              <select
                value={datasetId}
                onChange={(e) => setDatasetId(e.target.value)}
                className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              >
                <option value="">Selectionner</option>
                {datasets?.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {needsStartDate && (
            <div>
              <label className="block text-xs text-text-secondary mb-1">Date de debut</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              />
            </div>
          )}

          {/* Sliding window start date for single mode */}
          {mode === 'single' && modelId && (
            <div>
              <label className="block text-xs text-text-secondary mb-1">
                Date de debut (optionnel)
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                min={testDateRange?.min ?? undefined}
                max={testDateRange?.max ?? undefined}
                className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              />
              {testDateRange && (
                <p className="text-[10px] text-text-secondary mt-1">
                  Test : {new Date(testDateRange.min).toLocaleDateString('fr-FR')} — {new Date(testDateRange.max).toLocaleDateString('fr-FR')}
                </p>
              )}
            </div>
          )}

          {needsHorizon && (
            <div>
              <label className="block text-xs text-text-secondary mb-1">Horizon (jours)</label>
              <input
                type="number"
                min={1}
                max={365}
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              />
            </div>
          )}

          {needsStride && (
            <div>
              <label className="block text-xs text-text-secondary mb-1">Stride (pas)</label>
              <input
                type="number"
                min={1}
                max={horizon}
                value={stride}
                onChange={(e) => setStride(Number(e.target.value))}
                className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              />
            </div>
          )}

          <div>
            <button
              onClick={handleRun}
              disabled={isPending || !canRun()}
              className="w-full bg-accent-cyan text-white px-4 py-2 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium"
            >
              {isPending ? 'Calcul...' : 'Lancer la prevision'}
            </button>
          </div>
        </div>
      </div>

      {/* Model Info Panel */}
      {modelDetail && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-5 space-y-4">
          <h3 className="text-sm font-semibold text-text-primary">
            Informations du modele
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Dataset splits */}
            {datasetSplits && (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                  Decoupage du dataset
                </h4>
                <div className="space-y-1">
                  {datasetSplits.train != null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Train</span>
                      <span className="text-text-primary">{datasetSplits.train} pts</span>
                    </div>
                  )}
                  {datasetSplits.val != null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Validation</span>
                      <span className="text-text-primary">{datasetSplits.val} pts</span>
                    </div>
                  )}
                  {datasetSplits.test != null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Test</span>
                      <span className="text-text-primary">{datasetSplits.test} pts</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Preprocessing config summary */}
            {modelDetail.preprocessing_config && Object.keys(modelDetail.preprocessing_config).length > 0 && (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                  Pretraitement
                </h4>
                <div className="space-y-1">
                  {Object.entries(modelDetail.preprocessing_config).slice(0, 5).map(([key, val]) => (
                    <div key={key} className="flex justify-between text-xs">
                      <span className="text-text-secondary">{key}</span>
                      <span className="text-text-primary">
                        {typeof val === 'boolean' ? (val ? 'Oui' : 'Non') : String(val ?? '')}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Key hyperparams summary */}
            {inputChunkLength != null && (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                  Architecture
                </h4>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-text-secondary">Input chunk</span>
                    <span className="text-text-primary">{inputChunkLength}</span>
                  </div>
                  {outputChunkLength != null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Output chunk</span>
                      <span className="text-text-primary">{outputChunkLength}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Collapsible hyperparams */}
          {displayHyperparams && (
            <div>
              <button
                onClick={() => setHyperparamsOpen(!hyperparamsOpen)}
                className="flex items-center gap-1.5 text-xs text-text-secondary hover:text-text-primary transition-colors"
              >
                {hyperparamsOpen ? (
                  <ChevronDown className="w-3.5 h-3.5" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5" />
                )}
                Hyperparametres ({displayHyperparams.length})
              </button>
              {hyperparamsOpen && (
                <div className="mt-2 grid grid-cols-2 md:grid-cols-3 gap-x-6 gap-y-1 bg-white/[0.02] rounded-lg p-3">
                  {displayHyperparams.map(([key, val]) => (
                    <div key={key} className="flex justify-between text-xs py-0.5">
                      <span className="text-text-secondary truncate mr-2">{key}</span>
                      <span className="text-text-primary font-mono text-[11px] shrink-0">
                        {typeof val === 'boolean'
                          ? val ? 'true' : 'false'
                          : typeof val === 'object'
                            ? JSON.stringify(val)
                            : String(val ?? '')}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {isError && (
        <div className="bg-accent-red/10 border border-accent-red/20 rounded-xl p-4 flex items-center justify-between">
          <p className="text-sm text-accent-red">
            Erreur : {(error as Error).message}
          </p>
          <button
            onClick={handleRun}
            className="text-xs text-accent-cyan hover:underline"
          >
            Reessayer
          </button>
        </div>
      )}

      {/* Prediction window info + CSV export */}
      {result && (
        <div className="flex items-center justify-between">
          {predictionWindowText && (
            <p className="text-xs text-text-secondary">{predictionWindowText}</p>
          )}
          <button
            onClick={handleDownloadCSV}
            className="flex items-center gap-1.5 text-xs text-accent-cyan hover:text-accent-cyan/80 transition-colors px-3 py-1.5 rounded-lg border border-accent-cyan/20 hover:border-accent-cyan/40"
          >
            <Download className="w-3.5 h-3.5" />
            Telecharger CSV
          </button>
        </div>
      )}

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ForecastView
            result={result}
            isLoading={isPending}
            inputChunkLength={inputChunkLength}
            outputChunkLength={outputChunkLength}
            className="min-h-[400px]"
          />
        </div>
        <div className="space-y-4">
          {result ? (
            <>
              <MetricsPanel metrics={result.metrics} actuals={result.actuals} />
              {result.metrics_onestep && (
                <div className="mt-2">
                  <h4 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">
                    Metriques One-Step
                  </h4>
                  <MetricsPanel metrics={result.metrics_onestep} actuals={result.actuals} />
                </div>
              )}

              {/* Test set info */}
              {result.dates.length > 0 && (
                <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                  <h4 className="text-xs font-semibold text-text-secondary mb-3 uppercase tracking-wide">
                    Fenetre de prevision
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Debut</span>
                      <span className="text-text-primary">{new Date(result.dates[0]).toLocaleDateString('fr-FR')}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Fin</span>
                      <span className="text-text-primary">{new Date(result.dates[result.dates.length - 1]).toLocaleDateString('fr-FR')}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Points</span>
                      <span className="text-text-primary">{result.dates.length}</span>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="bg-bg-card rounded-xl border border-white/5 p-6 flex items-center justify-center min-h-[200px]">
              <p className="text-xs text-text-secondary text-center">
                Les metriques apparaitront ici apres la prevision.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Explainability */}
      {modelId && result && (
        <ExplainabilityPanel modelId={modelId} />
      )}
    </div>
  )
}
