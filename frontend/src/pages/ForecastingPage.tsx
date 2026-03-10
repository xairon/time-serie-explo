import { useState, useMemo } from 'react'
import { ModelSelector } from '@/components/forecasting/ModelSelector'
import { ForecastView } from '@/components/forecasting/ForecastView'
import { MetricsPanel } from '@/components/forecasting/MetricsPanel'
import { ExplainabilityPanel } from '@/components/forecasting/ExplainabilityPanel'
import {
  useForecast,
  useForecastRolling,
  useForecastComparison,
  useForecastGlobal,
} from '@/hooks/useForecasting'
import { useDatasets } from '@/hooks/useDatasets'

type ForecastMode = 'single' | 'rolling' | 'comparison' | 'global'

export default function ForecastingPage() {
  const [modelId, setModelId] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [horizon, setHorizon] = useState(30)
  const [startDate, setStartDate] = useState('')
  const [stride, setStride] = useState(1)
  const [mode, setMode] = useState<ForecastMode>('single')

  const { data: datasets } = useDatasets()

  const singleMutation = useForecast()
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

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ForecastView
            result={result}
            isLoading={isPending}
            className="min-h-[400px]"
          />
        </div>
        <div className="space-y-4">
          {result ? (
            <>
              <MetricsPanel metrics={result.metrics} />
              {result.metrics_onestep && (
                <div className="mt-2">
                  <h4 className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">
                    Metriques One-Step
                  </h4>
                  <MetricsPanel metrics={result.metrics_onestep} />
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
