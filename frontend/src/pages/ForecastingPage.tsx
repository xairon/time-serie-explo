import { useState } from 'react'
import { ModelSelector } from '@/components/forecasting/ModelSelector'
import { ForecastView } from '@/components/forecasting/ForecastView'
import { MetricsPanel } from '@/components/forecasting/MetricsPanel'
import { ExplainabilityPanel } from '@/components/forecasting/ExplainabilityPanel'
import { useForecast } from '@/hooks/useForecasting'
import { useDatasets } from '@/hooks/useDatasets'

type ForecastMode = 'single' | 'rolling' | 'comparison' | 'global'

export default function ForecastingPage() {
  const [modelId, setModelId] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [horizon, setHorizon] = useState(30)
  const [mode, setMode] = useState<ForecastMode>('single')

  const { data: datasets } = useDatasets()
  const forecastMutation = useForecast()

  const modes: { key: ForecastMode; label: string }[] = [
    { key: 'single', label: 'Unique' },
    { key: 'rolling', label: 'Glissant' },
    { key: 'comparison', label: 'Comparaison' },
    { key: 'global', label: 'Global' },
  ]

  const handleRun = () => {
    if (!modelId || !datasetId) return
    forecastMutation.mutate({ model_id: modelId, horizon, dataset_id: datasetId })
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-1">Prévision</h1>
        <p className="text-sm text-text-secondary">
          Prédiction piézométrique avec les modèles entraînés
        </p>
      </div>

      {/* Top controls */}
      <div className="bg-bg-card rounded-xl border border-white/5 p-5">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div className="md:col-span-1">
            <ModelSelector value={modelId} onChange={setModelId} />
          </div>
          <div>
            <label className="block text-xs text-text-secondary mb-1">Dataset</label>
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
            >
              <option value="">Sélectionner</option>
              {datasets?.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </div>
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
          <div>
            <button
              onClick={handleRun}
              disabled={forecastMutation.isPending || !modelId || !datasetId}
              className="w-full bg-accent-cyan text-white px-4 py-2 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium"
            >
              {forecastMutation.isPending ? 'Calcul...' : 'Lancer la prévision'}
            </button>
          </div>
        </div>

        {/* Mode tabs */}
        <div className="flex border-b border-white/10 mt-4">
          {modes.map((m) => (
            <button
              key={m.key}
              onClick={() => setMode(m.key)}
              className={`px-4 py-2 text-xs transition-colors ${
                mode === m.key
                  ? 'border-b-2 border-accent-cyan text-accent-cyan'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {forecastMutation.isError && (
        <div className="bg-accent-red/10 border border-accent-red/20 rounded-xl p-4 flex items-center justify-between">
          <p className="text-sm text-accent-red">
            Erreur : {(forecastMutation.error as Error).message}
          </p>
          <button
            onClick={handleRun}
            className="text-xs text-accent-cyan hover:underline"
          >
            Réessayer
          </button>
        </div>
      )}

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ForecastView
            result={forecastMutation.data ?? null}
            isLoading={forecastMutation.isPending}
            className="min-h-[400px]"
          />
        </div>
        <div>
          {forecastMutation.data ? (
            <MetricsPanel metrics={forecastMutation.data.metrics} />
          ) : (
            <div className="bg-bg-card rounded-xl border border-white/5 p-6 flex items-center justify-center min-h-[200px]">
              <p className="text-xs text-text-secondary text-center">
                Les métriques apparaîtront ici après la prévision.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Explainability */}
      {modelId && forecastMutation.data && (
        <ExplainabilityPanel modelId={modelId} />
      )}
    </div>
  )
}
