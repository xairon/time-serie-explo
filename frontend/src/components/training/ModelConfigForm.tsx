import { useState, useEffect, useMemo } from 'react'
import { useAvailableModels } from '@/hooks/useModels'
import { useDatasets } from '@/hooks/useDatasets'
import type { TrainingConfig, AvailableModel } from '@/lib/types'

const LOSS_FUNCTIONS = [
  { value: 'MAE', label: 'MAE (Mean Absolute Error)' },
  { value: 'MSE', label: 'MSE (Mean Squared Error)' },
  { value: 'Huber', label: 'Huber' },
  { value: 'Quantile', label: 'Quantile' },
  { value: 'RMSE', label: 'RMSE (Root Mean Squared Error)' },
]

interface ModelConfigFormProps {
  onSubmit: (config: TrainingConfig) => void
  isPending: boolean
}

export function ModelConfigForm({ onSubmit, isPending }: ModelConfigFormProps) {
  const { data: availableModels, isLoading: modelsLoading } = useAvailableModels()
  const { data: datasets, isLoading: datasetsLoading } = useDatasets()

  const [modelType, setModelType] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [station, setStation] = useState('')
  const [maxEpochs, setMaxEpochs] = useState(100)
  const [earlyStopping, setEarlyStopping] = useState(true)
  const [patience, setPatience] = useState(10)
  const [trainSplit, setTrainSplit] = useState(0.7)
  const [valSplit, setValSplit] = useState(0.15)
  const [lossFunction, setLossFunction] = useState('MAE')
  const [hyperparams, setHyperparams] = useState<Record<string, unknown>>({})

  // Group models by category
  const modelsByCategory = useMemo(() => {
    if (!availableModels) return new Map<string, AvailableModel[]>()
    const groups = new Map<string, AvailableModel[]>()
    for (const m of availableModels) {
      const cat = m.category || 'Autre'
      if (!groups.has(cat)) groups.set(cat, [])
      groups.get(cat)!.push(m)
    }
    return groups
  }, [availableModels])

  // Set defaults when available models load
  useEffect(() => {
    if (availableModels?.length && !modelType) {
      setModelType(availableModels[0].name)
      setHyperparams(availableModels[0].default_hyperparams)
    }
  }, [availableModels, modelType])

  useEffect(() => {
    if (datasets?.length && !datasetId) {
      setDatasetId(datasets[0].id)
    }
  }, [datasets, datasetId])

  // Update hyperparams when model type changes
  useEffect(() => {
    const model = availableModels?.find((m) => m.name === modelType)
    if (model) {
      setHyperparams(model.default_hyperparams)
    }
  }, [modelType, availableModels])

  const selectedDataset = datasets?.find((d) => d.id === datasetId)
  const selectedModel = availableModels?.find((m) => m.name === modelType)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({
      dataset_id: datasetId,
      model_name: modelType,
      hyperparams,
      train_ratio: trainSplit,
      val_ratio: valSplit,
      n_epochs: maxEpochs,
      early_stopping: earlyStopping,
      early_stopping_patience: earlyStopping ? patience : 0,
      station_name: station || null,
      use_covariates: true,
      loss_function: lossFunction,
    })
  }

  const updateHyperparam = (key: string, value: string) => {
    const num = Number(value)
    setHyperparams((prev) => ({ ...prev, [key]: isNaN(num) ? value : num }))
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h3 className="text-sm font-semibold text-text-primary">Configuration du modele</h3>

      {/* Model type - grouped by category */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Architecture</label>
        {modelsLoading ? (
          <div className="h-9 bg-bg-hover rounded-lg animate-pulse" />
        ) : (
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            {[...modelsByCategory.entries()].map(([category, models]) => (
              <optgroup key={category} label={category}>
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        )}
        {selectedModel?.description && (
          <p className="text-[10px] text-text-secondary mt-1">
            {selectedModel.description}
          </p>
        )}
      </div>

      {/* Dataset */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Dataset</label>
        {datasetsLoading ? (
          <div className="h-9 bg-bg-hover rounded-lg animate-pulse" />
        ) : (
          <select
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            {datasets?.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.n_rows} lignes)
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Station */}
      {selectedDataset && selectedDataset.stations.length > 0 && (
        <div>
          <label className="block text-xs text-text-secondary mb-1">Station</label>
          <select
            value={station}
            onChange={(e) => setStation(e.target.value)}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">Toutes les stations</option>
            {selectedDataset.stations.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Loss function */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Fonction de perte</label>
        <select
          value={lossFunction}
          onChange={(e) => setLossFunction(e.target.value)}
          className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
        >
          {LOSS_FUNCTIONS.map((lf) => (
            <option key={lf.value} value={lf.value}>
              {lf.label}
            </option>
          ))}
        </select>
      </div>

      {/* Splits */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-secondary mb-1">Train split</label>
          <input
            type="number"
            min={0.5}
            max={0.9}
            step={0.05}
            value={trainSplit}
            onChange={(e) => setTrainSplit(Number(e.target.value))}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          />
        </div>
        <div>
          <label className="block text-xs text-text-secondary mb-1">Val split</label>
          <input
            type="number"
            min={0.05}
            max={0.3}
            step={0.05}
            value={valSplit}
            onChange={(e) => setValSplit(Number(e.target.value))}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          />
        </div>
      </div>

      {/* Epochs */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Epochs max</label>
        <input
          type="number"
          min={1}
          max={1000}
          value={maxEpochs}
          onChange={(e) => setMaxEpochs(Number(e.target.value))}
          className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
        />
      </div>

      {/* Early stopping toggle + patience */}
      <div className="space-y-2">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={earlyStopping}
            onChange={(e) => setEarlyStopping(e.target.checked)}
            className="w-4 h-4 rounded border-white/10 bg-bg-input text-accent-cyan focus:ring-accent-cyan/50"
          />
          <span className="text-xs text-text-secondary">Early stopping</span>
        </label>
        {earlyStopping && (
          <div>
            <label className="block text-xs text-text-secondary mb-1">Patience</label>
            <input
              type="number"
              min={1}
              max={100}
              value={patience}
              onChange={(e) => setPatience(Number(e.target.value))}
              className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
            />
          </div>
        )}
      </div>

      {/* Dynamic hyperparams */}
      {Object.keys(hyperparams).length > 0 && (
        <div>
          <label className="block text-xs text-text-secondary mb-2">Hyperparametres</label>
          <div className="space-y-2">
            {Object.entries(hyperparams).map(([key, val]) => (
              <div key={key} className="flex items-center gap-2">
                <label className="text-xs text-text-secondary w-32 shrink-0">{key}</label>
                <input
                  type="text"
                  value={String(val)}
                  onChange={(e) => updateHyperparam(key, e.target.value)}
                  className="flex-1 bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-1.5 text-xs"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      <button
        type="submit"
        disabled={isPending || !modelType || !datasetId}
        className="w-full bg-accent-cyan text-white px-4 py-2 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium"
      >
        {isPending ? 'Lancement...' : "Lancer l'entrainement"}
      </button>
    </form>
  )
}
