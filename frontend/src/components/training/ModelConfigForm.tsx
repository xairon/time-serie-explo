import { useState, useEffect } from 'react'
import { useAvailableModels } from '@/hooks/useModels'
import { useDatasets } from '@/hooks/useDatasets'
import type { TrainingConfig } from '@/lib/types'

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
  const [patience, setPatience] = useState(10)
  const [trainSplit, setTrainSplit] = useState(0.7)
  const [valSplit, setValSplit] = useState(0.15)
  const [hyperparams, setHyperparams] = useState<Record<string, unknown>>({})

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({
      dataset_id: datasetId,
      model_type: modelType,
      hyperparams,
      train_split: trainSplit,
      val_split: valSplit,
      max_epochs: maxEpochs,
      early_stopping_patience: patience,
      station: station || null,
    })
  }

  const updateHyperparam = (key: string, value: string) => {
    const num = Number(value)
    setHyperparams((prev) => ({ ...prev, [key]: isNaN(num) ? value : num }))
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h3 className="text-sm font-semibold text-text-primary">Configuration du modèle</h3>

      {/* Model type */}
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
            {availableModels?.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name} — {m.category}
              </option>
            ))}
          </select>
        )}
        {availableModels?.find((m) => m.name === modelType)?.description && (
          <p className="text-[10px] text-text-secondary mt-1">
            {availableModels.find((m) => m.name === modelType)!.description}
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

      {/* Epochs / Patience */}
      <div className="grid grid-cols-2 gap-3">
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
      </div>

      {/* Dynamic hyperparams */}
      {Object.keys(hyperparams).length > 0 && (
        <div>
          <label className="block text-xs text-text-secondary mb-2">Hyperparamètres</label>
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
        {isPending ? 'Lancement...' : "Lancer l'entraînement"}
      </button>
    </form>
  )
}
