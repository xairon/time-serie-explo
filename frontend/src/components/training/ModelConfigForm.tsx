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
  const [inputChunk, setInputChunk] = useState(30)
  const [outputChunk, setOutputChunk] = useState(7)
  const [useCovariates, setUseCovariates] = useState(true)

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

  // Update hyperparams when model type changes — extract chunk lengths
  useEffect(() => {
    const model = availableModels?.find((m) => m.name === modelType)
    if (model) {
      const hp = { ...model.default_hyperparams }
      // Extract chunk lengths from hyperparams to top-level controls
      if (hp.input_chunk_length != null) {
        setInputChunk(Number(hp.input_chunk_length))
        delete hp.input_chunk_length
      }
      if (hp.output_chunk_length != null) {
        setOutputChunk(Number(hp.output_chunk_length))
        delete hp.output_chunk_length
      }
      setHyperparams(hp)
    }
  }, [modelType, availableModels])

  const selectedDataset = datasets?.find((d) => d.id === datasetId)
  const selectedModel = availableModels?.find((m) => m.name === modelType)

  // Default useCovariates based on dataset
  useEffect(() => {
    if (selectedDataset) {
      setUseCovariates(selectedDataset.covariates.length > 0)
    }
  }, [selectedDataset?.id])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({
      dataset_id: datasetId,
      model_name: modelType,
      hyperparams: {
        ...hyperparams,
        input_chunk_length: inputChunk,
        output_chunk_length: outputChunk,
      },
      train_ratio: trainSplit,
      val_ratio: valSplit,
      n_epochs: maxEpochs,
      early_stopping: earlyStopping,
      early_stopping_patience: earlyStopping ? patience : 0,
      station_name: station || undefined,
      use_covariates: useCovariates,
      loss_function: lossFunction,
    })
  }

  const updateHyperparam = (key: string, value: string) => {
    const num = Number(value)
    setHyperparams((prev) => ({ ...prev, [key]: isNaN(num) ? value : num }))
  }

  /** Determine if a hyperparam value is numeric */
  const isNumericParam = (val: unknown): boolean => {
    return typeof val === 'number'
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h3 className="text-sm font-semibold text-text-primary">Model configuration</h3>

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
            onChange={(e) => {
              setDatasetId(e.target.value)
              setStation('')
            }}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            {datasets?.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} ({d.n_rows} rows)
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Dataset info panel */}
      {selectedDataset && (
        <div className="bg-bg-hover rounded-lg p-3 space-y-1.5 border border-white/5">
          <h4 className="text-[10px] font-semibold text-text-secondary uppercase tracking-wide mb-1">
            Dataset information
          </h4>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <div>
              <span className="text-text-secondary">Target: </span>
              <span className="text-accent-cyan font-medium">{selectedDataset.target_variable}</span>
            </div>
            <div>
              <span className="text-text-secondary">Rows: </span>
              <span className="text-text-primary">{selectedDataset.n_rows.toLocaleString('en-US')}</span>
            </div>
            <div>
              <span className="text-text-secondary">Covariates: </span>
              <span className="text-text-primary">{selectedDataset.covariates.length}</span>
            </div>
            {selectedDataset.date_range.length >= 2 && (
              <div>
                <span className="text-text-secondary">Period: </span>
                <span className="text-text-primary">
                  {selectedDataset.date_range[0]?.slice(0, 10)} → {selectedDataset.date_range[1]?.slice(0, 10)}
                </span>
              </div>
            )}
            {selectedDataset.stations.length > 0 && (
              <div className="col-span-2">
                <span className="text-text-secondary">Stations: </span>
                <span className="text-text-primary">{selectedDataset.stations.length} station(s)</span>
              </div>
            )}
          </div>
          {selectedDataset.covariates.length > 0 && (
            <div className="mt-1">
              <p className="text-[10px] text-text-secondary">
                {selectedDataset.covariates.join(', ')}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Station */}
      {selectedDataset && selectedDataset.stations.length > 0 && (
        <div>
          <label className="block text-xs text-text-secondary mb-1">Station</label>
          <select
            value={station}
            onChange={(e) => setStation(e.target.value)}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">All stations</option>
            {selectedDataset.stations.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Covariate toggle */}
      {selectedDataset && selectedDataset.covariates.length > 0 && (
        <div>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useCovariates}
              onChange={(e) => setUseCovariates(e.target.checked)}
              className="w-4 h-4 rounded border-white/10 bg-bg-input text-accent-cyan focus:ring-accent-cyan/50"
            />
            <span className="text-xs text-text-secondary">
              Use covariates ({selectedDataset.covariates.length} features)
            </span>
          </label>
        </div>
      )}

      {/* Loss function */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Loss function</label>
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

      {/* Chunk lengths — common to all models */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-secondary mb-1">
            Input (context days)
          </label>
          <input
            type="number"
            min={1}
            max={365}
            value={inputChunk}
            onChange={(e) => setInputChunk(Number(e.target.value))}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          />
        </div>
        <div>
          <label className="block text-xs text-text-secondary mb-1">
            Output (prediction horizon)
          </label>
          <input
            type="number"
            min={1}
            max={365}
            value={outputChunk}
            onChange={(e) => setOutputChunk(Number(e.target.value))}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          />
        </div>
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
            <label className="block text-xs text-text-secondary mb-1">Patience (epochs)</label>
            <input
              type="number"
              min={3}
              max={50}
              value={patience}
              onChange={(e) => setPatience(Number(e.target.value))}
              className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
            />
          </div>
        )}
      </div>

      {/* Dynamic hyperparams with proper number inputs */}
      {Object.keys(hyperparams).length > 0 && (
        <div>
          <label className="block text-xs text-text-secondary mb-2">Hyperparameters</label>
          <div className="space-y-2">
            {Object.entries(hyperparams).map(([key, val]) => {
              const isNum = isNumericParam(val)
              const numVal = Number(val)
              // Determine appropriate step for numeric params
              const step = isNum && numVal < 1 ? 0.001 : isNum && numVal < 10 ? 1 : 1
              const label = key
                .replace(/_/g, ' ')
                .replace(/\b\w/g, (c) => c.toUpperCase())

              return (
                <div key={key} className="flex items-center gap-2">
                  <label className="text-xs text-text-secondary w-36 shrink-0" title={key}>
                    {label}
                  </label>
                  {isNum ? (
                    <input
                      type="number"
                      step={step}
                      value={String(val)}
                      onChange={(e) => updateHyperparam(key, e.target.value)}
                      className="flex-1 bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-1.5 text-xs"
                    />
                  ) : (
                    <input
                      type="text"
                      value={String(val)}
                      onChange={(e) => updateHyperparam(key, e.target.value)}
                      className="flex-1 bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-1.5 text-xs"
                    />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      <button
        type="submit"
        disabled={isPending || !modelType || !datasetId}
        className="w-full bg-accent-cyan text-white px-4 py-2 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium"
      >
        {isPending ? 'Starting...' : 'Start training'}
      </button>
    </form>
  )
}
