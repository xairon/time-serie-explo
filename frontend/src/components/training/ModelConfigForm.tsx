import { useState, useEffect, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { useAvailableModels, useTrainingPresets } from '@/hooks/useModels'
import { useDatasets } from '@/hooks/useDatasets'
import type { TrainingConfig, AvailableModel } from '@/lib/types'

const PRESET_NONE = ''

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
  const { t } = useTranslation()
  const { data: availableModels, isLoading: modelsLoading } = useAvailableModels()
  const { data: datasets, isLoading: datasetsLoading } = useDatasets()
  const { data: presets } = useTrainingPresets()

  const [presetId, setPresetId] = useState<string>(PRESET_NONE)
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

  const selectedPreset = useMemo(
    () => presets?.find((p) => p.id === presetId) ?? null,
    [presets, presetId],
  )

  // Applying a preset fills the form fields below. The user can still tweak
  // anything after — preset is a starting point, not a lock.
  const applyPreset = (id: string) => {
    setPresetId(id)
    if (!id) return
    const preset = presets?.find((p) => p.id === id)
    if (!preset) return
    // The backend exposes a subset of the full preset; full hyperparams live
    // there. Frontend mirrors the visible top-level fields and submits
    // preset_id so the backend fills in the rest server-side.
    setModelType(preset.model_name)
    setMaxEpochs(preset.n_epochs)
    setEarlyStopping(true)
  }

  // Group models by category
  const modelsByCategory = useMemo(() => {
    if (!availableModels) return new Map<string, AvailableModel[]>()
    const groups = new Map<string, AvailableModel[]>()
    for (const m of availableModels) {
      const cat = m.category || 'Other'
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

  // Update hyperparams + chunk lengths when the model OR the active preset
  // changes. A preset's input/output windows take priority over the model's
  // defaults, so picking a preset actually moves the "entrée/sortie" fields
  // (otherwise this effect would immediately overwrite them with model defaults).
  useEffect(() => {
    const model = availableModels?.find((m) => m.name === modelType)
    if (!model) return
    const hp = { ...model.default_hyperparams }
    delete hp.input_chunk_length
    delete hp.output_chunk_length
    setHyperparams(hp)
    const preset = presetId ? presets?.find((p) => p.id === presetId) : null
    if (preset && preset.model_name === modelType) {
      if (preset.input_chunk_length != null) setInputChunk(preset.input_chunk_length)
      setOutputChunk(preset.horizon_days)
    } else {
      if (model.default_hyperparams.input_chunk_length != null) setInputChunk(Number(model.default_hyperparams.input_chunk_length))
      if (model.default_hyperparams.output_chunk_length != null) setOutputChunk(Number(model.default_hyperparams.output_chunk_length))
    }
  }, [modelType, availableModels, presetId, presets])

  const selectedDataset = datasets?.find((d) => d.id === datasetId)
  const selectedModel = availableModels?.find((m) => m.name === modelType)

  // Default useCovariates based on dataset
  useEffect(() => {
    if (selectedDataset) {
      setUseCovariates(selectedDataset.covariates.length > 0)
    }
  }, [selectedDataset?.id])

  // Preselect the station when the dataset has exactly one — "toutes les
  // stations" makes no sense with a single station.
  useEffect(() => {
    if (selectedDataset?.stations.length === 1) setStation(selectedDataset.stations[0])
    else setStation('')
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
      preset_id: presetId || undefined,
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
      <h3 className="text-sm font-semibold text-text-primary">{t('sharedComponents.training.modelConfig')}</h3>

      {/* Preset selector — opinionated configs for non-experts */}
      <div className="bg-accent-cyan/5 border border-accent-cyan/20 rounded-lg p-3 space-y-2" data-tour="preset-selector">
        <label className="block text-xs font-semibold text-accent-cyan">
          {t('sharedComponents.training.preset')} <span className="text-text-secondary font-normal">{t('sharedComponents.training.presetRecommended')}</span>
        </label>
        <select
          value={presetId}
          onChange={(e) => applyPreset(e.target.value)}
          className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
        >
          <option value={PRESET_NONE}>{t('sharedComponents.training.manualConfig')}</option>
          {presets?.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
        {selectedPreset && (
          <p className="text-[11px] text-text-secondary leading-snug">
            {selectedPreset.description}
          </p>
        )}
        {!selectedPreset && (
          <p className="text-[11px] text-text-secondary leading-snug">
            {t('sharedComponents.training.presetHint')}
          </p>
        )}
      </div>

      {/* Model type - grouped by category */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">{t('sharedComponents.training.architecture')}</label>
        {modelsLoading ? (
          <div className="h-9 bg-bg-hover rounded-lg animate-pulse" />
        ) : (
          <select
            value={modelType}
            onChange={(e) => { setModelType(e.target.value); setPresetId(PRESET_NONE) }}
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
        <label className="block text-xs text-text-secondary mb-1">{t('sharedComponents.training.dataset')}</label>
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
                {d.name} ({d.n_rows} {t('sharedComponents.training.rowsUnit')})
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Dataset info panel */}
      {selectedDataset && (
        <div className="bg-bg-hover rounded-lg p-3 space-y-1.5 border border-white/5">
          <h4 className="text-[10px] font-semibold text-text-secondary uppercase tracking-wide mb-1">
            {t('sharedComponents.training.datasetInfo')}
          </h4>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <div>
              <span className="text-text-secondary">{t('sharedComponents.training.target')} : </span>
              <span className="text-accent-cyan font-medium">{selectedDataset.target_variable}</span>
            </div>
            <div>
              <span className="text-text-secondary">{t('sharedComponents.training.rows')} : </span>
              <span className="text-text-primary">{selectedDataset.n_rows.toLocaleString('en-US')}</span>
            </div>
            <div>
              <span className="text-text-secondary">{t('sharedComponents.training.covariates')} : </span>
              <span className="text-text-primary">{selectedDataset.covariates.length}</span>
            </div>
            {selectedDataset.date_range.length >= 2 && (
              <div>
                <span className="text-text-secondary">{t('sharedComponents.training.period')} : </span>
                <span className="text-text-primary">
                  {selectedDataset.date_range[0]?.slice(0, 10)} → {selectedDataset.date_range[1]?.slice(0, 10)}
                </span>
              </div>
            )}
            {selectedDataset.stations.length > 0 && (
              <div className="col-span-2">
                <span className="text-text-secondary">{t('sharedComponents.training.stations')} : </span>
                <span className="text-text-primary">{t('sharedComponents.training.stationsCount', { count: selectedDataset.stations.length })}</span>
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
          <label className="block text-xs text-text-secondary mb-1">{t('sharedComponents.training.station')}</label>
          <select
            value={station}
            onChange={(e) => setStation(e.target.value)}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">{t('sharedComponents.training.allStations')}</option>
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
              {t('sharedComponents.training.useCovariates', { count: selectedDataset.covariates.length })}
            </span>
          </label>
        </div>
      )}

      {/* Loss function */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">{t('sharedComponents.training.lossFunction')}</label>
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
            {t('sharedComponents.training.inputDays')}
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
            {t('sharedComponents.training.outputDays')}
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
          <label className="block text-xs text-text-secondary mb-1">{t('sharedComponents.training.trainSplit')}</label>
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
          <label className="block text-xs text-text-secondary mb-1">{t('sharedComponents.training.valSplit')}</label>
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
        <label className="block text-xs text-text-secondary mb-1">{t('sharedComponents.training.maxEpochs')}</label>
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
          <span className="text-xs text-text-secondary">{t('sharedComponents.training.earlyStopping')}</span>
        </label>
        {earlyStopping && (
          <div>
            <label className="block text-xs text-text-secondary mb-1">{t('sharedComponents.training.patience')}</label>
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
          <label className="block text-xs text-text-secondary mb-2">{t('sharedComponents.training.hyperparams')}</label>
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
        {isPending ? t('sharedComponents.training.starting') : t('sharedComponents.training.startTraining')}
      </button>
    </form>
  )
}
