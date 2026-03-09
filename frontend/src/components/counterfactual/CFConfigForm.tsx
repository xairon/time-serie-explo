import { useState } from 'react'
import { ModelSelector } from '@/components/forecasting/ModelSelector'
import { useDatasets } from '@/hooks/useDatasets'

type CFMethod = 'physcf' | 'optuna' | 'comet'

interface CFConfigFormProps {
  onSubmit: (config: {
    model_id: string
    dataset_id: string
    method: CFMethod
    modifications: Record<string, number>
  }) => void
  isPending: boolean
}

const COVARIATES = ['precipitation', 'temperature', 'etp', 'debit', 'humidite']

export function CFConfigForm({ onSubmit, isPending }: CFConfigFormProps) {
  const { data: datasets } = useDatasets()
  const [modelId, setModelId] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [method, setMethod] = useState<CFMethod>('physcf')
  const [modifications, setModifications] = useState<Record<string, number>>(
    Object.fromEntries(COVARIATES.map((c) => [c, 1.0])),
  )

  const handleSlider = (key: string, value: number) => {
    setModifications((prev) => ({ ...prev, [key]: value }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!modelId || !datasetId) return
    onSubmit({ model_id: modelId, dataset_id: datasetId, method, modifications })
  }

  const methods: { key: CFMethod; label: string }[] = [
    { key: 'physcf', label: 'PhysCF' },
    { key: 'optuna', label: 'Optuna' },
    { key: 'comet', label: 'COMET' },
  ]

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h3 className="text-sm font-semibold text-text-primary">Configuration contrefactuelle</h3>

      <ModelSelector value={modelId} onChange={setModelId} />

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

      {/* Method tabs */}
      <div>
        <label className="block text-xs text-text-secondary mb-2">Méthode</label>
        <div className="flex border-b border-white/10">
          {methods.map((m) => (
            <button
              key={m.key}
              type="button"
              onClick={() => setMethod(m.key)}
              className={`px-3 py-1.5 text-xs transition-colors ${
                method === m.key
                  ? 'border-b-2 border-accent-cyan text-accent-cyan'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Perturbation sliders */}
      <div>
        <label className="block text-xs text-text-secondary mb-2">
          Perturbations (facteur multiplicatif)
        </label>
        <div className="space-y-3">
          {COVARIATES.map((cov) => (
            <div key={cov}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-text-secondary">{cov}</span>
                <span className="text-xs text-text-primary font-medium">
                  {(modifications[cov] ?? 1).toFixed(2)}x
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={3}
                step={0.05}
                value={modifications[cov] ?? 1}
                onChange={(e) => handleSlider(cov, Number(e.target.value))}
                className="w-full accent-accent-cyan h-1"
              />
            </div>
          ))}
        </div>
      </div>

      <button
        type="submit"
        disabled={isPending || !modelId || !datasetId}
        className="w-full bg-accent-cyan text-white px-4 py-2 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium"
      >
        {isPending ? 'Génération...' : 'Générer le contrefactuel'}
      </button>
    </form>
  )
}
