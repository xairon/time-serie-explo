import { useState } from 'react'
import { Cpu, Zap } from 'lucide-react'
import { ModelSelector } from '@/components/forecasting/ModelSelector'
import { useDatasets } from '@/hooks/useDatasets'

type CFMethod = 'physcf' | 'optuna' | 'comet'

export interface CFFormConfig {
  model_id: string
  dataset_id: string
  method: CFMethod
  target_ips_class?: string
  from_ips_class?: string
  to_ips_class?: string
  modifications: Record<string, number>
  // PhysCF hyperparams
  lambda_prox: number
  n_iter: number
  lr: number
  // Optuna hyperparams
  n_trials: number
  // COMET hyperparams
  k_sigma: number
  lambda_smooth: number
  // Common
  cc_rate: number
  device: string
  seed: number
}

interface CFConfigFormProps {
  onSubmit: (config: CFFormConfig) => void
  isPending: boolean
}

const IPS_CLASSES = [
  'Très bas',
  'Bas',
  'Modérément bas',
  'Normal',
  'Modérément haut',
  'Haut',
  'Très haut',
] as const

const COVARIATES = ['precipitation', 'temperature', 'etp', 'debit', 'humidite']

export function CFConfigForm({ onSubmit, isPending }: CFConfigFormProps) {
  const { data: datasets } = useDatasets()
  const [modelId, setModelId] = useState('')
  const [datasetId, setDatasetId] = useState('')
  const [method, setMethod] = useState<CFMethod>('physcf')
  const [fromIpsClass, setFromIpsClass] = useState<string>('')
  const [toIpsClass, setToIpsClass] = useState<string>('')
  const [modifications, setModifications] = useState<Record<string, number>>(
    Object.fromEntries(COVARIATES.map((c) => [c, 1.0])),
  )

  // PhysCF hyperparams
  const [lambdaProx, setLambdaProx] = useState(0.1)
  const [nIter, setNIter] = useState(500)
  const [lr, setLr] = useState(0.02)

  // Optuna hyperparams
  const [nTrials, setNTrials] = useState(200)

  // COMET hyperparams
  const [kSigma, setKSigma] = useState(4)
  const [lambdaSmooth, setLambdaSmooth] = useState(0.1)

  // Common
  const [ccRate, setCcRate] = useState(0.07)
  const [device, setDevice] = useState('auto')
  const [seed, setSeed] = useState(42)

  const handleSlider = (key: string, value: number) => {
    setModifications((prev) => ({ ...prev, [key]: value }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!modelId || !datasetId) return
    onSubmit({
      model_id: modelId,
      dataset_id: datasetId,
      method,
      target_ips_class: toIpsClass || undefined,
      from_ips_class: fromIpsClass || undefined,
      to_ips_class: toIpsClass || undefined,
      modifications,
      lambda_prox: lambdaProx,
      n_iter: nIter,
      lr,
      n_trials: nTrials,
      k_sigma: kSigma,
      lambda_smooth: lambdaSmooth,
      cc_rate: ccRate,
      device,
      seed,
    })
  }

  const methods: { key: CFMethod; label: string }[] = [
    { key: 'physcf', label: 'PhysCF' },
    { key: 'optuna', label: 'Optuna' },
    { key: 'comet', label: 'COMET' },
  ]

  // Detect device display
  const deviceLabel = device === 'auto' ? 'Auto' : device === 'cuda' ? 'CUDA' : 'CPU'
  const DeviceIcon = device === 'cpu' ? Cpu : Zap

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
          <option value="">Selectionner</option>
          {datasets?.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
      </div>

      {/* Method tabs */}
      <div>
        <label className="block text-xs text-text-secondary mb-2">Methode</label>
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

      {/* IPS Transition selector */}
      <div>
        <label className="block text-xs text-text-secondary mb-2">Transition IPS</label>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-[10px] text-text-secondary mb-1 uppercase">De</label>
            <select
              value={fromIpsClass}
              onChange={(e) => setFromIpsClass(e.target.value)}
              className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-2 py-1.5 text-xs"
            >
              <option value="">Aucune</option>
              {IPS_CLASSES.map((cls) => (
                <option key={cls} value={cls}>
                  {cls}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-[10px] text-text-secondary mb-1 uppercase">Vers</label>
            <select
              value={toIpsClass}
              onChange={(e) => setToIpsClass(e.target.value)}
              className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-2 py-1.5 text-xs"
            >
              <option value="">Aucune</option>
              {IPS_CLASSES.map((cls) => (
                <option key={cls} value={cls}>
                  {cls}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Method-specific hyperparameters */}
      <div className="bg-bg-hover/30 rounded-lg p-3 space-y-3">
        <label className="block text-xs text-text-secondary font-medium">
          Hyperparametres ({methods.find((m) => m.key === method)?.label})
        </label>

        {method === 'physcf' && (
          <>
            <SliderParam
              label="lambda_prox"
              value={lambdaProx}
              min={0.001}
              max={2.0}
              step={0.001}
              onChange={setLambdaProx}
            />
            <SliderParam
              label="n_iter"
              value={nIter}
              min={50}
              max={2000}
              step={10}
              onChange={setNIter}
              integer
            />
            <SliderParam
              label="lr (learning rate)"
              value={lr}
              min={0.001}
              max={0.1}
              step={0.001}
              onChange={setLr}
            />
          </>
        )}

        {method === 'optuna' && (
          <SliderParam
            label="n_trials"
            value={nTrials}
            min={50}
            max={2000}
            step={10}
            onChange={setNTrials}
            integer
          />
        )}

        {method === 'comet' && (
          <>
            <SliderParam
              label="k_sigma"
              value={kSigma}
              min={1}
              max={10}
              step={1}
              onChange={setKSigma}
              integer
            />
            <SliderParam
              label="lambda_smooth"
              value={lambdaSmooth}
              min={0.01}
              max={1.0}
              step={0.01}
              onChange={setLambdaSmooth}
            />
          </>
        )}
      </div>

      {/* Device selector */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Device</label>
        <div className="flex items-center gap-2">
          <select
            value={device}
            onChange={(e) => setDevice(e.target.value)}
            className="flex-1 bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            <option value="auto">Auto</option>
            <option value="cuda">CUDA (GPU)</option>
            <option value="cpu">CPU</option>
          </select>
          <span
            className={`inline-flex items-center gap-1 px-2 py-1.5 rounded-lg text-xs font-medium border ${
              device === 'cpu'
                ? 'bg-bg-hover border-white/10 text-text-secondary'
                : 'bg-accent-green/10 border-accent-green/20 text-accent-green'
            }`}
          >
            <DeviceIcon className="w-3 h-3" />
            {deviceLabel}
          </span>
        </div>
      </div>

      {/* Seed */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Seed</label>
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(Number(e.target.value))}
          className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
        />
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
        {isPending ? 'Generation...' : 'Generer le contrefactuel'}
      </button>
    </form>
  )
}

// --- Reusable slider param component ---

interface SliderParamProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  integer?: boolean
}

function SliderParam({ label, value, min, max, step, onChange, integer }: SliderParamProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] text-text-secondary">{label}</span>
        <span className="text-[11px] text-text-primary font-mono">
          {integer ? Math.round(value) : value.toFixed(3)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-accent-cyan h-1"
      />
    </div>
  )
}
