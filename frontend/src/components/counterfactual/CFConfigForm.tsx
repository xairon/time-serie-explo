import { useState } from 'react'
import { ChevronDown, ChevronRight, Play } from 'lucide-react'

type CFMethod = 'physcf' | 'optuna' | 'comet'

const IPS_CLASS_KEYS = [
  'very_low',
  'low',
  'moderately_low',
  'normal',
  'moderately_high',
  'high',
  'very_high',
] as const

const IPS_LABELS: Record<string, string> = {
  very_low: 'Tres bas',
  low: 'Bas',
  moderately_low: 'Moderement bas',
  normal: 'Normal',
  moderately_high: 'Moderement haut',
  high: 'Haut',
  very_high: 'Tres haut',
}

const IPS_COLORS: Record<string, string> = {
  very_low: '#d73027',
  low: '#fc8d59',
  moderately_low: '#fee08b',
  normal: '#ffffbf',
  moderately_high: '#d9ef8b',
  high: '#91cf60',
  very_high: '#1a9850',
}

export interface CFFormData {
  model_id: string
  start_idx: number
  method: string
  target_ips_class: string
  from_ips_class: string
  to_ips_class: string
  lambda_prox: number
  n_iter: number
  lr: number
  cc_rate: number
  n_trials: number
  k_sigma: number
  lambda_smooth: number
}

interface CFConfigFormProps {
  modelId: string
  startIdx: number
  ipsWindow: number
  onIpsWindowChange: (w: number) => void
  onSubmit: (config: CFFormData) => void
  isPending: boolean
}

export function CFConfigForm({
  modelId,
  startIdx,
  ipsWindow,
  onIpsWindowChange,
  onSubmit,
  isPending,
}: CFConfigFormProps) {
  const [method, setMethod] = useState<CFMethod>('physcf')
  const [fromIps, setFromIps] = useState('normal')
  const [toIps, setToIps] = useState('moderately_low')
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Hyperparams with defaults per method
  const [lambdaProx, setLambdaProx] = useState(0.1)
  const [nIter, setNIter] = useState(500)
  const [lr, setLr] = useState(0.02)
  const [nTrials, setNTrials] = useState(200)
  const [kSigma, setKSigma] = useState(4)
  const [lambdaSmooth, setLambdaSmooth] = useState(0.1)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!modelId) return
    onSubmit({
      model_id: modelId,
      start_idx: startIdx,
      method,
      target_ips_class: toIps,
      from_ips_class: fromIps,
      to_ips_class: toIps,
      lambda_prox: lambdaProx,
      n_iter: nIter,
      lr,
      cc_rate: 0.07,
      n_trials: nTrials,
      k_sigma: kSigma,
      lambda_smooth: lambdaSmooth,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* IPS Window */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Fenetre IPS</label>
        <select
          value={ipsWindow}
          onChange={(e) => onIpsWindowChange(Number(e.target.value))}
          className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
        >
          {[1, 3, 6, 12].map((w) => (
            <option key={w} value={w}>IPS-{w}</option>
          ))}
        </select>
      </div>

      {/* IPS Transition */}
      <div>
        <label className="block text-xs text-text-secondary mb-2">Transition IPS</label>
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
          <IPSSelect value={fromIps} onChange={setFromIps} label="De" />
          <span className="text-text-secondary text-lg mt-5">→</span>
          <IPSSelect value={toIps} onChange={setToIps} label="Vers" />
        </div>
      </div>

      {/* Method */}
      <div>
        <label className="block text-xs text-text-secondary mb-2">Methode</label>
        <div className="flex border-b border-white/10">
          {(['physcf', 'optuna', 'comet'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMethod(m)}
              className={`px-3 py-1.5 text-xs transition-colors ${
                method === m
                  ? 'border-b-2 border-accent-cyan text-accent-cyan'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              {m === 'physcf' ? 'PhysCF' : m === 'optuna' ? 'Optuna' : 'COMET'}
            </button>
          ))}
        </div>
      </div>

      {/* Advanced hyperparams (collapsible) */}
      <div className="bg-bg-hover/30 rounded-lg">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-3 py-2 flex items-center gap-2 text-xs text-text-secondary hover:text-text-primary transition-colors"
        >
          {showAdvanced ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          Hyperparametres avances
        </button>
        {showAdvanced && (
          <div className="px-3 pb-3 space-y-3">
            {method === 'physcf' && (
              <>
                <SliderParam label="lambda_prox" value={lambdaProx} min={0.001} max={2.0} step={0.001} onChange={setLambdaProx} />
                <SliderParam label="n_iter" value={nIter} min={50} max={2000} step={10} onChange={setNIter} integer />
                <SliderParam label="lr" value={lr} min={0.001} max={0.1} step={0.001} onChange={setLr} />
              </>
            )}
            {method === 'optuna' && (
              <SliderParam label="n_trials" value={nTrials} min={50} max={2000} step={10} onChange={setNTrials} integer />
            )}
            {method === 'comet' && (
              <>
                <SliderParam label="k_sigma" value={kSigma} min={1} max={10} step={1} onChange={setKSigma} integer />
                <SliderParam label="lambda_smooth" value={lambdaSmooth} min={0.01} max={1.0} step={0.01} onChange={setLambdaSmooth} />
              </>
            )}
          </div>
        )}
      </div>

      {/* Generate button */}
      <button
        type="submit"
        disabled={isPending || !modelId}
        className="w-full bg-accent-cyan text-white px-4 py-2.5 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium flex items-center justify-center gap-2"
      >
        <Play className="w-4 h-4" />
        {isPending ? 'Generation...' : 'Generer le contrefactuel'}
      </button>
    </form>
  )
}

function IPSSelect({ value, onChange, label }: { value: string; onChange: (v: string) => void; label: string }) {
  return (
    <div>
      <label className="block text-[10px] text-text-secondary mb-1 uppercase">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-2 py-1.5 text-xs"
        style={{ borderColor: IPS_COLORS[value] ? `${IPS_COLORS[value]}80` : undefined }}
      >
        {IPS_CLASS_KEYS.map((cls) => (
          <option key={cls} value={cls}>
            {IPS_LABELS[cls]}
          </option>
        ))}
      </select>
      {value && (
        <div
          className="mt-1 h-1 rounded-full"
          style={{ backgroundColor: IPS_COLORS[value] }}
        />
      )}
    </div>
  )
}

function SliderParam({ label, value, min, max, step, onChange, integer }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; integer?: boolean
}) {
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
