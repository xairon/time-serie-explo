import { useState, useMemo } from 'react'
import { ChevronDown, ChevronRight, ChevronLeft, Play, Loader2 } from 'lucide-react'
import {
  IPS_CLASS_ORDER,
  IPS_LABELS,
  IPS_COLORS,
  shiftClass,
  classIndex,
  mode,
  type MonthIps,
  type QualityVerdict,
} from '@/lib/ips'

export interface CFTargetData {
  model_id: string
  start_idx: number
  target_ips_classes: Record<string, string>
  lambda_prox: number
  n_iter: number
  lr: number
  cc_rate: number
  k_sigma: number
  lambda_smooth: number
}

interface CFTargetSelectorProps {
  modelId: string
  startIdx: number
  gtIps: MonthIps[]
  verdict: QualityVerdict
  isForecastLoading: boolean
  onSubmit: (data: CFTargetData) => void
  isPending: boolean
}

export function CFTargetSelector({
  modelId,
  startIdx,
  gtIps,
  verdict,
  isForecastLoading,
  onSubmit,
  isPending,
}: CFTargetSelectorProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [delta, setDelta] = useState(0)

  const [lambdaProx, setLambdaProx] = useState(0.1)
  const [nIter, setNIter] = useState(500)
  const [lr, setLr] = useState(0.02)
  const [kSigma, setKSigma] = useState(4)
  const [lambdaSmooth, setLambdaSmooth] = useState(0.1)

  const currentClasses = useMemo(() => gtIps.map((m) => m.cls), [gtIps])
  const currentMode = useMemo(() => mode(currentClasses), [currentClasses])
  const currentModeLabel = currentMode ? IPS_LABELS[currentMode] ?? currentMode : '\u2014'

  const minorityInfo = useMemo(() => {
    if (!currentMode || currentClasses.length === 0) return null
    const others = currentClasses.filter((c) => c !== currentMode)
    if (others.length === 0) return null
    const unique = [...new Set(others)]
    return unique.map((c) => `${others.filter((o) => o === c).length} mois: ${IPS_LABELS[c] ?? c}`).join(', ')
  }, [currentClasses, currentMode])

  const targetClasses = useMemo(() => {
    const result: Record<string, string> = {}
    for (const m of gtIps) {
      result[m.yearMonth] = shiftClass(m.cls, delta)
    }
    return result
  }, [gtIps, delta])

  const targetMode = useMemo(() => {
    const vals = Object.values(targetClasses)
    return mode(vals)
  }, [targetClasses])
  const targetModeLabel = targetMode ? IPS_LABELS[targetMode] ?? targetMode : '\u2014'

  const allAtMin = currentClasses.length > 0 && currentClasses.every((c) => classIndex(shiftClass(c, delta)) === 0)
  const allAtMax = currentClasses.length > 0 && currentClasses.every((c) => classIndex(shiftClass(c, delta)) === IPS_CLASS_ORDER.length - 1)

  const canSubmit = verdict !== 'not_qualified' && !isPending && !isForecastLoading && !!modelId && delta !== 0

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!canSubmit) return
    const targetByMonth: Record<string, string> = {}
    for (const m of gtIps) {
      targetByMonth[String(m.monthNumber)] = shiftClass(m.cls, delta)
    }
    onSubmit({
      model_id: modelId,
      start_idx: startIdx,
      target_ips_classes: targetByMonth,
      lambda_prox: lambdaProx,
      n_iter: nIter,
      lr,
      cc_rate: 0.07,
      k_sigma: kSigma,
      lambda_smooth: lambdaSmooth,
    })
  }

  if (isForecastLoading) {
    return (
      <div className="flex items-center gap-2 text-text-secondary text-sm py-4">
        <Loader2 className="w-4 h-4 animate-spin" />
        Verification en cours...
      </div>
    )
  }

  if (gtIps.length === 0) {
    return (
      <p className="text-[10px] text-text-secondary/40 italic py-2">
        Selectionnez une fenetre pour configurer la cible
      </p>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Current IPS display */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">IPS actuel de la fenetre</label>
        <div className="flex items-center gap-2">
          <span
            className="px-3 py-1.5 rounded-full text-sm font-medium"
            style={{
              backgroundColor: `${IPS_COLORS[currentMode ?? 'normal']}33`,
              color: IPS_COLORS[currentMode ?? 'normal'],
            }}
          >
            {currentModeLabel}
          </span>
          {minorityInfo && (
            <span className="text-[10px] text-text-secondary/60">({minorityInfo})</span>
          )}
        </div>
      </div>

      {/* Transition buttons */}
      <div>
        <label className="block text-xs text-text-secondary mb-2">Transition de classe</label>
        <div className="flex items-center gap-3">
          <button
            type="button"
            disabled={allAtMin}
            onClick={() => setDelta((d) => d - 1)}
            className="flex items-center gap-1 px-3 py-2 rounded-lg border border-white/10 bg-bg-hover/30 hover:bg-bg-hover/60 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-sm text-text-primary"
          >
            <ChevronLeft className="w-4 h-4" />
            Baisser
          </button>

          <div className="flex-1 text-center">
            {delta === 0 ? (
              <span className="text-xs text-text-secondary/50">Aucun changement</span>
            ) : (
              <span className="text-xs text-text-secondary">
                {delta > 0 ? '+' : ''}{delta} classe{Math.abs(delta) > 1 ? 's' : ''}
              </span>
            )}
          </div>

          <button
            type="button"
            disabled={allAtMax}
            onClick={() => setDelta((d) => d + 1)}
            className="flex items-center gap-1 px-3 py-2 rounded-lg border border-white/10 bg-bg-hover/30 hover:bg-bg-hover/60 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-sm text-text-primary"
          >
            Monter
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {delta !== 0 && (
          <button
            type="button"
            onClick={() => setDelta(0)}
            className="mt-1 text-[10px] text-text-secondary/50 hover:text-text-secondary underline"
          >
            Reinitialiser
          </button>
        )}
      </div>

      {/* Target display */}
      {delta !== 0 && targetMode && (
        <div className="bg-bg-hover/20 rounded-lg p-3">
          <span className="text-xs text-text-secondary">Cible :</span>
          <span
            className="ml-2 px-2.5 py-1 rounded-full text-sm font-medium"
            style={{
              backgroundColor: `${IPS_COLORS[targetMode]}33`,
              color: IPS_COLORS[targetMode],
            }}
          >
            {targetModeLabel}
          </span>
        </div>
      )}

      {verdict === 'partial' && (
        <p className="text-[10px] text-amber-400/80 bg-amber-500/10 rounded px-2 py-1.5">
          Attention : le modele diverge de l'observe sur certains mois. Les resultats contrefactuels peuvent etre moins fiables.
        </p>
      )}

      <p className="text-[10px] text-text-secondary/50 bg-bg-hover/20 rounded px-2 py-1.5">
        PhysCF (physique) et COMET (baseline) seront lances en parallele.
      </p>

      {/* Advanced hyperparams */}
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
            <p className="text-[10px] text-text-secondary/50 uppercase">PhysCF</p>
            <SliderParam label="lambda_prox" value={lambdaProx} min={0.001} max={2.0} step={0.001} onChange={setLambdaProx} />
            <SliderParam label="n_iter" value={nIter} min={50} max={2000} step={10} onChange={setNIter} integer />
            <SliderParam label="lr" value={lr} min={0.001} max={0.1} step={0.001} onChange={setLr} />
            <div className="border-t border-white/5 my-2" />
            <p className="text-[10px] text-text-secondary/50 uppercase">COMET</p>
            <SliderParam label="k_sigma" value={kSigma} min={1} max={10} step={1} onChange={setKSigma} integer />
            <SliderParam label="lambda_smooth" value={lambdaSmooth} min={0.01} max={1.0} step={0.01} onChange={setLambdaSmooth} />
          </div>
        )}
      </div>

      <button
        type="submit"
        disabled={!canSubmit}
        className="w-full bg-accent-cyan text-white px-4 py-2.5 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium flex items-center justify-center gap-2"
      >
        <Play className="w-4 h-4" />
        {isPending ? 'Generation...' : delta === 0 ? 'Choisir une direction' : 'Generer le contrefactuel'}
      </button>
    </form>
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
