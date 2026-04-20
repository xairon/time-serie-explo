import { usePastasOptions } from '@/hooks/usePastas'

const DESCRIPTIONS: Record<string, Record<string, string>> = {
  recharge: {
    Linear: 'P − f·E : excès de précipitation linéaire (von Asmuth 2002)',
    FlexModel: 'Bilan hydrique sol complet : zone racinaire, interception, neige optionnel',
    Berendrecht: 'Non-linéaire : relation exponentielle stockage-écoulement (Berendrecht 2006)',
    Peterson: 'Non-linéaire : loi puissance, transformations logarithmiques (Peterson 2014)',
  },
  response: {
    Gamma: '3 paramètres (A, a, n) — réponse retardée, le plus courant',
    Exponential: '2 paramètres (A, a) — décroissance simple, réponse rapide',
    Hantush: '3 paramètres — aquifère confiné, inclut le leaky factor',
    DoubleExponential: '4 paramètres — deux temps de réponse (karst : conduits + matrice)',
    FourParam: '4 paramètres (A, a, b, n) — très flexible, comportements complexes',
  },
  noise: {
    ArNoiseModel: 'AR(1) — corrige l\'autocorrélation des résidus',
    ArmaNoiseModel: 'ARMA(1,1) — meilleure modélisation du bruit (recommandé pour le karst)',
    none: 'Pas de modèle de bruit — résidus bruts',
  },
  solver: {
    LeastSquares: 'Moindres carrés (scipy) — rapide, déterministe, par défaut',
    Lmfit: 'Levenberg-Marquardt (lmfit) — meilleure gestion des bornes',
  },
}

interface PastasConfigFormProps {
  recharge: string
  onRechargeChange: (v: string) => void
  response: string
  onResponseChange: (v: string) => void
  noise: string
  onNoiseChange: (v: string) => void
  solver: string
  onSolverChange: (v: string) => void
  tmin: string
  onTminChange: (v: string) => void
  tmax: string
  onTmaxChange: (v: string) => void
}

export function PastasConfigForm({
  recharge, onRechargeChange,
  response, onResponseChange,
  noise, onNoiseChange,
  solver, onSolverChange,
  tmin, onTminChange,
  tmax, onTmaxChange,
}: PastasConfigFormProps) {
  const { data: options, isLoading } = usePastasOptions()

  const selectClass =
    'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50 disabled:opacity-50'

  return (
    <div className="space-y-4">
      <ConfigSelect
        label="Recharge model"
        tooltip="Comment la recharge (P − E) est calculée avant convolution"
        value={recharge}
        onChange={onRechargeChange}
        options={options?.recharge ?? ['Linear']}
        descriptions={DESCRIPTIONS.recharge}
        className={selectClass}
        disabled={isLoading}
      />

      <ConfigSelect
        label="Response function"
        tooltip="Forme de la réponse impulsionnelle de l'aquifère au stress"
        value={response}
        onChange={onResponseChange}
        options={options?.response ?? ['Gamma']}
        descriptions={DESCRIPTIONS.response}
        className={selectClass}
        disabled={isLoading}
      />

      <div className="grid grid-cols-2 gap-4">
        <ConfigSelect
          label="Noise model"
          tooltip="Modèle stochastique sur les résidus"
          value={noise}
          onChange={onNoiseChange}
          options={options?.noise ?? ['ArNoiseModel', 'none']}
          descriptions={DESCRIPTIONS.noise}
          className={selectClass}
          disabled={isLoading}
        />

        <ConfigSelect
          label="Solver"
          tooltip="Algorithme d'optimisation des paramètres"
          value={solver}
          onChange={onSolverChange}
          options={options?.solver ?? ['LeastSquares']}
          descriptions={DESCRIPTIONS.solver}
          className={selectClass}
          disabled={isLoading}
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Calibration start
          </label>
          <input
            type="date"
            value={tmin}
            onChange={(e) => onTminChange(e.target.value)}
            className={selectClass}
          />
          <p className="text-xs text-text-muted mt-1">Vide = début des données</p>
        </div>
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Calibration end
          </label>
          <input
            type="date"
            value={tmax}
            onChange={(e) => onTmaxChange(e.target.value)}
            className={selectClass}
          />
          <p className="text-xs text-text-muted mt-1">Vide = fin des données</p>
        </div>
      </div>
    </div>
  )
}

function ConfigSelect({
  label, tooltip, value, onChange, options, descriptions, className, disabled,
}: {
  label: string
  tooltip: string
  value: string
  onChange: (v: string) => void
  options: string[]
  descriptions: Record<string, string>
  className: string
  disabled: boolean
}) {
  const desc = descriptions[value]

  return (
    <div>
      <label className="block text-sm font-medium text-text-secondary mb-1" title={tooltip}>
        {label}
        <span className="ml-1 text-text-muted cursor-help" title={tooltip}>ⓘ</span>
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={className}
        disabled={disabled}
      >
        {options.map((opt) => (
          <option key={opt} value={opt} title={descriptions[opt] ?? ''}>
            {opt}
          </option>
        ))}
      </select>
      {desc && (
        <p className="text-xs text-text-muted mt-1">{desc}</p>
      )}
    </div>
  )
}
