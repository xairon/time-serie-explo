import { usePastasOptions } from '@/hooks/usePastas'

const DESCRIPTIONS: Record<string, Record<string, string>> = {
  recharge: {
    Linear: 'P − f·E: linear excess precipitation (von Asmuth 2002)',
    FlexModel: 'Full soil water balance: root zone, interception, optional snow',
    Berendrecht: 'Non-linear: exponential storage-runoff relationship (Berendrecht 2006)',
    Peterson: 'Non-linear: power law, logarithmic transforms (Peterson 2014)',
  },
  response: {
    Gamma: '3 parameters (A, a, n) — delayed response, most common',
    Exponential: '2 parameters (A, a) — simple decay, fast response',
    Hantush: '3 parameters — confined aquifer, includes leaky factor',
    DoubleExponential: '4 parameters — two response times (karst: conduits + matrix)',
    FourParam: '4 parameters (A, a, b, n) — very flexible, complex behavior',
  },
  noise: {
    ArNoiseModel: 'AR(1) — corrects residual autocorrelation',
    ArmaNoiseModel: 'ARMA(1,1) — better noise modeling (recommended for karst)',
    none: 'No noise model — raw residuals',
  },
  solver: {
    LeastSquares: 'Least squares (scipy) — fast, deterministic, default',
    Lmfit: 'Levenberg-Marquardt (lmfit) — better bounds handling',
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
        tooltip="How recharge (P − E) is computed before convolution"
        value={recharge}
        onChange={onRechargeChange}
        options={options?.recharge ?? ['Linear']}
        descriptions={DESCRIPTIONS.recharge}
        className={selectClass}
        disabled={isLoading}
      />

      <ConfigSelect
        label="Response function"
        tooltip="Shape of the aquifer impulse response to stress"
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
          tooltip="Stochastic model on residuals"
          value={noise}
          onChange={onNoiseChange}
          options={options?.noise ?? ['ArNoiseModel', 'none']}
          descriptions={DESCRIPTIONS.noise}
          className={selectClass}
          disabled={isLoading}
        />

        <ConfigSelect
          label="Solver"
          tooltip="Parameter optimization algorithm"
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
          <p className="text-xs text-text-muted mt-1">Empty = start of data</p>
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
          <p className="text-xs text-text-muted mt-1">Empty = end of data</p>
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
