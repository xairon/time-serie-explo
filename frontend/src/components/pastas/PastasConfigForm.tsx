import { usePastasOptions } from '@/hooks/usePastas'

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
  recharge,
  onRechargeChange,
  response,
  onResponseChange,
  noise,
  onNoiseChange,
  solver,
  onSolverChange,
  tmin,
  onTminChange,
  tmax,
  onTmaxChange,
}: PastasConfigFormProps) {
  const { data: options, isLoading } = usePastasOptions()

  const selectClass =
    'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50 disabled:opacity-50'

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Recharge model
          </label>
          <select
            value={recharge}
            onChange={(e) => onRechargeChange(e.target.value)}
            className={selectClass}
            disabled={isLoading}
          >
            {(options?.recharge ?? ['Linear']).map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Response function
          </label>
          <select
            value={response}
            onChange={(e) => onResponseChange(e.target.value)}
            className={selectClass}
            disabled={isLoading}
          >
            {(options?.response ?? ['Gamma']).map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">Noise model</label>
          <select
            value={noise}
            onChange={(e) => onNoiseChange(e.target.value)}
            className={selectClass}
            disabled={isLoading}
          >
            {(options?.noise ?? ['ArmaModel']).map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">Solver</label>
          <select
            value={solver}
            onChange={(e) => onSolverChange(e.target.value)}
            className={selectClass}
            disabled={isLoading}
          >
            {(options?.solver ?? ['LeastSquares']).map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Calibration start (tmin)
          </label>
          <input
            type="date"
            value={tmin}
            onChange={(e) => onTminChange(e.target.value)}
            className={selectClass}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Calibration end (tmax)
          </label>
          <input
            type="date"
            value={tmax}
            onChange={(e) => onTmaxChange(e.target.value)}
            className={selectClass}
          />
        </div>
      </div>
    </div>
  )
}
