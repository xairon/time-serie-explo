import { useTranslation } from 'react-i18next'
import { usePastasOptions } from '@/hooks/usePastas'

function getDescriptions(t: (k: string) => string): Record<string, Record<string, string>> {
  return {
    recharge: {
      Linear: t('pastas.configForm.rechargeLinear'),
      FlexModel: t('pastas.configForm.rechargeFlex'),
      Berendrecht: t('pastas.configForm.rechargeBerendrecht'),
      Peterson: t('pastas.configForm.rechargePeterson'),
    },
    response: {
      Gamma: t('pastas.configForm.respGamma'),
      Exponential: t('pastas.configForm.respExp'),
      Hantush: t('pastas.configForm.respHantush'),
      DoubleExponential: t('pastas.configForm.respDoubleExp'),
      FourParam: t('pastas.configForm.respFourParam'),
    },
    noise: {
      ArNoiseModel: t('pastas.configForm.noiseAr'),
      ArmaNoiseModel: t('pastas.configForm.noiseArma'),
      none: t('pastas.configForm.noiseNone'),
    },
    solver: {
      LeastSquares: t('pastas.configForm.solverLs'),
      Lmfit: t('pastas.configForm.solverLmfit'),
    },
  }
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
  const { t } = useTranslation()
  const DESCRIPTIONS = getDescriptions(t)
  const { data: options, isLoading } = usePastasOptions()

  const selectClass =
    'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50 disabled:opacity-50'

  return (
    <div className="space-y-4">
      <ConfigSelect
        label={t('pastas.fit.rechargeModel')}
        tooltip={t('pastas.fit.rechargeTooltip')}
        value={recharge}
        onChange={onRechargeChange}
        options={options?.recharge ?? ['Linear']}
        descriptions={DESCRIPTIONS.recharge}
        className={selectClass}
        disabled={isLoading}
      />

      <ConfigSelect
        label={t('pastas.fit.responseFunction')}
        tooltip={t('pastas.fit.responseFunctionTooltip')}
        value={response}
        onChange={onResponseChange}
        options={options?.response ?? ['Gamma']}
        descriptions={DESCRIPTIONS.response}
        className={selectClass}
        disabled={isLoading}
      />

      <div className="grid grid-cols-2 gap-4">
        <ConfigSelect
          label={t('pastas.fit.noiseModel')}
          tooltip={t('pastas.fit.noiseTooltip')}
          value={noise}
          onChange={onNoiseChange}
          options={options?.noise ?? ['ArNoiseModel', 'none']}
          descriptions={DESCRIPTIONS.noise}
          className={selectClass}
          disabled={isLoading}
        />

        <ConfigSelect
          label={t('pastas.fit.solver')}
          tooltip={t('pastas.fit.solverTooltip')}
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
            {t('pastas.fit.calibStart')}
          </label>
          <input
            type="date"
            value={tmin}
            onChange={(e) => onTminChange(e.target.value)}
            className={selectClass}
          />
          <p className="text-xs text-text-muted mt-1">{t('pastas.fit.emptyStart')}</p>
        </div>
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">
            {t('pastas.fit.calibEnd')}
          </label>
          <input
            type="date"
            value={tmax}
            onChange={(e) => onTmaxChange(e.target.value)}
            className={selectClass}
          />
          <p className="text-xs text-text-muted mt-1">{t('pastas.fit.emptyEnd')}</p>
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
