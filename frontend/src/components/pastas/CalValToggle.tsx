import { useTranslation } from 'react-i18next'

interface Props {
  valSplit: number | null
  onChange: (v: number | null) => void
}

export function CalValToggle({ valSplit, onChange }: Props) {
  const { t } = useTranslation()
  const enabled = valSplit !== null
  const pct = (valSplit ?? 0.3) * 100

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-text-secondary">{t('pastas.validation.label')}</label>
          <span className="ml-1 text-text-muted cursor-help" title={t('pastas.validation.tooltip')}>ⓘ</span>
        </div>
        <button
          onClick={() => onChange(enabled ? null : 0.3)}
          className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${
            enabled
              ? 'border-accent-cyan text-accent-cyan bg-accent-cyan/10'
              : 'border-white/10 text-text-muted'
          }`}
        >
          {enabled ? t('common.active') : t('common.inactive')}
        </button>
      </div>
      {!enabled && (
        <p className="text-xs text-text-muted">
          {t('pastas.validation.disabledHint')}
        </p>
      )}
      {enabled && (
        <div>
          <input
            type="range"
            min={10}
            max={50}
            step={5}
            value={pct}
            onChange={(e) => onChange(+e.target.value / 100)}
            className="w-full accent-accent-cyan"
          />
          <div className="flex justify-between text-xs text-text-muted">
            <span>{t('pastas.validation.calibrationPct', { pct: (100 - pct).toFixed(0) })}</span>
            <span>{t('pastas.validation.testPct', { pct: pct.toFixed(0) })}</span>
          </div>
          <p className="text-xs text-text-muted mt-1">
            {t('pastas.validation.description')}
          </p>
        </div>
      )}
    </div>
  )
}
