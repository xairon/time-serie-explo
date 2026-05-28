import { useTranslation } from 'react-i18next'

interface Props {
  mode: 'guided' | 'expert'
  onChange: (mode: 'guided' | 'expert') => void
}

export function GuidedExpertToggle({ mode, onChange }: Props) {
  const { t } = useTranslation()
  const OPTIONS: { value: 'guided' | 'expert'; label: string }[] = [
    { value: 'guided', label: t('pastas.calibrate.guidedMode') },
    { value: 'expert', label: t('pastas.calibrate.expertMode') },
  ]
  return (
    <div className="inline-flex items-center rounded-full border border-white/10 bg-bg-card p-0.5 gap-0.5">
      {OPTIONS.map(({ value, label }) => (
        <button
          key={value}
          onClick={() => onChange(value)}
          className={`text-xs py-1 px-3 rounded-full transition-colors ${
            mode === value
              ? 'bg-accent-cyan/20 text-accent-cyan'
              : 'text-text-muted hover:text-text-secondary'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  )
}
