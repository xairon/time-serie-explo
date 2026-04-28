interface Props {
  mode: 'guided' | 'expert'
  onChange: (mode: 'guided' | 'expert') => void
}

const OPTIONS: { value: 'guided' | 'expert'; label: string; disabled?: boolean }[] = [
  { value: 'guided', label: 'Guided' },
  { value: 'expert', label: 'Expert', disabled: true },
]

export function GuidedExpertToggle({ mode, onChange }: Props) {
  return (
    <div className="inline-flex items-center rounded-full border border-white/10 bg-bg-card p-0.5 gap-0.5">
      {OPTIONS.map(({ value, label, disabled }) => (
        <button
          key={value}
          onClick={() => !disabled && onChange(value)}
          disabled={disabled}
          className={`text-xs py-1 px-3 rounded-full transition-colors ${
            disabled
              ? 'text-text-muted/40 cursor-not-allowed'
              : mode === value
                ? 'bg-accent-cyan/20 text-accent-cyan'
                : 'text-text-muted hover:text-text-secondary'
          }`}
          title={disabled ? 'Coming soon' : undefined}
        >
          {label}
        </button>
      ))}
    </div>
  )
}
