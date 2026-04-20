interface Props {
  valSplit: number | null
  onChange: (v: number | null) => void
}

export function CalValToggle({ valSplit, onChange }: Props) {
  const enabled = valSplit !== null

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-text-secondary">Cal/Val split</label>
        <button
          onClick={() => onChange(enabled ? null : 0.3)}
          className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${
            enabled
              ? 'border-accent-cyan text-accent-cyan bg-accent-cyan/10'
              : 'border-white/10 text-text-muted'
          }`}
        >
          {enabled ? 'On' : 'Off'}
        </button>
      </div>
      {enabled && (
        <div>
          <input
            type="range"
            min={10}
            max={50}
            step={5}
            value={(valSplit ?? 0.3) * 100}
            onChange={(e) => onChange(+e.target.value / 100)}
            className="w-full accent-accent-cyan"
          />
          <div className="flex justify-between text-xs text-text-muted">
            <span>Cal: {((1 - (valSplit ?? 0.3)) * 100).toFixed(0)}%</span>
            <span>Val: {((valSplit ?? 0.3) * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}
    </div>
  )
}
