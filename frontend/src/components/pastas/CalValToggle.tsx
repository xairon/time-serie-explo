interface Props {
  valSplit: number | null
  onChange: (v: number | null) => void
}

export function CalValToggle({ valSplit, onChange }: Props) {
  const enabled = valSplit !== null
  const pct = (valSplit ?? 0.3) * 100

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-text-secondary">Validation</label>
          <span className="ml-1 text-text-muted cursor-help" title="Hold out a portion of data to test the model on data it hasn't seen during training. Checks that the model generalizes well.">ⓘ</span>
        </div>
        <button
          onClick={() => onChange(enabled ? null : 0.3)}
          className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${
            enabled
              ? 'border-accent-cyan text-accent-cyan bg-accent-cyan/10'
              : 'border-white/10 text-text-muted'
          }`}
        >
          {enabled ? 'Enabled' : 'Disabled'}
        </button>
      </div>
      {!enabled && (
        <p className="text-xs text-text-muted">
          The model will train on the full period. Enable to hold out a portion for testing.
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
            <span>Training: {(100 - pct).toFixed(0)}% (early years)</span>
            <span>Test: {pct.toFixed(0)}% (late years)</span>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Test metrics show model quality on unseen data.
          </p>
        </div>
      )}
    </div>
  )
}
