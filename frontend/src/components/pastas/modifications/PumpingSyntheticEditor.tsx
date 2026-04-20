interface PumpingSyntheticData {
  type: 'pumping_synthetic'
  pattern: 'constant' | 'seasonal' | 'pulse'
  rate_m3d: number
  distance_m: number
  start: string
  end: string
  rfunc: 'Exponential' | 'Hantush'
}

interface PumpingSyntheticEditorProps {
  data: PumpingSyntheticData
  onChange: (data: PumpingSyntheticData) => void
}

const PATTERNS = ['constant', 'seasonal', 'pulse'] as const
const RFUNCS = ['Exponential', 'Hantush'] as const

const inputClass =
  'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-1.5 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

export function PumpingSyntheticEditor({ data, onChange }: PumpingSyntheticEditorProps) {
  function update(patch: Partial<PumpingSyntheticData>) {
    onChange({ ...data, ...patch })
  }

  return (
    <div className="space-y-3">
      {/* Pattern buttons */}
      <div>
        <label className="block text-xs text-text-muted mb-1">Pattern</label>
        <div className="flex gap-1">
          {PATTERNS.map((p) => (
            <button
              key={p}
              onClick={() => update({ pattern: p })}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                data.pattern === p
                  ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
                  : 'bg-bg-primary text-text-muted border border-white/10 hover:text-text-secondary'
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Rate (m³/d)</label>
          <input
            type="number"
            value={data.rate_m3d}
            onChange={(e) => update({ rate_m3d: parseFloat(e.target.value) || 0 })}
            className={inputClass}
            step="0.1"
            min="0"
          />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">Distance (m)</label>
          <input
            type="number"
            value={data.distance_m}
            onChange={(e) => update({ distance_m: parseFloat(e.target.value) || 0 })}
            className={inputClass}
            step="1"
            min="0"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Start date</label>
          <input
            type="date"
            value={data.start}
            onChange={(e) => update({ start: e.target.value })}
            className={inputClass}
          />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">End date</label>
          <input
            type="date"
            value={data.end}
            onChange={(e) => update({ end: e.target.value })}
            className={inputClass}
          />
        </div>
      </div>

      <div>
        <label className="block text-xs text-text-muted mb-1">Response function</label>
        <select
          value={data.rfunc}
          onChange={(e) => update({ rfunc: e.target.value as 'Exponential' | 'Hantush' })}
          className={inputClass}
        >
          {RFUNCS.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

export type { PumpingSyntheticData }
