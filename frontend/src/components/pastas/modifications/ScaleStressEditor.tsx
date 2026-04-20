interface ScaleStressData {
  type: 'scale_stress'
  stress: 'precip' | 'evap'
  factor: number
  start: string
  end: string
}

interface ScaleStressEditorProps {
  data: ScaleStressData
  onChange: (data: ScaleStressData) => void
}

const inputClass =
  'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-1.5 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

export function ScaleStressEditor({ data, onChange }: ScaleStressEditorProps) {
  function update(patch: Partial<ScaleStressData>) {
    onChange({ ...data, ...patch })
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Stress</label>
          <select
            value={data.stress}
            onChange={(e) => update({ stress: e.target.value as 'precip' | 'evap' })}
            className={inputClass}
          >
            <option value="precip">Precipitation</option>
            <option value="evap">Evapotranspiration</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">Scale factor</label>
          <input
            type="number"
            value={data.factor}
            onChange={(e) => update({ factor: parseFloat(e.target.value) || 1 })}
            className={inputClass}
            step="0.01"
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
    </div>
  )
}

export type { ScaleStressData }
