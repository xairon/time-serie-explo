import type { PumpingRange } from '@/lib/types'

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
  limits?: PumpingRange | null
}

const inputClass =
  'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-1.5 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

const QUICK_FACTORS = [
  { label: '−30%', value: 0.7 },
  { label: '−20%', value: 0.8 },
  { label: '−10%', value: 0.9 },
  { label: '+10%', value: 1.1 },
  { label: '+20%', value: 1.2 },
  { label: '+30%', value: 1.3 },
]

export function ScaleStressEditor({ data, onChange, limits }: ScaleStressEditorProps) {
  function update(patch: Partial<ScaleStressData>) {
    onChange({ ...data, ...patch })
  }

  const pctChange = Math.round((data.factor - 1) * 100)
  const pctLabel = pctChange >= 0 ? `+${pctChange}%` : `${pctChange}%`
  const stressLabel = data.stress === 'precip' ? 'precipitation' : 'evapotranspiration (PET)'

  return (
    <div className="space-y-3">
      <div>
        <label className="block text-xs text-text-muted mb-1">Climate variable</label>
        <div className="flex gap-1.5">
          <button
            onClick={() => update({ stress: 'precip' })}
            className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              data.stress === 'precip'
                ? 'bg-blue-500/15 text-blue-400 border border-blue-500/30'
                : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10'
            }`}
          >
            Precipitation
          </button>
          <button
            onClick={() => update({ stress: 'evap' })}
            className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              data.stress === 'evap'
                ? 'bg-orange-500/15 text-orange-400 border border-orange-500/30'
                : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10'
            }`}
          >
            Evapotranspiration
          </button>
        </div>
      </div>

      <div>
        <label className="block text-xs text-text-muted mb-1">Scaling factor</label>
        <div className="flex gap-1 mb-2">
          {QUICK_FACTORS.map(f => (
            <button
              key={f.value}
              onClick={() => update({ factor: f.value })}
              className={`flex-1 px-1 py-1 rounded text-[10px] font-medium transition-colors ${
                Math.abs(data.factor - f.value) < 0.001
                  ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
                  : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <input
          type="number"
          value={data.factor}
          onChange={(e) => update({ factor: parseFloat(e.target.value) || 1 })}
          className={inputClass}
          step="0.05"
          min={limits?.hard_min ?? 0}
          max={limits?.hard_max}
        />
        <p className="text-[10px] mt-1">
          {data.factor === 1
            ? <span className="text-text-muted">No change</span>
            : <span className={pctChange < 0 ? 'text-red-400' : 'text-green-400'}>
                {pctLabel} of {stressLabel}
              </span>
          }
        </p>
        {limits && data.factor !== 1 && (data.factor < limits.typical_min || data.factor > limits.typical_max) && (
          <p className="text-[10px] mt-1 text-yellow-400">
            ⚠ Facteur inhabituel — plage typique : {limits.typical_min}–{limits.typical_max}
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Start</label>
          <input type="date" value={data.start} onChange={(e) => update({ start: e.target.value })} className={inputClass} />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">End</label>
          <input type="date" value={data.end} onChange={(e) => update({ end: e.target.value })} className={inputClass} />
        </div>
      </div>
    </div>
  )
}

export type { ScaleStressData }
