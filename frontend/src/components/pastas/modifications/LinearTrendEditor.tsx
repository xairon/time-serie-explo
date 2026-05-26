import type { PumpingRange } from '@/lib/types'

interface LinearTrendData {
  type: 'linear_trend'
  start: string
  end: string
  slope_m_per_year: number
}

interface LinearTrendEditorProps {
  data: LinearTrendData
  onChange: (data: LinearTrendData) => void
  limits?: PumpingRange | null
}

const inputClass =
  'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-1.5 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

const QUICK_SLOPES = [
  { label: '−5 cm/an', value: -0.05 },
  { label: '−1 cm/an', value: -0.01 },
  { label: '+1 cm/an', value: 0.01 },
  { label: '+5 cm/an', value: 0.05 },
]

export function LinearTrendEditor({ data, onChange, limits }: LinearTrendEditorProps) {
  function update(patch: Partial<LinearTrendData>) {
    onChange({ ...data, ...patch })
  }

  const years = data.start && data.end
    ? Math.max(0, (new Date(data.end).getTime() - new Date(data.start).getTime()) / (365.25 * 86400000))
    : 0
  const totalChange = years * data.slope_m_per_year

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Début</label>
          <input type="date" value={data.start} onChange={(e) => update({ start: e.target.value })} className={inputClass} />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">Fin</label>
          <input type="date" value={data.end} onChange={(e) => update({ end: e.target.value })} className={inputClass} />
        </div>
      </div>

      <div>
        <label className="block text-xs text-text-muted mb-1" title="Vitesse de variation du niveau d'eau par an. Négatif = tendance baissière, positif = tendance haussière.">Pente (m/an)</label>
        <div className="flex gap-1 mb-2">
          {QUICK_SLOPES.map(s => (
            <button
              key={s.value}
              onClick={() => update({ slope_m_per_year: s.value })}
              className={`flex-1 px-1 py-1 rounded text-[10px] font-medium transition-colors ${
                Math.abs(data.slope_m_per_year - s.value) < 0.0001
                  ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
                  : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
        <input
          type="number"
          value={data.slope_m_per_year || ''}
          onChange={(e) => update({ slope_m_per_year: e.target.value === '' ? 0 : parseFloat(e.target.value) })}
          placeholder="ex. -0.01"
          className={inputClass}
          step="0.001"
          min={limits?.hard_min}
          max={limits?.hard_max}
        />
        {years > 0 && (
          <p className="text-[10px] mt-1">
            Sur {years.toFixed(1)} ans : variation totale de{' '}
            <span className={totalChange < 0 ? 'text-red-400' : 'text-green-400'}>
              {totalChange > 0 ? '+' : ''}{(totalChange * 100).toFixed(1)} cm
            </span>
          </p>
        )}
        {limits && (data.slope_m_per_year < limits.typical_min || data.slope_m_per_year > limits.typical_max) && (
          <p className="text-[10px] mt-1 text-yellow-400">
            Attention : pente inhabituelle — plage typique : {limits.typical_min} à {limits.typical_max} m/an
          </p>
        )}
      </div>
    </div>
  )
}

export type { LinearTrendData }
