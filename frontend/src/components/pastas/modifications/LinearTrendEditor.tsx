interface LinearTrendData {
  type: 'linear_trend'
  start: string
  end: string
  slope_m_per_year: number
}

interface LinearTrendEditorProps {
  data: LinearTrendData
  onChange: (data: LinearTrendData) => void
}

const inputClass =
  'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-1.5 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

export function LinearTrendEditor({ data, onChange }: LinearTrendEditorProps) {
  function update(patch: Partial<LinearTrendData>) {
    onChange({ ...data, ...patch })
  }

  return (
    <div className="space-y-3">
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
        <label className="block text-xs text-text-muted mb-1">Slope (m/year)</label>
        <input
          type="number"
          value={data.slope_m_per_year}
          onChange={(e) => update({ slope_m_per_year: parseFloat(e.target.value) || 0 })}
          className={inputClass}
          step="0.001"
        />
      </div>
    </div>
  )
}

export type { LinearTrendData }
