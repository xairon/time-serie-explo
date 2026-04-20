import { useCallback } from 'react'
import { Plus, Trash2, Upload } from 'lucide-react'

export interface StressConfig {
  type: 'well' | 'river' | 'custom'
  name: string
  rfunc: string
  csv_rows: Array<{ date: string; value: number }>
}

const TYPE_OPTIONS = [
  { value: 'well', label: 'Pumping well', description: 'Q (m³/j)' },
  { value: 'river', label: 'River level', description: 'H (m)' },
  { value: 'custom', label: 'Custom', description: 'Any stress' },
] as const

interface Props {
  stresses: StressConfig[]
  onChange: (stresses: StressConfig[]) => void
}

export function StressListEditor({ stresses, onChange }: Props) {
  const add = () => {
    onChange([...stresses, { type: 'well', name: `stress_${stresses.length + 1}`, rfunc: 'Exponential', csv_rows: [] }])
  }

  const update = (i: number, s: StressConfig) => {
    const next = [...stresses]
    next[i] = s
    onChange(next)
  }

  const remove = (i: number) => {
    onChange(stresses.filter((_, idx) => idx !== i))
  }

  return (
    <div className="space-y-3">
      {stresses.map((s, i) => (
        <StressCard key={i} stress={s} onChange={v => update(i, v)} onDelete={() => remove(i)} />
      ))}
      <button onClick={add}
        className="w-full flex items-center justify-center gap-2 border border-dashed border-white/10 hover:border-white/20 rounded-lg px-4 py-2 text-sm text-text-muted hover:text-text-secondary transition-colors">
        <Plus className="w-4 h-4" />
        Add stress
      </button>
    </div>
  )
}

function StressCard({ stress, onChange, onDelete }: {
  stress: StressConfig; onChange: (s: StressConfig) => void; onDelete: () => void
}) {
  const handleFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const text = await file.text()
    const lines = text.trim().split('\n')
    const rows = lines.slice(1).map(line => {
      const parts = line.split(/[,;\t]/)
      return { date: parts[0].trim(), value: parseFloat(parts[1]) }
    }).filter(r => !isNaN(r.value))
    onChange({ ...stress, csv_rows: rows })
  }, [stress, onChange])

  return (
    <div className="bg-bg-card rounded-lg border border-white/5 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {TYPE_OPTIONS.map(t => (
            <button key={t.value}
              onClick={() => onChange({ ...stress, type: t.value })}
              className={`text-xs px-2 py-0.5 rounded border transition-colors ${
                stress.type === t.value
                  ? 'border-accent-cyan bg-accent-cyan/10 text-accent-cyan'
                  : 'border-white/10 text-text-muted hover:text-text-secondary'
              }`}>
              {t.label}
            </button>
          ))}
        </div>
        <button onClick={onDelete} className="p-1 hover:bg-bg-hover rounded text-text-muted hover:text-red-400">
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-text-muted">Name</label>
          <input type="text" value={stress.name}
            onChange={e => onChange({ ...stress, name: e.target.value })}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary" />
        </div>
        <div>
          <label className="text-xs text-text-muted">Response function</label>
          <select value={stress.rfunc}
            onChange={e => onChange({ ...stress, rfunc: e.target.value })}
            className="w-full bg-bg-primary border border-white/10 rounded px-2 py-1 text-sm text-text-primary">
            <option value="Exponential">Exponential</option>
            <option value="Gamma">Gamma</option>
            <option value="Hantush">Hantush</option>
          </select>
        </div>
      </div>

      <div>
        <label className="text-xs text-text-muted">
          CSV data ({TYPE_OPTIONS.find(t => t.value === stress.type)?.description})
        </label>
        <div className="flex items-center gap-2 mt-1">
          <label className="flex items-center gap-1 px-3 py-1.5 bg-bg-primary border border-white/10 rounded text-xs text-text-secondary hover:bg-bg-hover cursor-pointer transition-colors">
            <Upload className="w-3.5 h-3.5" />
            Upload CSV
            <input type="file" accept=".csv,.txt" onChange={handleFile} className="hidden" />
          </label>
          {stress.csv_rows.length > 0 && (
            <span className="text-xs text-accent-cyan">{stress.csv_rows.length} rows</span>
          )}
        </div>
      </div>
    </div>
  )
}
