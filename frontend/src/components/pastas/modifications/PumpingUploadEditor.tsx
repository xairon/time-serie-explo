import { useRef } from 'react'

interface PumpingRow {
  date: string
  Q_m3d: number
}

interface PumpingUploadData {
  type: 'pumping_upload'
  rows: PumpingRow[]
  distance_m: number
  rfunc: 'Exponential' | 'Hantush'
}

interface PumpingUploadEditorProps {
  data: PumpingUploadData
  onChange: (data: PumpingUploadData) => void
}

const RFUNCS = ['Exponential', 'Hantush'] as const

const inputClass =
  'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-1.5 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

function parseCsv(text: string): PumpingRow[] {
  const lines = text.trim().split('\n')
  if (lines.length < 2) return []
  const header = lines[0].split(',').map((h) => h.trim().toLowerCase())
  const dateIdx = header.indexOf('date')
  const qIdx = header.findIndex((h) => h === 'q_m3d' || h === 'q')
  if (dateIdx === -1 || qIdx === -1) return []
  return lines.slice(1).flatMap((line) => {
    const cols = line.split(',')
    const date = cols[dateIdx]?.trim()
    const q = parseFloat(cols[qIdx]?.trim() ?? '')
    if (!date || isNaN(q)) return []
    return [{ date, Q_m3d: q }]
  })
}

export function PumpingUploadEditor({ data, onChange }: PumpingUploadEditorProps) {
  const fileRef = useRef<HTMLInputElement>(null)

  function update(patch: Partial<PumpingUploadData>) {
    onChange({ ...data, ...patch })
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target?.result as string
      const rows = parseCsv(text)
      update({ rows })
    }
    reader.readAsText(file)
  }

  return (
    <div className="space-y-3">
      <div
        className="border border-dashed border-white/20 rounded-lg p-4 text-center cursor-pointer hover:border-accent-cyan/40 transition-colors"
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          onChange={handleFile}
          className="hidden"
        />
        {data.rows.length > 0 ? (
          <p className="text-xs text-accent-cyan">{data.rows.length} rows loaded</p>
        ) : (
          <p className="text-xs text-text-muted">
            Drop CSV here or click — columns: <code>date, Q_m3d</code>
          </p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
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
    </div>
  )
}

export type { PumpingUploadData }
