// frontend/src/components/meteo/MeteoTypePanel.tsx
// "Type" card clone: the original's 5 station types; rows without Junon data
// are kept (documents the gap) but disabled.
import type { StationType } from './layers/stations-layer'

interface Props {
  visible: Record<StationType, boolean>
  onToggle: (key: StationType) => void
}

const ROWS: { key: StationType | null; label: string }[] = [
  { key: 'piezo', label: 'Piézomètre' },
  { key: null,    label: 'Source' },
  { key: null,    label: 'Pluviomètre' },
  { key: 'hydro', label: 'Station de débit' },
  { key: null,    label: 'Avec modèle' },
]

export function MeteoTypePanel({ visible, onToggle }: Props) {
  return (
    <div className="bg-white rounded-lg shadow-md border border-slate-200 p-3 w-56">
      <h4 className="text-[11px] font-semibold text-slate-600 mb-1.5">Type</h4>
      <div className="space-y-1">
        {ROWS.map(({ key, label }) => {
          const enabled = key != null
          return (
            <label
              key={label}
              title={enabled ? undefined : 'Données bientôt disponibles'}
              className={`flex items-center gap-2 ${enabled ? 'cursor-pointer' : 'opacity-40 cursor-not-allowed'}`}
            >
              <input
                type="checkbox"
                aria-label={label}
                disabled={!enabled}
                checked={enabled ? visible[key] : false}
                onChange={() => { if (enabled) onToggle(key) }}
                className="w-3.5 h-3.5 accent-blue-600 rounded"
              />
              <span className="text-[11px] text-slate-700">{label}</span>
            </label>
          )
        })}
      </div>
      <p className="mt-2 text-[10px] text-slate-400 leading-tight">
        Stations visibles en zoomant sur la carte
      </p>
    </div>
  )
}
