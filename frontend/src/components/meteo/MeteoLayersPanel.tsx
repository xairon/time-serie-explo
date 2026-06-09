export type LayerKey = 'bsn' | 'piezo' | 'rain' | 'hydro'

interface Props {
  visible: Record<LayerKey, boolean>
  onToggle: (key: LayerKey) => void
}

const LAYERS: { key: LayerKey; label: string }[] = [
  { key: 'bsn',   label: 'Bulletin de situation des nappes' },
  { key: 'piezo', label: 'Piézomètres' },
  { key: 'rain',  label: 'Pluviomètres' },
  { key: 'hydro', label: 'Hydrométrie' },
]

export function MeteoLayersPanel({ visible, onToggle }: Props) {
  return (
    <div className="absolute top-[100px] left-3 z-10 w-56 bg-white rounded-lg shadow-md border border-slate-200 p-3">
      <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-2">
        Couches affichées
      </h3>
      <div className="space-y-1.5">
        {LAYERS.map(({ key, label }) => (
          <label key={key} className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={visible[key]}
              onChange={() => onToggle(key)}
              className="w-3.5 h-3.5 accent-blue-600 rounded"
            />
            <span className="text-xs text-slate-700 group-hover:text-slate-900 transition-colors leading-tight">
              {label}
            </span>
          </label>
        ))}
      </div>
      <p className="mt-2.5 text-[10px] text-slate-400 leading-tight">
        Certains éléments ne sont visibles qu'en zoomant
      </p>
    </div>
  )
}
