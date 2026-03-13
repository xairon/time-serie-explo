interface StationMeta {
  id: string
  metadata: Record<string, unknown>
  cluster_id?: number | null
}

interface FilterPanelProps {
  domain: 'piezo' | 'hydro'
  stations: StationMeta[]
  filters: Record<string, string | number | null>
  onFiltersChange: (filters: Record<string, string | number | null>) => void
  colorBy: string
  onColorByChange: (attr: string) => void
}

const PIEZO_ATTRS = [
  { key: 'libelle_eh', label: 'Libellé' },
  { key: 'milieu_eh', label: 'Milieu' },
  { key: 'theme_eh', label: 'Thème' },
  { key: 'etat_eh', label: 'État' },
  { key: 'departement', label: 'Département' },
  { key: 'cluster_id', label: 'Cluster' },
]

const HYDRO_ATTRS = [
  { key: 'nom_cours_eau', label: "Cours d'eau" },
  { key: 'departement', label: 'Département' },
  { key: 'cluster_id', label: 'Cluster' },
]

const COLOR_BY_EXTRA = [{ key: 'altitude', label: 'Altitude' }]

function getDistinctValues(
  stations: StationMeta[],
  key: string,
): (string | number)[] {
  const values = new Set<string | number>()
  for (const s of stations) {
    const val = key === 'cluster_id' ? s.cluster_id : s.metadata[key]
    if (val !== undefined && val !== null && val !== '') {
      values.add(val as string | number)
    }
  }
  return Array.from(values).sort((a, b) =>
    String(a).localeCompare(String(b), undefined, { numeric: true }),
  )
}

export function FilterPanel({
  domain,
  stations,
  filters,
  onFiltersChange,
  colorBy,
  onColorByChange,
}: FilterPanelProps) {
  const attrs = domain === 'piezo' ? PIEZO_ATTRS : HYDRO_ATTRS
  const colorByOptions = [...attrs, ...COLOR_BY_EXTRA]

  function handleFilterChange(key: string, value: string) {
    if (value === '') {
      const next = { ...filters }
      delete next[key]
      onFiltersChange(next)
    } else {
      onFiltersChange({ ...filters, [key]: value })
    }
  }

  function handleReset() {
    onFiltersChange({})
  }

  const selectClass =
    'w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-accent-cyan/50 transition-colors'

  return (
    <div className="flex flex-col gap-4 w-60">
      <div className="flex items-center justify-between">
        <span className="text-text-primary text-sm font-medium">Filtres</span>
        <button
          onClick={handleReset}
          className="text-text-muted text-xs hover:text-text-secondary transition-colors"
        >
          Réinitialiser
        </button>
      </div>

      {attrs.map(({ key, label }) => {
        const values = getDistinctValues(stations, key)
        if (values.length === 0) return null
        return (
          <div key={key} className="flex flex-col gap-1">
            <label className="text-text-muted text-xs">{label}</label>
            <select
              className={selectClass}
              value={String(filters[key] ?? '')}
              onChange={(e) => handleFilterChange(key, e.target.value)}
            >
              <option value="">Tous</option>
              {values.map((v) => (
                <option key={String(v)} value={String(v)}>
                  {String(v)}
                </option>
              ))}
            </select>
          </div>
        )
      })}

      <div className="border-t border-white/5 pt-4 flex flex-col gap-1">
        <label className="text-text-muted text-xs">Colorer par</label>
        <select
          className={selectClass}
          value={colorBy}
          onChange={(e) => onColorByChange(e.target.value)}
        >
          {colorByOptions.map(({ key, label }) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
