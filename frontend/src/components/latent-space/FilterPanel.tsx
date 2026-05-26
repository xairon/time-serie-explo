import { useState, useMemo } from 'react'
import { Search, ChevronDown, ChevronRight } from 'lucide-react'

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
  onStationSelect?: (stationId: string) => void
  hideUnclassified: boolean
  onHideUnclassifiedChange: (v: boolean) => void
  onlyActive: boolean
  onOnlyActiveChange: (v: boolean) => void
}

const PIEZO_PRIMARY = [
  { key: 'libelle_eh', label: 'Aquifère' },
  { key: 'departement', label: 'Département' },
]

const PIEZO_ADVANCED = [
  { key: 'milieu_eh', label: 'Milieu' },
  { key: 'theme_eh', label: 'Thème' },
  { key: 'etat_eh', label: 'État' },
  { key: 'nature_eh', label: 'Nature' },
]

const HYDRO_PRIMARY = [
  { key: 'nom_cours_eau', label: "Cours d'eau" },
  { key: 'departement', label: 'Département' },
]

const COLOR_OPTIONS_PIEZO = [
  { key: 'cluster_id', label: 'Cluster' },
  { key: 'libelle_eh', label: 'Aquifère' },
  { key: 'departement', label: 'Département' },
  { key: 'altitude', label: 'Altitude' },
]

const COLOR_OPTIONS_HYDRO = [
  { key: 'cluster_id', label: 'Cluster' },
  { key: 'nom_cours_eau', label: "Cours d'eau" },
  { key: 'departement', label: 'Département' },
  { key: 'altitude', label: 'Altitude' },
]

function getDistinctValues(stations: StationMeta[], key: string): (string | number)[] {
  const values = new Set<string | number>()
  for (const s of stations) {
    const val = key === 'cluster_id' ? s.cluster_id : s.metadata[key]
    if (val !== undefined && val !== null && val !== '') values.add(val as string | number)
  }
  return Array.from(values).sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }))
}

export function FilterPanel({
  domain, stations, filters, onFiltersChange, colorBy, onColorByChange,
  onStationSelect, hideUnclassified, onHideUnclassifiedChange, onlyActive, onOnlyActiveChange,
}: FilterPanelProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)

  const primaryAttrs = domain === 'piezo' ? PIEZO_PRIMARY : HYDRO_PRIMARY
  const advancedAttrs = domain === 'piezo' ? PIEZO_ADVANCED : []
  const colorOptions = domain === 'piezo' ? COLOR_OPTIONS_PIEZO : COLOR_OPTIONS_HYDRO

  const searchResults = useMemo(() => {
    if (!searchQuery || searchQuery.length < 2) return []
    const q = searchQuery.toUpperCase()
    return stations.filter((s) => s.id.toUpperCase().includes(q)).slice(0, 8)
  }, [searchQuery, stations])

  function handleFilterChange(key: string, value: string) {
    if (value === '') {
      const next = { ...filters }; delete next[key]; onFiltersChange(next)
    } else {
      onFiltersChange({ ...filters, [key]: value })
    }
  }

  const selectClass = 'w-full bg-bg-input text-text-primary border border-white/10 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-accent-cyan/50 transition-colors'

  return (
    <div className="flex flex-col gap-3 w-48">
      <div className="flex items-center justify-between">
        <span className="text-text-primary text-xs font-medium">Filtres</span>
        <button onClick={() => { onFiltersChange({}); setSearchQuery('') }}
          className="text-text-muted text-[10px] hover:text-text-secondary transition-colors">Réinitialiser</button>
      </div>

      {/* Station search */}
      <div className="flex flex-col gap-1 relative">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-text-muted pointer-events-none" />
          <input type="text" className={`${selectClass} pl-7`} placeholder="Station code..."
            value={searchQuery}
            onChange={(e) => { setSearchQuery(e.target.value); setShowSuggestions(true) }}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
          />
        </div>
        {showSuggestions && searchResults.length > 0 && (
          <div className="absolute top-full left-0 right-0 z-20 mt-0.5 bg-bg-card border border-white/10 rounded shadow-xl max-h-40 overflow-y-auto">
            {searchResults.map((s) => (
              <button key={s.id}
                className="w-full text-left px-2 py-1 text-[10px] text-text-primary hover:bg-bg-hover transition-colors truncate"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => { setSearchQuery(s.id); setShowSuggestions(false); onStationSelect?.(s.id) }}
              >{s.id}</button>
            ))}
          </div>
        )}
      </div>

      {/* Toggles */}
      <div className="flex flex-col gap-1.5">
        <label className="flex items-center gap-1.5 text-[10px] text-text-muted cursor-pointer">
          <input type="checkbox" checked={onlyActive} onChange={(e) => onOnlyActiveChange(e.target.checked)} className="accent-accent-cyan w-3 h-3" />
          Active stations only
        </label>
        {domain === 'piezo' && (
          <label className="flex items-center gap-1.5 text-[10px] text-text-muted cursor-pointer">
            <input type="checkbox" checked={hideUnclassified} onChange={(e) => onHideUnclassifiedChange(e.target.checked)} className="accent-accent-cyan w-3 h-3" />
            Hide unclassified
          </label>
        )}
      </div>

      <div className="border-t border-white/5" />

      {/* Color by */}
      <div className="flex flex-col gap-1">
        <label className="text-text-muted text-[10px]">Color by</label>
        <select className={selectClass} value={colorBy} onChange={(e) => onColorByChange(e.target.value)}>
          {colorOptions.map(({ key, label }) => <option key={key} value={key}>{label}</option>)}
        </select>
      </div>

      {/* Cluster filter */}
      {(() => {
        const clusterValues = getDistinctValues(stations, 'cluster_id')
        return clusterValues.length > 0 ? (
          <div className="flex flex-col gap-1">
            <label className="text-text-muted text-[10px]">Cluster</label>
            <select className={selectClass} value={String(filters.cluster_id ?? '')}
              onChange={(e) => handleFilterChange('cluster_id', e.target.value)}>
              <option value="">All</option>
              {clusterValues.map((v) => <option key={String(v)} value={String(v)}>{String(v)}</option>)}
            </select>
          </div>
        ) : null
      })()}

      {/* Primary filters */}
      {primaryAttrs.map(({ key, label }) => {
        const values = getDistinctValues(stations, key)
        if (values.length === 0) return null
        return (
          <div key={key} className="flex flex-col gap-1">
            <label className="text-text-muted text-[10px]">{label}</label>
            <select className={selectClass} value={String(filters[key] ?? '')}
              onChange={(e) => handleFilterChange(key, e.target.value)}>
              <option value="">All</option>
              {values.map((v) => <option key={String(v)} value={String(v)}>{String(v)}</option>)}
            </select>
          </div>
        )
      })}

      {/* Advanced filters (collapsible) */}
      {advancedAttrs.length > 0 && (
        <div>
          <button onClick={() => setAdvancedOpen(!advancedOpen)}
            className="flex items-center gap-1 text-[10px] text-text-muted hover:text-text-secondary transition-colors">
            {advancedOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            Advanced filters
          </button>
          {advancedOpen && (
            <div className="flex flex-col gap-2 mt-2">
              {advancedAttrs.map(({ key, label }) => {
                const values = getDistinctValues(stations, key)
                if (values.length === 0) return null
                return (
                  <div key={key} className="flex flex-col gap-1">
                    <label className="text-text-muted text-[10px]">{label}</label>
                    <select className={selectClass} value={String(filters[key] ?? '')}
                      onChange={(e) => handleFilterChange(key, e.target.value)}>
                      <option value="">All</option>
                      {values.map((v) => <option key={String(v)} value={String(v)}>{String(v)}</option>)}
                    </select>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
