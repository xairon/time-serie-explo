import { useState, useMemo, useRef, useEffect, useCallback, useDeferredValue } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Search, X } from 'lucide-react'
import type { StationGeoJSONFeature, WfsLayerId } from '@/lib/observatory-types'

export interface SearchAction {
  kind: 'station' | 'department' | 'region' | 'bassin' | 'her' | 'wfs'
  code: string
  label: string
  stationType?: 'piezo' | 'hydro'
  geometry?: any
  wfsLayerId?: WfsLayerId
}

interface SearchItem {
  kind: SearchAction['kind']
  badge: string
  badgeClass: string
  label: string
  detail: string
  action: SearchAction
}

interface Props {
  features?: StationGeoJSONFeature[]
  wfsData?: Record<string, any>
  onSearchAction: (action: SearchAction) => void
}

function norm(s: string): string {
  return s.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '')
}

function useGeoJSON<T = any>(path: string) {
  return useQuery<T>({
    queryKey: ['search-geo', path],
    queryFn: () => fetch(path).then(r => r.json()),
    staleTime: Infinity,
  })
}

const WFS_SEARCH: Record<string, { nameField: string; codeField: string; badge: string; badgeClass: string }> = {
  'cours-eau-1':      { nameField: 'NomEntiteHydrographique', codeField: 'CdEntiteHydrographique', badge: 'COURS D\'EAU', badgeClass: 'bg-blue-400/20 text-blue-300' },
  'cours-eau-2':      { nameField: 'NomEntiteHydrographique', codeField: 'CdEntiteHydrographique', badge: 'COURS D\'EAU', badgeClass: 'bg-blue-400/20 text-blue-300' },
  'plan-eau':         { nameField: 'NomEntiteHydrographique', codeField: 'CdEntiteHydrographique', badge: 'PLAN D\'EAU', badgeClass: 'bg-sky-400/20 text-sky-300' },
  'region-hydro':     { nameField: 'LbRegionHydro',           codeField: 'CdRegionHydro',          badge: 'RGN HYDRO', badgeClass: 'bg-blue-500/20 text-blue-400' },
  'secteur-hydro':    { nameField: 'LbSecteurHydro',          codeField: 'CdSecteurHydro',         badge: 'SECTEUR', badgeClass: 'bg-cyan-500/20 text-cyan-400' },
  'sous-secteur-hydro': { nameField: 'LbSousSecteurHydro',    codeField: 'CdSousSecteurHydro',     badge: 'SOUS-SECTEUR', badgeClass: 'bg-teal-500/20 text-teal-400' },
  'zone-hydro':       { nameField: 'LbZoneHydro',             codeField: 'CdZoneHydro',            badge: 'ZONE HYDRO', badgeClass: 'bg-emerald-500/20 text-emerald-400' },
  'masse-eau-riv':    { nameField: 'NomMasseDEau',            codeField: 'CdMasseDEau',            badge: 'MASSE D\'EAU', badgeClass: 'bg-purple-400/20 text-purple-300' },
}

const CATEGORY_ORDER = ['station', 'department', 'region', 'bassin', 'her', 'wfs'] as const
const CATEGORY_LABELS: Record<string, string> = {
  station: 'Stations', department: 'Départements', region: 'Régions',
  bassin: 'Bassins hydrographiques', her: 'Hydroécorégions', wfs: 'Hydrographie',
}

export function SearchBar({ features, wfsData, onSearchAction }: Props) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [highlightIndex, setHighlightIndex] = useState(-1)
  const inputRef = useRef<HTMLInputElement>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)

  const { data: deptsGeo } = useGeoJSON<{ features: Array<{ properties: { code: string; nom: string }; geometry: any }> }>('/geo/departments.geojson')
  const { data: regionsGeo } = useGeoJSON<{ features: Array<{ properties: { code: string; nom: string }; geometry: any }> }>('/geo/regions.geojson')
  const { data: bassinsGeo } = useGeoJSON<{ features: Array<{ properties: { CdBH: string; LbBH: string }; geometry: any }> }>('/geo/bassins.geojson')
  const { data: herGeo } = useGeoJSON<{ features: Array<{ properties: { code: number; nom: string; nom_her1?: string }; geometry: any }> }>('/geo/her.geojson')

  useEffect(() => {
    const handler = (e: MouseEvent) => { if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const deferredQuery = useDeferredValue(query)

  const results = useMemo<SearchItem[]>(() => {
    if (!deferredQuery || deferredQuery.length < 2) return []
    const q = norm(deferredQuery); const items: SearchItem[] = []; const MAX_PER_CAT = 3; const MAX_STATIONS = 5

    let stationCount = 0
    for (const f of (features ?? [])) {
      if (stationCount >= MAX_STATIONS) break
      const code = norm(f.properties.code); const commune = norm(f.properties.commune ?? ''); const dept = norm(f.properties.departement ?? '')
      if (code.includes(q) || commune.includes(q) || dept.includes(q)) {
        items.push({ kind: 'station', badge: f.properties.type === 'piezo' ? 'PIÉZO' : 'HYDRO', badgeClass: f.properties.type === 'piezo' ? 'bg-accent-cyan/20 text-accent-cyan' : 'bg-accent-indigo/20 text-accent-indigo', label: f.properties.commune || f.properties.code, detail: `${f.properties.departement || ''} - ${f.properties.code}`, action: { kind: 'station', code: f.properties.code, label: f.properties.commune || f.properties.code, stationType: f.properties.type, geometry: f.geometry } })
        stationCount++
      }
    }

    let deptCount = 0
    for (const f of (deptsGeo?.features ?? [])) {
      if (deptCount >= MAX_PER_CAT) break
      if (norm(f.properties.code).includes(q) || norm(f.properties.nom).includes(q)) {
        items.push({ kind: 'department', badge: `${f.properties.code}`, badgeClass: 'bg-orange-400/20 text-orange-300', label: f.properties.nom, detail: `Département ${f.properties.code}`, action: { kind: 'department', code: f.properties.code, label: f.properties.nom, geometry: f.geometry } })
        deptCount++
      }
    }

    let regionCount = 0
    for (const f of (regionsGeo?.features ?? [])) {
      if (regionCount >= MAX_PER_CAT) break
      if (norm(f.properties.nom).includes(q) || norm(f.properties.code).includes(q)) {
        items.push({ kind: 'region', badge: 'RÉGION', badgeClass: 'bg-indigo-400/20 text-indigo-300', label: f.properties.nom, detail: `Région ${f.properties.code}`, action: { kind: 'region', code: f.properties.code, label: f.properties.nom, geometry: f.geometry } })
        regionCount++
      }
    }

    let bassinCount = 0
    for (const f of (bassinsGeo?.features ?? [])) {
      if (bassinCount >= MAX_PER_CAT) break
      if (norm(f.properties.CdBH).includes(q) || norm(f.properties.LbBH).includes(q)) {
        items.push({ kind: 'bassin', badge: f.properties.CdBH, badgeClass: 'bg-blue-500/20 text-blue-300', label: f.properties.LbBH, detail: `Bassin ${f.properties.CdBH}`, action: { kind: 'bassin', code: f.properties.CdBH, label: f.properties.LbBH, geometry: f.geometry } })
        bassinCount++
      }
    }

    let herCount = 0
    for (const f of (herGeo?.features ?? [])) {
      if (herCount >= MAX_PER_CAT) break
      if (norm(f.properties.nom).includes(q) || norm(f.properties.nom_her1 ?? '').includes(q) || String(f.properties.code).includes(q)) {
        items.push({ kind: 'her', badge: 'HER', badgeClass: 'bg-emerald-400/20 text-emerald-300', label: f.properties.nom, detail: f.properties.nom_her1 ? `HER-1: ${f.properties.nom_her1}` : `Code ${f.properties.code}`, action: { kind: 'her', code: String(f.properties.code), label: f.properties.nom, geometry: f.geometry } })
        herCount++
      }
    }

    const wfsCounts: Record<string, number> = {}
    if (wfsData) {
      for (const [layerId, config] of Object.entries(WFS_SEARCH)) {
        const data = wfsData[layerId]; if (!data?.features) continue
        const catKey = config.badge; if (!wfsCounts[catKey]) wfsCounts[catKey] = 0; if (wfsCounts[catKey] >= MAX_PER_CAT) continue
        for (const f of data.features) {
          if (wfsCounts[catKey]! >= MAX_PER_CAT) break
          const name = norm(f.properties?.[config.nameField] ?? ''); const code = norm(f.properties?.[config.codeField] ?? '')
          if (name.includes(q) || code.includes(q)) {
            items.push({ kind: 'wfs', badge: config.badge, badgeClass: config.badgeClass, label: f.properties?.[config.nameField] ?? f.properties?.[config.codeField] ?? layerId, detail: f.properties?.[config.codeField] ?? '', action: { kind: 'wfs', code: f.properties?.[config.codeField] ?? '', label: f.properties?.[config.nameField] ?? '', geometry: f.geometry, wfsLayerId: layerId as WfsLayerId } })
            wfsCounts[catKey]!++
          }
        }
      }
    }

    return items.sort((a, b) => { const ai = CATEGORY_ORDER.indexOf(a.kind as typeof CATEGORY_ORDER[number]); const bi = CATEGORY_ORDER.indexOf(b.kind as typeof CATEGORY_ORDER[number]); return ai - bi })
  }, [deferredQuery, features, deptsGeo, regionsGeo, bassinsGeo, herGeo, wfsData])

  const selectItem = useCallback((item: SearchItem) => { onSearchAction(item.action); setQuery(''); setOpen(false); setHighlightIndex(-1) }, [onSearchAction])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!open || results.length === 0) return
    if (e.key === 'ArrowDown') { e.preventDefault(); setHighlightIndex(prev => Math.min(prev + 1, results.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setHighlightIndex(prev => Math.max(prev - 1, 0)) }
    else if (e.key === 'Enter' && highlightIndex >= 0) { e.preventDefault(); selectItem(results[highlightIndex]) }
    else if (e.key === 'Escape') { setOpen(false); setHighlightIndex(-1); inputRef.current?.blur() }
  }, [open, results, highlightIndex, selectItem])

  const grouped = useMemo(() => {
    const groups: { kind: string; label: string; items: SearchItem[] }[] = []; let currentKind = ''
    for (const item of results) {
      if (item.kind !== currentKind) { currentKind = item.kind; groups.push({ kind: item.kind, label: CATEGORY_LABELS[item.kind] ?? item.kind, items: [] }) }
      groups[groups.length - 1].items.push(item)
    }
    return groups
  }, [results])

  let flatIndex = -1

  return (
    <div ref={wrapperRef} className="absolute top-4 left-4 z-10 w-72 sm:w-96">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
        <input ref={inputRef} type="text" value={query} onChange={(e) => { setQuery(e.target.value); setOpen(true); setHighlightIndex(-1) }} onFocus={() => setOpen(true)} onKeyDown={handleKeyDown} placeholder="Station, département, cours d'eau, aquifère..." aria-label="Recherche universelle" aria-expanded={open && results.length > 0} aria-controls="search-results-list" role="combobox" aria-autocomplete="list" className="w-full pl-9 pr-8 py-2.5 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:border-accent-cyan/50" />
        {query && (
          <button onClick={() => { setQuery(''); setOpen(false); setHighlightIndex(-1) }} aria-label="Effacer la recherche" className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-bg-hover rounded">
            <X className="w-3 h-3 text-text-secondary" />
          </button>
        )}
      </div>
      {open && results.length > 0 && (
        <div id="search-results-list" role="listbox" className="mt-1 bg-bg-card/95 backdrop-blur-md border border-white/10 rounded-lg overflow-hidden shadow-xl max-h-80 overflow-y-auto">
          {grouped.map((group) => (
            <div key={group.kind}>
              <div className="px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-text-secondary bg-white/[0.03] border-b border-white/5">{group.label}</div>
              {group.items.map((item) => {
                flatIndex++; const idx = flatIndex
                return (
                  <button key={`${item.kind}-${item.action.code}-${idx}`} role="option" aria-selected={idx === highlightIndex} onClick={() => selectItem(item)} className={`w-full text-left px-3 py-2 transition-colors border-b border-white/5 last:border-0 ${idx === highlightIndex ? 'bg-bg-hover' : 'hover:bg-bg-hover'}`}>
                    <div className="flex items-center gap-2">
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium shrink-0 ${item.badgeClass}`}>{item.badge}</span>
                      <span className="text-sm text-text-primary truncate">{item.label}</span>
                    </div>
                    {item.detail && <p className="text-xs text-text-secondary mt-0.5 ml-14 truncate">{item.detail}</p>}
                  </button>
                )
              })}
            </div>
          ))}
        </div>
      )}
      {open && query.length >= 2 && results.length === 0 && (
        <div className="mt-1 bg-bg-card/95 backdrop-blur-md border border-white/10 rounded-lg px-4 py-3 text-sm text-text-secondary">Aucun résultat pour « {query} »</div>
      )}
    </div>
  )
}
