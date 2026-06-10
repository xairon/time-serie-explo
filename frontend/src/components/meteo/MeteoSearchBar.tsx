// frontend/src/components/meteo/MeteoSearchBar.tsx
// Search combobox clone: BAN address geocoding + Junon stations (code/commune).
import { useState, useEffect, useRef } from 'react'
import type { StationGeoJSONFeature } from '@/lib/observatory-types'

export interface SearchTarget {
  lng: number
  lat: number
  zoom: number
  label: string
}

interface Suggestion extends SearchTarget {
  kind: 'adresse' | 'station'
}

interface Props {
  stations: StationGeoJSONFeature[]
  onSelect: (target: SearchTarget) => void
}

const BAN_URL = 'https://api-adresse.data.gouv.fr/search/'

async function searchBan(q: string): Promise<Suggestion[]> {
  try {
    const res = await fetch(`${BAN_URL}?q=${encodeURIComponent(q)}&limit=4`)
    if (!res.ok) return []
    const data = await res.json()
    return (data.features ?? []).map((f: { geometry: { coordinates: [number, number] }; properties: { label: string; context?: string } }) => ({
      kind: 'adresse' as const,
      lng: f.geometry.coordinates[0],
      lat: f.geometry.coordinates[1],
      zoom: 11,
      label: f.properties.context ? `${f.properties.label}, ${f.properties.context}` : f.properties.label,
    }))
  } catch {
    return [] // geocoding down → stations only
  }
}

function searchStations(stations: StationGeoJSONFeature[], q: string): Suggestion[] {
  const needle = q.trim().toLowerCase()
  if (needle.length < 2) return []
  return stations
    .filter(f =>
      f.properties.code.toLowerCase().startsWith(needle) ||
      (f.properties.commune ?? '').toLowerCase().includes(needle))
    .slice(0, 4)
    .map(f => ({
      kind: 'station' as const,
      lng: f.geometry.coordinates[0],
      lat: f.geometry.coordinates[1],
      zoom: 12,
      label: `${f.properties.code}${f.properties.commune ? ` — ${f.properties.commune}` : ''}`,
    }))
}

export function MeteoSearchBar({ stations, onSelect }: Props) {
  const [q, setQ] = useState('')
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [open, setOpen] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  useEffect(() => {
    clearTimeout(debounceRef.current)
    if (q.trim().length < 3) { setSuggestions([]); return }
    debounceRef.current = setTimeout(async () => {
      const [ban, sta] = [await searchBan(q), searchStations(stations, q)]
      setSuggestions([...sta, ...ban])
      setOpen(true)
    }, 300)
    return () => clearTimeout(debounceRef.current)
  }, [q, stations])

  const pick = (s: Suggestion) => {
    onSelect(s)
    setQ('')
    setSuggestions([])
    setOpen(false)
  }

  return (
    <div className="relative w-80">
      <div className="flex items-center gap-2 bg-white rounded-full shadow-md border border-slate-200 px-3.5 py-2">
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" className="text-slate-400 flex-shrink-0">
          <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
          <path d="M9.5 9.5 L13 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <input
          role="combobox"
          aria-expanded={open}
          aria-label="Rechercher une adresse ou une station"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && suggestions.length) pick(suggestions[0]) }}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onFocus={() => { if (suggestions.length) setOpen(true) }}
          placeholder="adresse, station, piézomètre, etc."
          className="flex-1 text-xs text-slate-700 placeholder-slate-400 bg-transparent outline-none"
        />
      </div>
      {open && suggestions.length > 0 && (
        <ul role="listbox" className="absolute top-full mt-1 left-0 right-0 bg-white rounded-lg shadow-lg border border-slate-200 py-1 max-h-64 overflow-y-auto">
          {suggestions.map((s, i) => (
            <li key={`${s.kind}-${i}`} role="option" aria-selected="false">
              <button
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(s)}
                className="w-full text-left px-3 py-1.5 text-xs text-slate-700 hover:bg-slate-50 flex items-center gap-2"
              >
                <span className={`text-[9px] uppercase font-semibold flex-shrink-0 ${s.kind === 'station' ? 'text-blue-600' : 'text-slate-400'}`}>
                  {s.kind === 'station' ? 'Station' : 'Adresse'}
                </span>
                <span className="truncate">{s.label}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
