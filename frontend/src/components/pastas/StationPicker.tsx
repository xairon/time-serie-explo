import { useState, useEffect } from 'react'
import { Search, MapPin } from 'lucide-react'
import { api } from '@/lib/api'

interface StationResult {
  code_bss: string
  nom_commune: string
  code_departement: string
  nom_departement: string
  nb_mesures_total: string
  premiere_mesure: string | null
  derniere_mesure: string | null
}

interface Props {
  codeBss: string
  onChange: (codeBss: string) => void
}

export function StationPicker({ codeBss, onChange }: Props) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<StationResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [showResults, setShowResults] = useState(false)

  useEffect(() => {
    if (query.length < 2) {
      setResults([])
      return
    }
    const timer = setTimeout(async () => {
      setIsSearching(true)
      try {
        const data = await api.db.searchStations({ q: query, limit: 15 })
        setResults(data.stations as unknown as StationResult[])
      } catch {
        setResults([])
      }
      setIsSearching(false)
      setShowResults(true)
    }, 300)
    return () => clearTimeout(timer)
  }, [query])

  const handleSelect = (station: StationResult) => {
    onChange(station.code_bss)
    setQuery(station.code_bss)
    setShowResults(false)
  }

  return (
    <div className="space-y-2">
      <label className="block text-xs text-text-muted">Station (code BSS)</label>
      <div className="relative">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            value={query}
            onChange={e => { setQuery(e.target.value); if (!e.target.value) onChange('') }}
            onFocus={() => results.length > 0 && setShowResults(true)}
            placeholder="Search by BSS code, commune, or département..."
            className="w-full bg-bg-primary border border-white/10 rounded-lg pl-9 pr-3 py-2 text-sm text-text-primary placeholder:text-text-muted"
          />
        </div>

        {showResults && results.length > 0 && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-bg-card border border-white/10 rounded-lg shadow-xl z-20 max-h-64 overflow-y-auto">
            {results.map(s => (
              <button
                key={s.code_bss}
                onClick={() => handleSelect(s)}
                className="w-full text-left px-3 py-2 hover:bg-bg-hover transition-colors first:rounded-t-lg last:rounded-b-lg"
              >
                <div className="flex items-center gap-2">
                  <MapPin className="w-3.5 h-3.5 text-accent-cyan shrink-0" />
                  <span className="text-sm font-mono text-text-primary">{s.code_bss}</span>
                </div>
                <div className="text-xs text-text-muted ml-5.5 mt-0.5">
                  {s.nom_commune} ({s.code_departement}) — {s.nb_mesures_total} mesures
                  {s.premiere_mesure && s.derniere_mesure && (
                    <span> · {s.premiere_mesure?.slice(0, 4)}–{s.derniere_mesure?.slice(0, 4)}</span>
                  )}
                </div>
              </button>
            ))}
          </div>
        )}

        {isSearching && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <div className="w-4 h-4 border-2 border-accent-cyan/30 border-t-accent-cyan rounded-full animate-spin" />
          </div>
        )}
      </div>

      {codeBss && (
        <div className="flex items-center gap-2 text-xs text-accent-cyan">
          <MapPin className="w-3 h-3" />
          <span>Selected: {codeBss}</span>
        </div>
      )}
    </div>
  )
}
