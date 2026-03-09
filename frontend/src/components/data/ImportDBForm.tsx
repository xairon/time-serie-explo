import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { StationInfo } from '@/lib/types'
import { Search, MapPin, TrendingUp, AlertTriangle, Calendar, ChevronDown, ChevronUp, Plus, X } from 'lucide-react'

export function ImportDBForm() {
  const qc = useQueryClient()

  // Search state
  const [searchTerm, setSearchTerm] = useState('')
  const [deptFilter, setDeptFilter] = useState('')
  const [tendanceFilter, setTendanceFilter] = useState('')
  const [alerteFilter, setAlerteFilter] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  // Selected stations
  const [selectedStations, setSelectedStations] = useState<StationInfo[]>([])

  // Import config
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [datasetName, setDatasetName] = useState('')

  // Fetch filter options
  const { data: filters } = useQuery({
    queryKey: ['station-filters'],
    queryFn: () => api.db.stationFilters(),
    staleTime: 300_000,
  })

  // Search stations
  const { data: searchResults, isLoading: searching } = useQuery({
    queryKey: ['station-search', searchTerm, deptFilter, tendanceFilter, alerteFilter],
    queryFn: () =>
      api.db.searchStations({
        q: searchTerm || undefined,
        departement: deptFilter || undefined,
        tendance: tendanceFilter || undefined,
        alerte: alerteFilter || undefined,
        limit: 50,
      }),
    enabled: searchTerm.length >= 2 || !!deptFilter || !!tendanceFilter || !!alerteFilter,
    staleTime: 30_000,
  })

  // Paste station codes
  const [pasteMode, setPasteMode] = useState(false)
  const [pastedCodes, setPastedCodes] = useState('')

  const toggleStation = (station: StationInfo) => {
    setSelectedStations((prev) => {
      const exists = prev.some((s) => s.code_bss === station.code_bss)
      if (exists) return prev.filter((s) => s.code_bss !== station.code_bss)
      return [...prev, station]
    })
  }

  const removeStation = (code: string) => {
    setSelectedStations((prev) => prev.filter((s) => s.code_bss !== code))
  }

  const isSelected = (code: string) => selectedStations.some((s) => s.code_bss === code)

  // Auto-generate dataset name from selected stations
  const autoName = () => {
    if (selectedStations.length === 1) return selectedStations[0].code_bss.replace('/', '_')
    if (selectedStations.length > 1) return `piezo_${selectedStations.length}stations`
    return 'hubeau_import'
  }

  // Import mutation
  const importMutation = useMutation({
    mutationFn: () => {
      const codes = selectedStations.map((s) => s.code_bss)
      return api.datasets.importDB({
        table_name: 'hubeau_daily_chroniques',
        schema_name: 'gold',
        columns: [
          'code_bss',
          'date',
          'niveau_nappe_eau',
          'profondeur_nappe',
          'temperature_2m',
          'total_precipitation',
          'potential_evaporation',
        ],
        date_column: 'date',
        start_date: dateFrom || undefined,
        end_date: dateTo || undefined,
        filters: { code_bss: codes },
        dataset_name: datasetName || autoName(),
      })
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['datasets'] })
    },
  })

  // Handle paste
  const handlePaste = () => {
    const codes = pastedCodes
      .replace(/,/g, '\n')
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean)
    if (codes.length > 0) {
      // Search each code to get metadata
      codes.forEach((code) => {
        if (!isSelected(code)) {
          // Add as minimal station info
          setSelectedStations((prev) => [
            ...prev,
            {
              code_bss: code,
              nom_commune: null,
              code_departement: null,
              nom_departement: null,
              codes_bdlisa: null,
              altitude_station: null,
              latitude: null,
              longitude: null,
              premiere_mesure: null,
              derniere_mesure: null,
              nb_mesures_total: null,
              niveau_moyen_global: null,
              amplitude_totale: null,
              tendance_classification: null,
              niveau_alerte: null,
              classification_derniere_annee: null,
              qualite_tendance: null,
            },
          ])
        }
      })
      setPastedCodes('')
      setPasteMode(false)
    }
  }

  const alerteColor = (alerte: string | null) => {
    if (!alerte) return 'text-text-secondary'
    if (alerte === 'NORMAL') return 'text-accent-green'
    if (alerte === 'VIGILANCE') return 'text-accent-amber'
    return 'text-accent-red'
  }

  const classifLabel = (c: string | null) => {
    if (!c) return null
    const map: Record<string, string> = {
      TRES_HAUT: 'Tres haut',
      HAUT: 'Haut',
      NORMAL: 'Normal',
      BAS: 'Bas',
      TRES_BAS: 'Tres bas',
    }
    return map[c] || c
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-text-primary">
        Import piézométrique
      </h3>
      <p className="text-xs text-text-secondary">
        Recherchez et sélectionnez des stations depuis la base BRGM (hubeau_daily_chroniques)
      </p>

      {/* Search bar */}
      <div className="relative">
        <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-text-secondary" />
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg pl-9 pr-3 py-2 text-sm"
          placeholder="Code BSS, commune, département..."
        />
      </div>

      {/* Filter toggles */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-1 text-xs text-text-secondary hover:text-text-primary"
        >
          {showFilters ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          Filtres
        </button>
        <button
          type="button"
          onClick={() => setPasteMode(!pasteMode)}
          className="flex items-center gap-1 text-xs text-accent-cyan hover:text-accent-cyan/80"
        >
          <Plus className="h-3 w-3" />
          Coller des codes
        </button>
      </div>

      {/* Filters */}
      {showFilters && filters && (
        <div className="grid grid-cols-3 gap-2">
          <select
            value={deptFilter}
            onChange={(e) => setDeptFilter(e.target.value)}
            className="bg-bg-input text-text-primary border border-white/10 rounded-lg px-2 py-1.5 text-xs"
          >
            <option value="">Département</option>
            {filters.departements.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
          <select
            value={tendanceFilter}
            onChange={(e) => setTendanceFilter(e.target.value)}
            className="bg-bg-input text-text-primary border border-white/10 rounded-lg px-2 py-1.5 text-xs"
          >
            <option value="">Tendance</option>
            {filters.tendances.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <select
            value={alerteFilter}
            onChange={(e) => setAlerteFilter(e.target.value)}
            className="bg-bg-input text-text-primary border border-white/10 rounded-lg px-2 py-1.5 text-xs"
          >
            <option value="">Alerte</option>
            {filters.alertes.map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </div>
      )}

      {/* Paste mode */}
      {pasteMode && (
        <div className="space-y-2">
          <textarea
            value={pastedCodes}
            onChange={(e) => setPastedCodes(e.target.value)}
            rows={3}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-xs resize-none"
            placeholder="Coller des codes BSS (un par ligne ou séparés par des virgules)..."
          />
          <button
            type="button"
            onClick={handlePaste}
            disabled={!pastedCodes.trim()}
            className="text-xs bg-accent-cyan/10 text-accent-cyan px-3 py-1 rounded-lg hover:bg-accent-cyan/20 disabled:opacity-50"
          >
            Ajouter
          </button>
        </div>
      )}

      {/* Search results */}
      {(searchResults?.stations?.length ?? 0) > 0 && (
        <div className="max-h-64 overflow-y-auto border border-white/10 rounded-lg divide-y divide-white/5">
          {searchResults!.stations.map((station) => (
            <button
              key={station.code_bss}
              type="button"
              onClick={() => toggleStation(station)}
              className={`w-full text-left px-3 py-2 hover:bg-bg-hover transition-colors ${
                isSelected(station.code_bss) ? 'bg-accent-cyan/5 border-l-2 border-l-accent-cyan' : ''
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="text-xs font-mono text-text-primary">
                    {station.code_bss}
                  </div>
                  <div className="text-[10px] text-text-secondary flex items-center gap-1.5 mt-0.5 flex-wrap">
                    {station.nom_commune && (
                      <span className="flex items-center gap-0.5">
                        <MapPin className="h-2.5 w-2.5" />
                        {station.nom_commune} ({station.code_departement})
                      </span>
                    )}
                    {station.codes_bdlisa && (
                      <span className="text-accent-indigo" title="Code BDLISA (entité hydrogéologique)">
                        {station.codes_bdlisa}
                      </span>
                    )}
                  </div>
                  <div className="text-[10px] text-text-secondary flex items-center gap-2 mt-0.5">
                    {station.premiere_mesure && station.derniere_mesure && (
                      <span className="flex items-center gap-0.5">
                        <Calendar className="h-2.5 w-2.5" />
                        {station.premiere_mesure.slice(0, 4)}–{station.derniere_mesure.slice(0, 4)}
                      </span>
                    )}
                    {station.nb_mesures_total != null && (
                      <span>{Math.round(station.nb_mesures_total).toLocaleString()} mes.</span>
                    )}
                    {station.amplitude_totale != null && (
                      <span className="flex items-center gap-0.5">
                        <TrendingUp className="h-2.5 w-2.5" />
                        {station.amplitude_totale.toFixed(1)}m
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-0.5 flex-shrink-0">
                  {station.niveau_alerte && (
                    <span className={`text-[10px] ${alerteColor(station.niveau_alerte)}`}>
                      {station.niveau_alerte === 'NORMAL' ? '' : station.niveau_alerte}
                    </span>
                  )}
                  {station.classification_derniere_annee && (
                    <span className="text-[10px] text-text-secondary">
                      {classifLabel(station.classification_derniere_annee)}
                    </span>
                  )}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {searching && (
        <div className="text-xs text-text-secondary text-center py-4">Recherche...</div>
      )}

      {searchTerm.length >= 2 && !searching && searchResults?.stations?.length === 0 && (
        <div className="text-xs text-text-secondary text-center py-4">
          Aucune station trouvée pour « {searchTerm} »
        </div>
      )}

      {/* Selected stations */}
      {selectedStations.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-text-secondary">
            {selectedStations.length} station(s) sélectionnée(s)
          </div>
          <div className="flex flex-wrap gap-1.5">
            {selectedStations.map((s) => (
              <span
                key={s.code_bss}
                className="inline-flex items-center gap-1 bg-accent-cyan/10 text-accent-cyan text-xs px-2 py-0.5 rounded-lg"
              >
                <span className="font-mono">{s.code_bss}</span>
                <button
                  type="button"
                  onClick={() => removeStation(s.code_bss)}
                  className="hover:text-accent-red"
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>

          {/* Date range */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-text-secondary mb-1">Date début</label>
              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-text-secondary mb-1">Date fin</label>
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              />
            </div>
          </div>

          {/* Dataset name */}
          <div>
            <label className="block text-xs text-text-secondary mb-1">Nom du dataset</label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder={autoName()}
              className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
            />
          </div>

          {/* Import button */}
          <button
            type="button"
            disabled={importMutation.isPending}
            onClick={() => importMutation.mutate()}
            className="w-full bg-accent-cyan text-white px-4 py-2 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium"
          >
            {importMutation.isPending
              ? 'Importation...'
              : `Importer ${selectedStations.length} station(s)`}
          </button>

          {importMutation.isError && (
            <p className="text-xs text-accent-red">
              Erreur : {(importMutation.error as Error).message}
            </p>
          )}
          {importMutation.isSuccess && (
            <p className="text-xs text-accent-green">Dataset importé avec succès.</p>
          )}
        </div>
      )}
    </div>
  )
}
