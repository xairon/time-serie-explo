import { useState, useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/lib/api'

export function ImportDBForm() {
  const qc = useQueryClient()

  // Step 1: Schema & Table selection
  const [schema, setSchema] = useState('gold')
  const [selectedItem, setSelectedItem] = useState('')

  // Step 2: Columns
  const [dateColumn, setDateColumn] = useState('')
  const [stationColumn, setStationColumn] = useState('')
  const [selectedColumns, setSelectedColumns] = useState<string[]>([])

  // Step 3: Filters
  const [stationFilterMode, setStationFilterMode] = useState<'all' | 'select'>('all')
  const [stationInput, setStationInput] = useState('')
  const [selectedStations, setSelectedStations] = useState<string[]>([])
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')

  // Step 4: Dataset name
  const [datasetName, setDatasetName] = useState('')

  // Fetch schemas
  const { data: schemas } = useQuery({
    queryKey: ['db-schemas'],
    queryFn: () => api.db.schemas(),
    staleTime: 60_000,
  })

  // Fetch tables for selected schema
  const { data: tablesData } = useQuery({
    queryKey: ['db-tables', schema],
    queryFn: () => api.db.tables(schema),
    enabled: !!schema,
    staleTime: 60_000,
  })

  // Combine tables and views for display
  const allItems = [
    ...(tablesData?.views?.map((v) => `[VIEW] ${v}`) ?? []),
    ...(tablesData?.tables?.map((t) => `[TABLE] ${t}`) ?? []),
  ]

  // Extract actual table name from selected item
  const tableName = selectedItem.replace('[TABLE] ', '').replace('[VIEW] ', '')

  // Fetch columns when table selected
  const { data: columnsData } = useQuery({
    queryKey: ['db-columns', schema, tableName],
    queryFn: () => api.db.columns(tableName, schema),
    enabled: !!tableName,
    staleTime: 60_000,
  })

  // Fetch date range when date column selected
  const { data: dateRange } = useQuery({
    queryKey: ['db-daterange', schema, tableName, dateColumn],
    queryFn: () => api.db.dateRange(tableName, dateColumn, schema),
    enabled: !!tableName && !!dateColumn,
    staleTime: 60_000,
  })

  // Fetch station values when station column selected
  const { data: stationValues, isLoading: loadingStations } = useQuery({
    queryKey: ['db-stations', schema, tableName, stationColumn],
    queryFn: () => api.db.distinct(tableName, stationColumn, schema),
    enabled: !!tableName && !!stationColumn,
    staleTime: 60_000,
  })

  // Auto-detect date column when columns load
  useEffect(() => {
    if (columnsData?.date_columns?.length) {
      setDateColumn(columnsData.date_columns[0])
    }
  }, [columnsData])

  // Auto-detect station column (text/varchar columns)
  useEffect(() => {
    if (columnsData?.columns) {
      const textTypes = ['varchar', 'text', 'char', 'character']
      const textCols = columnsData.columns.filter(
        (c) =>
          textTypes.some((t) => c.type.toLowerCase().includes(t)) &&
          c.name !== dateColumn,
      )
      if (textCols.length > 0) {
        setStationColumn(textCols[0].name)
      }

      // Auto-select numeric columns
      const numericTypes = ['int', 'float', 'numeric', 'decimal', 'double', 'real', 'bigint', 'smallint']
      const numCols = columnsData.columns
        .filter((c) => numericTypes.some((t) => c.type.toLowerCase().includes(t)))
        .map((c) => c.name)
      setSelectedColumns(numCols.slice(0, 10))
    }
  }, [columnsData, dateColumn])

  // Set default date range
  useEffect(() => {
    if (dateRange?.min && !dateFrom) setDateFrom(dateRange.min.split(' ')[0])
    if (dateRange?.max && !dateTo) setDateTo(dateRange.max.split(' ')[0])
  }, [dateRange, dateFrom, dateTo])

  // Auto-name dataset
  useEffect(() => {
    if (tableName && !datasetName) {
      setDatasetName(`db_${tableName}`)
    }
  }, [tableName, datasetName])

  // Parse pasted station codes
  const parsedStations = stationInput
    .replace(/,/g, '\n')
    .split('\n')
    .map((s) => s.trim())
    .filter(Boolean)
  const validStations = stationValues
    ? parsedStations.filter((s) => stationValues.includes(s))
    : []
  const invalidStations = stationValues
    ? parsedStations.filter((s) => !stationValues.includes(s))
    : []

  // Filtered station list for search
  const [stationSearch, setStationSearch] = useState('')
  const filteredStations =
    stationSearch.length >= 2
      ? (stationValues ?? []).filter((s) =>
          s.toLowerCase().includes(stationSearch.toLowerCase()),
        )
      : (stationValues ?? []).slice(0, 100)

  // Effective stations for the query
  const effectiveStations =
    stationFilterMode === 'all'
      ? []
      : stationInput.trim()
        ? validStations
        : selectedStations

  // Numeric columns for display
  const numericTypes = ['int', 'float', 'numeric', 'decimal', 'double', 'real', 'bigint', 'smallint']
  const numericCols =
    columnsData?.columns
      .filter(
        (c) =>
          numericTypes.some((t) => c.type.toLowerCase().includes(t)) &&
          c.name !== dateColumn &&
          c.name !== stationColumn,
      )
      .map((c) => c.name) ?? []

  // All columns for fallback
  const allColumnNames =
    columnsData?.columns
      .filter((c) => c.name !== dateColumn && c.name !== stationColumn)
      .map((c) => c.name) ?? []

  const toggleColumn = (col: string) => {
    setSelectedColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col],
    )
  }

  const importMutation = useMutation({
    mutationFn: () => {
      const filters: Record<string, string[]> = {}
      if (stationColumn && effectiveStations.length > 0) {
        filters[stationColumn] = effectiveStations
      }

      const queryColumns = [
        ...(dateColumn ? [dateColumn] : []),
        ...(stationColumn ? [stationColumn] : []),
        ...selectedColumns,
      ]

      return api.datasets.importDB({
        table_name: tableName,
        schema_name: schema,
        columns: queryColumns,
        date_column: dateColumn || undefined,
        start_date: dateFrom || undefined,
        end_date: dateTo || undefined,
        filters: Object.keys(filters).length > 0 ? filters : undefined,
        dataset_name: datasetName,
      })
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['datasets'] })
    },
  })

  const loadDisabled =
    !selectedColumns.length ||
    (stationColumn &&
      stationFilterMode === 'select' &&
      effectiveStations.length === 0)

  return (
    <div className="space-y-5">
      <h3 className="text-sm font-semibold text-text-primary">
        Import depuis PostgreSQL
      </h3>

      {/* Step 1: Schema & Table */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-secondary mb-1">Schéma</label>
          <select
            value={schema}
            onChange={(e) => {
              setSchema(e.target.value)
              setSelectedItem('')
            }}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            {schemas?.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-text-secondary mb-1">Table / Vue</label>
          <select
            value={selectedItem}
            onChange={(e) => {
              setSelectedItem(e.target.value)
              setDateColumn('')
              setStationColumn('')
              setSelectedColumns([])
              setDateFrom('')
              setDateTo('')
              setDatasetName('')
            }}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">Sélectionner...</option>
            {allItems.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Table info */}
      {columnsData && (
        <div className="flex gap-4 text-xs text-text-secondary">
          <span>
            <strong className="text-text-primary">{columnsData.row_count.toLocaleString()}</strong>{' '}
            lignes
          </span>
          <span>
            <strong className="text-text-primary">{columnsData.columns.length}</strong>{' '}
            colonnes
          </span>
        </div>
      )}

      {/* Step 2: Date & Station columns */}
      {columnsData && (
        <>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-text-secondary mb-1">
                Colonne date
              </label>
              <select
                value={dateColumn}
                onChange={(e) => {
                  setDateColumn(e.target.value)
                  setDateFrom('')
                  setDateTo('')
                }}
                className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              >
                <option value="">Aucune</option>
                {columnsData.columns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}{' '}
                    {columnsData.date_columns.includes(c.name) ? '(date)' : ''}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-text-secondary mb-1">
                Colonne station (optionnel)
              </label>
              <select
                value={stationColumn}
                onChange={(e) => {
                  setStationColumn(e.target.value)
                  setSelectedStations([])
                  setStationInput('')
                  setStationSearch('')
                }}
                className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              >
                <option value="">Aucune</option>
                {columnsData.columns
                  .filter((c) => c.name !== dateColumn)
                  .map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name}
                    </option>
                  ))}
              </select>
            </div>
          </div>

          {/* Station filter */}
          {stationColumn && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs text-text-secondary">
                {loadingStations ? (
                  <span>Chargement des stations...</span>
                ) : (
                  <span>
                    <strong className="text-text-primary">
                      {stationValues?.length?.toLocaleString() ?? '?'}
                    </strong>{' '}
                    stations trouvées
                  </span>
                )}
              </div>

              <div className="flex gap-3 text-xs">
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="radio"
                    name="stationFilter"
                    checked={stationFilterMode === 'all'}
                    onChange={() => setStationFilterMode('all')}
                    className="accent-accent-cyan"
                  />
                  Toutes les stations
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="radio"
                    name="stationFilter"
                    checked={stationFilterMode === 'select'}
                    onChange={() => setStationFilterMode('select')}
                    className="accent-accent-cyan"
                  />
                  Sélection
                </label>
              </div>

              {stationFilterMode === 'select' && (
                <div className="space-y-2">
                  {/* Paste codes */}
                  <textarea
                    value={stationInput}
                    onChange={(e) => setStationInput(e.target.value)}
                    rows={2}
                    className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-xs resize-none"
                    placeholder="Coller des codes (un par ligne ou séparés par des virgules)..."
                  />

                  {stationInput.trim() && parsedStations.length > 0 && (
                    <div className="text-xs space-y-1">
                      {validStations.length > 0 && (
                        <p className="text-accent-green">
                          {validStations.length} code(s) valide(s)
                        </p>
                      )}
                      {invalidStations.length > 0 && (
                        <p className="text-accent-amber">
                          {invalidStations.length} code(s) non trouvé(s) :{' '}
                          {invalidStations.slice(0, 3).join(', ')}
                          {invalidStations.length > 3 && '...'}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Or search */}
                  {!stationInput.trim() && (
                    <>
                      <input
                        type="text"
                        value={stationSearch}
                        onChange={(e) => setStationSearch(e.target.value)}
                        className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-xs"
                        placeholder="Rechercher une station..."
                      />
                      {filteredStations.length > 0 && (
                        <div className="max-h-32 overflow-y-auto border border-white/10 rounded-lg">
                          {filteredStations.map((s) => (
                            <label
                              key={s}
                              className="flex items-center gap-2 px-3 py-1 text-xs hover:bg-bg-hover cursor-pointer"
                            >
                              <input
                                type="checkbox"
                                checked={selectedStations.includes(s)}
                                onChange={() =>
                                  setSelectedStations((prev) =>
                                    prev.includes(s)
                                      ? prev.filter((x) => x !== s)
                                      : [...prev, s],
                                  )
                                }
                                className="accent-accent-cyan"
                              />
                              <span className="text-text-primary">{s}</span>
                            </label>
                          ))}
                        </div>
                      )}
                      {stationSearch.length >= 2 && filteredStations.length === 0 && (
                        <p className="text-xs text-text-secondary">
                          Aucune station trouvée pour « {stationSearch} »
                        </p>
                      )}
                    </>
                  )}

                  {effectiveStations.length > 0 && (
                    <p className="text-xs text-accent-cyan">
                      {effectiveStations.length} station(s) sélectionnée(s)
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Data columns */}
          <div>
            <label className="block text-xs text-text-secondary mb-1">
              Colonnes de données
            </label>
            <div className="flex flex-wrap gap-1.5">
              {(numericCols.length > 0 ? numericCols : allColumnNames).map((col) => (
                <button
                  key={col}
                  type="button"
                  onClick={() => toggleColumn(col)}
                  className={`text-xs px-2 py-0.5 rounded border transition-colors ${
                    selectedColumns.includes(col)
                      ? 'bg-accent-cyan/10 border-accent-cyan text-accent-cyan'
                      : 'bg-bg-hover border-white/10 text-text-secondary hover:text-text-primary'
                  }`}
                >
                  {col}
                </button>
              ))}
            </div>
          </div>

          {/* Date range filter */}
          {dateColumn && dateRange && (
            <div>
              <label className="block text-xs text-text-secondary mb-1">
                Plage de dates
              </label>
              <div className="grid grid-cols-2 gap-3">
                <input
                  type="date"
                  value={dateFrom}
                  onChange={(e) => setDateFrom(e.target.value)}
                  className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                />
                <input
                  type="date"
                  value={dateTo}
                  onChange={(e) => setDateTo(e.target.value)}
                  className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                />
              </div>
            </div>
          )}

          {/* Large table warning */}
          {columnsData.row_count > 1_000_000 && stationColumn && stationFilterMode === 'all' && (
            <p className="text-xs text-accent-amber">
              Table volumineuse ({columnsData.row_count.toLocaleString()} lignes) —
              sélectionnez des stations pour réduire le temps de chargement.
            </p>
          )}

          {/* Dataset name */}
          <div>
            <label className="block text-xs text-text-secondary mb-1">
              Nom du dataset
            </label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              placeholder="Mon dataset"
            />
          </div>

          {/* Import button */}
          <button
            type="button"
            disabled={!!loadDisabled || importMutation.isPending || !datasetName}
            onClick={() => importMutation.mutate()}
            className="w-full bg-accent-cyan text-white px-4 py-2 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium"
          >
            {importMutation.isPending ? 'Importation...' : 'Importer depuis la BDD'}
          </button>

          {importMutation.isError && (
            <p className="text-xs text-accent-red">
              Erreur : {(importMutation.error as Error).message}
            </p>
          )}
          {importMutation.isSuccess && (
            <p className="text-xs text-accent-green">Dataset importé avec succès.</p>
          )}
        </>
      )}
    </div>
  )
}
