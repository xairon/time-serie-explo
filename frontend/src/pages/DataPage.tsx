import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { useDatasets, useDatasetPreview, useDatasetProfile } from '@/hooks/useDatasets'
import type { DatasetSummary } from '@/lib/types'
import { ImportDBForm } from '@/components/data/ImportDBForm'
import { ImportCSVForm } from '@/components/data/ImportCSVForm'
import { DatasetCard } from '@/components/cards/DatasetCard'
import { DataTable } from '@/components/data/DataTable'
import { DataProfiler } from '@/components/data/DataProfiler'
import { TimeseriesPlot } from '@/components/charts/TimeseriesPlot'
import { CorrelationMatrix } from '@/components/charts/CorrelationMatrix'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

type Tab = 'import' | 'explore' | 'config'
type ExploreSubTab = 'apercu' | 'qualite' | 'series' | 'correlation'

export default function DataPage() {
  const [searchParams] = useSearchParams()
  const stationFromUrl = searchParams.get('station')
  const [tab, setTab] = useState<Tab>('import')
  const { data: datasets, isLoading } = useDatasets()
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('')

  // If navigated with ?station=, ensure we're on import tab and clear the param
  useEffect(() => {
    if (stationFromUrl) {
      setTab('import')
    }
  }, [stationFromUrl])

  const [exploreSubTab, setExploreSubTab] = useState<ExploreSubTab>('apercu')
  const [targetVariable, setTargetVariable] = useState('')
  const [selectedCovariates, setSelectedCovariates] = useState<string[]>([])
  const [fillMethod, setFillMethod] = useState('interpolate')
  const [normalize, setNormalize] = useState(true)

  const dsCount = datasets?.length ?? 0
  const tabs: { key: Tab; label: string }[] = [
    { key: 'import', label: 'Importer' },
    { key: 'explore', label: dsCount > 0 ? `Explorer (${dsCount})` : 'Explorer' },
    { key: 'config', label: 'Configurer' },
  ]

  const selectedDataset = datasets?.find((d) => d.id === selectedDatasetId)

  // Derive available variables from the selected dataset
  const datasetVariables = selectedDataset
    ? [
        selectedDataset.target_variable,
        ...selectedDataset.covariates,
      ].filter(Boolean)
    : []

  const toggleCovariate = (c: string) => {
    setSelectedCovariates((prev) =>
      prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c],
    )
  }

  // When dataset changes, update target/covariates
  const handleDatasetChange = (id: string, ds?: DatasetSummary) => {
    setSelectedDatasetId(id)
    const found = ds ?? datasets?.find((d) => d.id === id)
    if (found) {
      setTargetVariable(found.target_variable || '')
      setSelectedCovariates(found.covariates || [])
    }
  }

  // Fetch real preview and profile data for the selected dataset
  const { data: preview } = useDatasetPreview(selectedDatasetId || null)
  const { data: profile } = useDatasetProfile(selectedDatasetId || null)

  // Transform preview data for DataTable
  const previewColumns = preview?.columns ?? []
  const previewRows: (string | number | null)[][] = useMemo(() => {
    if (!preview?.columns || !preview?.rows) return []
    return preview.rows.map((row) =>
      preview.columns.map((col) => {
        const val = row[col]
        if (val === null || val === undefined) return null
        if (typeof val === 'string' || typeof val === 'number') return val
        return String(val)
      }),
    )
  }, [preview])

  // Transform profile data for DataProfiler
  const profileStats = useMemo(() => {
    if (!profile?.columns) return []
    return Object.entries(profile.columns).map(([column, info]) => ({
      column,
      count: typeof info.count === 'number' ? info.count : 0,
      mean: typeof info.mean === 'number' ? info.mean : null,
      std: typeof info.std === 'number' ? info.std : null,
      min: typeof info.min === 'number' ? info.min : null,
      max: typeof info.max === 'number' ? info.max : null,
      missing: profile.missing?.[column] ?? 0,
      missingPct:
        profile.shape?.[0] && profile.shape[0] > 0
          ? ((profile.missing?.[column] ?? 0) / profile.shape[0]) * 100
          : 0,
    }))
  }, [profile])

  // Extract first numeric series for TimeseriesPlot
  const timeseriesData = useMemo(() => {
    if (!profile?.timeseries_data) return { dates: [] as string[], values: [] as (number | null)[], label: '' }
    const { dates, series } = profile.timeseries_data
    // Prefer the target variable, fall back to first available series
    const targetKey = selectedDataset?.target_variable
    const seriesKey = targetKey && series[targetKey] ? targetKey : Object.keys(series)[0]
    if (!seriesKey) return { dates: [] as string[], values: [] as (number | null)[], label: '' }
    return { dates, values: series[seriesKey], label: seriesKey }
  }, [profile, selectedDataset])

  // Transform correlation data for CorrelationMatrix
  const correlationData = useMemo(() => {
    if (!profile?.correlation) return { labels: [] as string[], matrix: [] as number[][] }
    const labels = Object.keys(profile.correlation)
    const matrix = labels.map((row) =>
      labels.map((col) => profile.correlation![row]?.[col] ?? 0),
    )
    return { labels, matrix }
  }, [profile])

  // Data quality stats derived from profile
  const qualityStats = useMemo(() => {
    if (!profile) return null
    const totalRows = profile.shape?.[0] ?? 0
    const columns = Object.keys(profile.missing ?? {})
    const missingPerColumn = columns.map((col) => ({
      column: col,
      missing: profile.missing?.[col] ?? 0,
      completeness: totalRows > 0 ? ((totalRows - (profile.missing?.[col] ?? 0)) / totalRows) * 100 : 0,
    }))
    const totalMissing = columns.reduce((sum, col) => sum + (profile.missing?.[col] ?? 0), 0)
    const overallCompleteness = totalRows > 0 && columns.length > 0
      ? ((totalRows * columns.length - totalMissing) / (totalRows * columns.length)) * 100
      : 0
    return {
      totalRows,
      totalColumns: columns.length,
      missingPerColumn,
      overallCompleteness,
    }
  }, [profile])

  // Date range from selected dataset
  const dateRange = selectedDataset?.date_range

  const exploreSubTabs: { key: ExploreSubTab; label: string }[] = [
    { key: 'apercu', label: 'Apercu' },
    { key: 'qualite', label: 'Qualite' },
    { key: 'series', label: 'Serie temporelle' },
    { key: 'correlation', label: 'Correlation' },
  ]

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-1">Donnees</h1>
        <p className="text-sm text-text-secondary">
          Importation, exploration et configuration des donnees
        </p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-white/10">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-sm transition-colors ${
              tab === t.key
                ? 'border-b-2 border-accent-cyan text-accent-cyan'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Import tab */}
      {tab === 'import' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-bg-card rounded-xl border border-white/5 p-5">
            <ImportDBForm
              initialStation={stationFromUrl ?? undefined}
              onImportSuccess={(id, ds) => {
                handleDatasetChange(id, ds)
                setTab('explore')
              }}
            />
          </div>
          <div className="bg-bg-card rounded-xl border border-white/5 p-5">
            <ImportCSVForm />
          </div>
        </div>
      )}

      {/* Explore tab */}
      {tab === 'explore' && (
        <div className="space-y-6">
          {/* Dataset grid */}
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="h-28 bg-bg-card rounded-xl animate-pulse border border-white/5" />
              ))}
            </div>
          ) : datasets && datasets.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {datasets.map((ds) => (
                <button
                  key={ds.id}
                  onClick={() => handleDatasetChange(ds.id)}
                  className={`text-left transition-all ${
                    selectedDatasetId === ds.id
                      ? 'ring-2 ring-accent-cyan rounded-xl'
                      : 'hover:ring-1 hover:ring-white/20 rounded-xl'
                  }`}
                >
                  <DatasetCard dataset={ds} />
                </button>
              ))}
            </div>
          ) : (
            <div className="bg-bg-card rounded-xl border border-white/5 p-12 text-center">
              <p className="text-sm text-text-secondary">
                Aucun dataset. Importez des donnees depuis l'onglet Importer.
              </p>
            </div>
          )}

          {/* Detail view when dataset selected */}
          {selectedDataset && (
            <>
              {/* Sub-tabs */}
              <div className="flex border-b border-white/10">
                {exploreSubTabs.map((st) => (
                  <button
                    key={st.key}
                    onClick={() => setExploreSubTab(st.key)}
                    className={`px-4 py-2 text-xs transition-colors ${
                      exploreSubTab === st.key
                        ? 'border-b-2 border-accent-cyan text-accent-cyan'
                        : 'text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    {st.label}
                  </button>
                ))}
              </div>

              {/* Apercu sub-tab */}
              {exploreSubTab === 'apercu' && (
                <>
                  <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                    <h3 className="text-sm font-semibold text-text-primary mb-3">
                      Apercu des donnees
                    </h3>
                    <DataTable columns={previewColumns} rows={previewRows} />
                  </div>
                  <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                    <DataProfiler stats={profileStats} />
                  </div>
                </>
              )}

              {/* Qualite sub-tab */}
              {exploreSubTab === 'qualite' && qualityStats && (
                <div className="space-y-4">
                  {/* Summary cards */}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="bg-bg-card rounded-xl border border-white/5 p-4 text-center">
                      <p className="text-[10px] text-text-secondary uppercase tracking-wide">Lignes</p>
                      <p className="text-2xl font-bold text-text-primary">{qualityStats.totalRows.toLocaleString()}</p>
                    </div>
                    <div className="bg-bg-card rounded-xl border border-white/5 p-4 text-center">
                      <p className="text-[10px] text-text-secondary uppercase tracking-wide">Colonnes</p>
                      <p className="text-2xl font-bold text-text-primary">{qualityStats.totalColumns}</p>
                    </div>
                    <div className="bg-bg-card rounded-xl border border-white/5 p-4 text-center">
                      <p className="text-[10px] text-text-secondary uppercase tracking-wide">Completude globale</p>
                      <p className={`text-2xl font-bold ${qualityStats.overallCompleteness >= 95 ? 'text-accent-green' : qualityStats.overallCompleteness >= 80 ? 'text-yellow-400' : 'text-accent-red'}`}>
                        {qualityStats.overallCompleteness.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-bg-card rounded-xl border border-white/5 p-4 text-center">
                      <p className="text-[10px] text-text-secondary uppercase tracking-wide">Plage temporelle</p>
                      <p className="text-sm font-medium text-text-primary mt-1">
                        {dateRange && dateRange.length >= 2
                          ? `${new Date(dateRange[0]).toLocaleDateString('fr-FR')} — ${new Date(dateRange[1]).toLocaleDateString('fr-FR')}`
                          : '—'}
                      </p>
                    </div>
                  </div>

                  {/* Missing values bar chart */}
                  <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                    <h3 className="text-sm font-semibold text-text-primary mb-3">Valeurs manquantes par colonne</h3>
                    <div className="h-[300px]">
                      <Plot
                        data={[
                          {
                            x: qualityStats.missingPerColumn.map((c) => c.column),
                            y: qualityStats.missingPerColumn.map((c) => c.missing),
                            type: 'bar' as const,
                            marker: {
                              color: qualityStats.missingPerColumn.map((c) =>
                                c.missing === 0 ? '#22c55e' : c.completeness >= 95 ? '#06b6d4' : c.completeness >= 80 ? '#eab308' : '#ef4444'
                              ),
                            },
                            text: qualityStats.missingPerColumn.map((c) => `${c.missing} (${(100 - c.completeness).toFixed(1)}%)`),
                            textposition: 'auto' as const,
                          },
                        ]}
                        layout={{
                          ...darkLayout,
                          xaxis: { ...darkLayout.xaxis, title: { text: 'Colonne' }, tickangle: -45 },
                          yaxis: { ...darkLayout.yaxis, title: { text: 'Valeurs manquantes' } },
                          margin: { t: 20, r: 20, b: 80, l: 60 },
                        }}
                        config={plotlyConfig}
                        useResizeHandler
                        style={{ width: '100%', height: '100%' }}
                      />
                    </div>
                  </div>

                  {/* Completeness per column table */}
                  <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                    <h3 className="text-sm font-semibold text-text-primary mb-3">Completude par colonne</h3>
                    <div className="overflow-x-auto rounded-lg border border-white/5">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="bg-bg-hover">
                            <th className="px-3 py-2 text-left text-xs font-medium text-text-secondary uppercase">Colonne</th>
                            <th className="px-3 py-2 text-left text-xs font-medium text-text-secondary uppercase">Type</th>
                            <th className="px-3 py-2 text-right text-xs font-medium text-text-secondary uppercase">Manquantes</th>
                            <th className="px-3 py-2 text-right text-xs font-medium text-text-secondary uppercase">Completude</th>
                            <th className="px-3 py-2 text-left text-xs font-medium text-text-secondary uppercase w-48"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {qualityStats.missingPerColumn.map((c) => (
                            <tr key={c.column} className="border-t border-white/5 hover:bg-bg-hover/50">
                              <td className="px-3 py-1.5 text-text-primary font-medium">{c.column}</td>
                              <td className="px-3 py-1.5 text-text-secondary text-xs">{profile?.dtypes?.[c.column] ?? '—'}</td>
                              <td className="px-3 py-1.5 text-text-secondary text-right">{c.missing.toLocaleString()}</td>
                              <td className={`px-3 py-1.5 text-right font-medium ${c.completeness >= 95 ? 'text-accent-green' : c.completeness >= 80 ? 'text-yellow-400' : 'text-accent-red'}`}>
                                {c.completeness.toFixed(1)}%
                              </td>
                              <td className="px-3 py-1.5">
                                <div className="h-2 bg-bg-hover rounded-full overflow-hidden">
                                  <div
                                    className={`h-full rounded-full ${c.completeness >= 95 ? 'bg-accent-green' : c.completeness >= 80 ? 'bg-yellow-400' : 'bg-accent-red'}`}
                                    style={{ width: `${c.completeness}%` }}
                                  />
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* Serie temporelle sub-tab */}
              {exploreSubTab === 'series' && (
                <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                  <h3 className="text-sm font-semibold text-text-primary mb-3">Serie temporelle</h3>
                  <TimeseriesPlot
                    dates={timeseriesData.dates}
                    values={timeseriesData.values}
                    label={timeseriesData.label || selectedDataset.target_variable}
                    className="h-[300px]"
                  />
                </div>
              )}

              {/* Correlation sub-tab */}
              {exploreSubTab === 'correlation' && (
                <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                  <h3 className="text-sm font-semibold text-text-primary mb-3">
                    Matrice de correlation
                  </h3>
                  <CorrelationMatrix
                    labels={correlationData.labels}
                    matrix={correlationData.matrix}
                    className="h-[400px]"
                  />
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Config tab */}
      {tab === 'config' && (
        <div className="max-w-2xl space-y-6">
          {/* Dataset selector */}
          <div className="bg-bg-card rounded-xl border border-white/5 p-5 space-y-3">
            <h3 className="text-sm font-semibold text-text-primary">Dataset</h3>
            <select
              value={selectedDatasetId}
              onChange={(e) => handleDatasetChange(e.target.value)}
              className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
            >
              <option value="">Selectionner un dataset</option>
              {datasets?.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.n_rows} lignes)
                </option>
              ))}
            </select>
          </div>

          {datasetVariables.length > 0 ? (
            <>
              <div className="bg-bg-card rounded-xl border border-white/5 p-5 space-y-4">
                <h3 className="text-sm font-semibold text-text-primary">Variable cible</h3>
                <select
                  value={targetVariable}
                  onChange={(e) => setTargetVariable(e.target.value)}
                  className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                >
                  {datasetVariables.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </div>

              <div className="bg-bg-card rounded-xl border border-white/5 p-5 space-y-4">
                <h3 className="text-sm font-semibold text-text-primary">Covariables</h3>
                <div className="flex flex-wrap gap-2">
                  {datasetVariables
                    .filter((v) => v !== targetVariable)
                    .map((v) => (
                      <button
                        key={v}
                        onClick={() => toggleCovariate(v)}
                        className={`text-xs px-2.5 py-1 rounded-lg border transition-colors ${
                          selectedCovariates.includes(v)
                            ? 'bg-accent-cyan/10 border-accent-cyan text-accent-cyan'
                            : 'bg-bg-hover border-white/10 text-text-secondary hover:text-text-primary'
                        }`}
                      >
                        {v}
                      </button>
                    ))}
                </div>
              </div>

              <div className="bg-bg-card rounded-xl border border-white/5 p-5 space-y-4">
                <h3 className="text-sm font-semibold text-text-primary">Pretraitement</h3>
                <div>
                  <label className="block text-xs text-text-secondary mb-1">
                    Methode de remplissage des valeurs manquantes
                  </label>
                  <select
                    value={fillMethod}
                    onChange={(e) => setFillMethod(e.target.value)}
                    className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                  >
                    <option value="interpolate">Interpolation lineaire</option>
                    <option value="ffill">Forward fill</option>
                    <option value="bfill">Backward fill</option>
                    <option value="drop">Supprimer</option>
                  </select>
                </div>
                <div className="flex items-center gap-3">
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={normalize}
                      onChange={(e) => setNormalize(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-9 h-5 bg-bg-hover rounded-full peer peer-checked:bg-accent-cyan/30 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:after:translate-x-full peer-checked:after:bg-accent-cyan" />
                  </label>
                  <span className="text-xs text-text-secondary">
                    Normalisation (StandardScaler)
                  </span>
                </div>
              </div>
            </>
          ) : (
            <div className="bg-bg-card rounded-xl border border-white/5 p-12 text-center">
              <p className="text-sm text-text-secondary">
                Selectionnez un dataset pour configurer les variables.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
