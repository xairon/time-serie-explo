import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useDatasets } from '@/hooks/useDatasets'
import { ImportDBForm } from '@/components/data/ImportDBForm'
import { ImportCSVForm } from '@/components/data/ImportCSVForm'
import { DataTable } from '@/components/data/DataTable'
import { DataProfiler } from '@/components/data/DataProfiler'
import { TimeseriesPlot } from '@/components/charts/TimeseriesPlot'
import { CorrelationMatrix } from '@/components/charts/CorrelationMatrix'

type Tab = 'import' | 'explore' | 'config'

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

  const [targetVariable, setTargetVariable] = useState('')
  const [selectedCovariates, setSelectedCovariates] = useState<string[]>([])
  const [fillMethod, setFillMethod] = useState('interpolate')
  const [normalize, setNormalize] = useState(true)

  const tabs: { key: Tab; label: string }[] = [
    { key: 'import', label: 'Importer' },
    { key: 'explore', label: 'Explorer' },
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
  const handleDatasetChange = (id: string) => {
    setSelectedDatasetId(id)
    const ds = datasets?.find((d) => d.id === id)
    if (ds) {
      setTargetVariable(ds.target_variable || '')
      setSelectedCovariates(ds.covariates || [])
    }
  }

  // Preview data placeholders (populated by real API data when available)
  const previewColumns = selectedDataset
    ? ['date', selectedDataset.target_variable, ...selectedDataset.covariates.slice(0, 3)]
    : []
  const previewRows: (string | number | null)[][] = []

  const mockStats = selectedDataset
    ? [selectedDataset.target_variable, ...selectedDataset.covariates].map((col) => ({
        column: col,
        count: selectedDataset.n_rows,
        mean: null as number | null,
        std: null as number | null,
        min: null as number | null,
        max: null as number | null,
        missing: 0,
        missingPct: 0,
      }))
    : []

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-1">Données</h1>
        <p className="text-sm text-text-secondary">
          Importation, exploration et configuration des données
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
            <ImportDBForm initialStation={stationFromUrl ?? undefined} />
          </div>
          <div className="bg-bg-card rounded-xl border border-white/5 p-5">
            <ImportCSVForm />
          </div>
        </div>
      )}

      {/* Explore tab */}
      {tab === 'explore' && (
        <div className="space-y-6">
          <div>
            <label className="block text-xs text-text-secondary mb-1">Dataset</label>
            {isLoading ? (
              <div className="h-9 bg-bg-hover rounded-lg animate-pulse" />
            ) : (
              <select
                value={selectedDatasetId}
                onChange={(e) => handleDatasetChange(e.target.value)}
                className="w-full max-w-md bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
              >
                <option value="">Sélectionner un dataset</option>
                {datasets?.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.n_rows} lignes)
                  </option>
                ))}
              </select>
            )}
          </div>

          {selectedDataset ? (
            <>
              {/* Data table */}
              <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                <h3 className="text-sm font-semibold text-text-primary mb-3">
                  Aperçu des données
                </h3>
                <DataTable columns={previewColumns} rows={previewRows} />
              </div>

              {/* Profiler */}
              <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                <DataProfiler stats={mockStats} />
              </div>

              {/* Time series plot */}
              <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                <h3 className="text-sm font-semibold text-text-primary mb-3">Série temporelle</h3>
                <TimeseriesPlot
                  dates={[]}
                  values={[]}
                  label={selectedDataset.target_variable}
                  className="h-[300px]"
                />
              </div>

              {/* Correlation matrix */}
              <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                <h3 className="text-sm font-semibold text-text-primary mb-3">
                  Matrice de corrélation
                </h3>
                <CorrelationMatrix labels={[]} matrix={[]} className="h-[400px]" />
              </div>
            </>
          ) : (
            <div className="bg-bg-card rounded-xl border border-white/5 p-12 text-center">
              <p className="text-sm text-text-secondary">
                Sélectionnez un dataset pour explorer les données.
              </p>
            </div>
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
              <option value="">Sélectionner un dataset</option>
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
                <h3 className="text-sm font-semibold text-text-primary">Prétraitement</h3>
                <div>
                  <label className="block text-xs text-text-secondary mb-1">
                    Méthode de remplissage des valeurs manquantes
                  </label>
                  <select
                    value={fillMethod}
                    onChange={(e) => setFillMethod(e.target.value)}
                    className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
                  >
                    <option value="interpolate">Interpolation linéaire</option>
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
                Sélectionnez un dataset pour configurer les variables.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
