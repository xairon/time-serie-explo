import { useState, useMemo, useCallback, useEffect } from 'react'
import { AlertTriangle } from 'lucide-react'
import { useStationEmbeddings, useComputeUMAP } from '@/hooks/useLatentSpace'
import { EmbeddingScatter } from '@/components/latent-space/EmbeddingScatter'
import { FilterPanel } from '@/components/latent-space/FilterPanel'
import { UMAPControls } from '@/components/latent-space/UMAPControls'
import { StationDetail } from '@/components/latent-space/StationDetail'
import { QualityMetrics } from '@/components/latent-space/QualityMetrics'
import { ClusterProfiling } from '@/components/latent-space/ClusterProfiling'

type Domain = 'piezo' | 'hydro'
type Mode = '2d' | '3d'
type Level = 'stations' | 'windows'

interface StationRaw {
  id: string
  umap_2d: [number, number] | null
  umap_3d: [number, number, number] | null
  cluster_id: number | null
  n_windows: number | null
  metadata: Record<string, unknown>
}

interface ComputedPointRaw {
  id: string
  coords: number[]
  cluster_label: number
  window_start?: string
  window_end?: string
  metadata: Record<string, unknown>
}

export default function LatentSpacePage() {
  // Core state
  const [domain, setDomain] = useState<Domain>('piezo')
  const [filters, setFilters] = useState<Record<string, string | number | null>>({})
  const [colorBy, setColorBy] = useState('cluster_id')
  const [mode, setMode] = useState<Mode>('2d')
  const [level, setLevel] = useState<Level>('stations')
  const [umapParams, setUmapParams] = useState({ n_neighbors: 30, min_dist: 0.05 })
  const [clusteringParams, setClusteringParams] = useState<{
    method: 'hdbscan' | 'kmeans'
    min_cluster_size: number
    min_samples: number
    n_clusters: number
    n_umap_dims: number
  }>({
    method: 'hdbscan',
    min_cluster_size: 10,
    min_samples: 5,
    n_clusters: 8,
    n_umap_dims: 10,
  })
  const [selectedStation, setSelectedStation] = useState<string | null>(null)
  const [hideUnclassified, setHideUnclassified] = useState(false)
  const [activeTab, setActiveTab] = useState<'scatter' | 'profiling'>('scatter')
  const [yearRange, setYearRange] = useState<[number, number]>([2015, 2025])
  const [season, setSeason] = useState<string | null>(null)

  // Computed points from /compute endpoint (overrides pre-computed)
  const [computedPoints, setComputedPoints] = useState<ComputedPointRaw[] | null>(null)
  const [subsampled, setSubsampled] = useState<{ from: number } | null>(null)
  const [qualityMetrics, setQualityMetrics] = useState<Record<string, unknown> | null>(null)

  // Data fetching
  const { data: stationsData, isLoading, isError, refetch } = useStationEmbeddings(domain)
  const computeMutation = useComputeUMAP()

  // Extract raw stations from API response
  const allStations = useMemo(() => {
    if (!stationsData) return []
    return (stationsData.stations ?? []) as unknown as StationRaw[]
  }, [stationsData])

  const EH_KEYS = ['milieu_eh', 'theme_eh', 'etat_eh', 'nature_eh', 'libelle_eh']

  const stations = useMemo(() => {
    if (!hideUnclassified || domain !== 'piezo') return allStations
    return allStations.filter((s) =>
      EH_KEYS.some((k) => s.metadata[k] != null && s.metadata[k] !== ''),
    )
  }, [allStations, hideUnclassified, domain])

  // Apply client-side filters for highlight
  const matchesFilters = useCallback(
    (station: StationRaw) => {
      for (const [key, value] of Object.entries(filters)) {
        if (value === null || value === '') continue
        if (key === 'cluster_id') {
          if (station.cluster_id !== Number(value)) return false
        } else {
          if (String(station.metadata[key] ?? '') !== String(value)) return false
        }
      }
      return true
    },
    [filters],
  )

  const hasActiveFilters = Object.values(filters).some((v) => v !== null && v !== '')

  // Build scatter points from pre-computed or computed data
  const scatterPoints = useMemo(() => {
    if (computedPoints) {
      // Use computed points (all highlighted since server already filtered)
      return computedPoints.map((p) => ({
        id: p.id,
        coords: (mode === '3d' ? p.coords.slice(0, 3) : p.coords.slice(0, 2)) as
          | [number, number]
          | [number, number, number],
        cluster_label: p.cluster_label,
        metadata: p.metadata,
        highlighted: true,
      }))
    }

    // Use pre-computed station UMAP coords
    return stations
      .filter((s) => (mode === '3d' ? s.umap_3d : s.umap_2d))
      .map((s) => ({
        id: s.id,
        coords: (mode === '3d' ? s.umap_3d! : s.umap_2d!) as
          | [number, number]
          | [number, number, number],
        cluster_label: s.cluster_id ?? -1,
        metadata: s.metadata,
        highlighted: !hasActiveFilters || matchesFilters(s),
      }))
  }, [stations, computedPoints, mode, hasActiveFilters, matchesFilters])

  // Station list for FilterPanel (always from pre-computed, not computed points)
  const stationsForFilter = useMemo(
    () =>
      stations.map((s) => ({
        id: s.id,
        metadata: s.metadata,
        cluster_id: s.cluster_id,
      })),
    [stations],
  )

  // Selected station metadata for the detail panel
  const selectedStationMeta = useMemo(() => {
    if (!selectedStation) return undefined
    const s = allStations.find((st) => st.id === selectedStation)
    return s?.metadata as Record<string, unknown> | undefined
  }, [selectedStation, allStations])

  // When a station is selected (click or BSS search), auto-filter by its aquifer (piezo) or waterway (hydro)
  function handleStationSelect(stationId: string) {
    setSelectedStation(stationId)
    const station = stations.find((s) => s.id === stationId)
    if (!station) return

    if (domain === 'piezo') {
      const aquifer = station.metadata['libelle_eh']
      if (aquifer && typeof aquifer === 'string') {
        setFilters({ libelle_eh: aquifer })
      }
    } else {
      const waterway = station.metadata['nom_cours_eau']
      if (waterway && typeof waterway === 'string') {
        setFilters({ nom_cours_eau: waterway })
      }
    }
  }

  // Handle domain switch
  function handleDomainChange(d: Domain) {
    setDomain(d)
    setFilters({})
    setColorBy('cluster_id')
    setComputedPoints(null)
    setSubsampled(null)
    setSelectedStation(null)
  }

  // Handle recalculate
  async function handleRecalculate() {
    const body: Record<string, unknown> = {
      domain,
      embeddings_type: level,
      filters: {
        ...(filters.libelle_eh ? { libelle_eh: filters.libelle_eh } : {}),
        ...(filters.milieu_eh ? { milieu_eh: filters.milieu_eh } : {}),
        ...(filters.theme_eh ? { theme_eh: filters.theme_eh } : {}),
        ...(filters.etat_eh ? { etat_eh: filters.etat_eh } : {}),
        ...(filters.nature_eh ? { nature_eh: filters.nature_eh } : {}),
        ...(filters.departement ? { departement: filters.departement } : {}),
        ...(filters.cluster_id ? { cluster_id: Number(filters.cluster_id) } : {}),
        ...(filters.nom_cours_eau ? { nom_cours_eau: filters.nom_cours_eau } : {}),
      },
      umap: {
        n_components: mode === '3d' ? 3 : 2,
        n_neighbors: umapParams.n_neighbors,
        min_dist: umapParams.min_dist,
        metric: 'cosine',
      },
      clustering: {
        method: clusteringParams.method,
        n_umap_dims: clusteringParams.n_umap_dims,
        hdbscan: {
          min_cluster_size: clusteringParams.min_cluster_size,
          min_samples: clusteringParams.min_samples,
        },
        kmeans: { n_clusters: clusteringParams.n_clusters },
      },
    }

    if (level === 'windows') {
      body.year_min = yearRange[0]
      body.year_max = yearRange[1]
      if (season) body.season = season
    }

    try {
      const result = (await computeMutation.mutateAsync(body)) as {
        points: ComputedPointRaw[]
        subsampled: boolean
        subsampled_from: number | null
        metrics: Record<string, unknown> | null
      }
      setComputedPoints(result.points)
      setSubsampled(result.subsampled && result.subsampled_from ? { from: result.subsampled_from } : null)
      setQualityMetrics(result.metrics ?? null)
    } catch {
      // Error handled by mutation state
    }
  }

  // Handle reset
  function handleReset() {
    setComputedPoints(null)
    setSubsampled(null)
    setQualityMetrics(null)
  }

  // Auto-compute UMAP when stations load but have no pre-computed coords
  const hasPrecomputedUMAP = useMemo(
    () => stations.some((s) => s.umap_2d !== null),
    [stations],
  )
  useEffect(() => {
    if (stations.length > 0 && !hasPrecomputedUMAP && !computedPoints && !computeMutation.isPending) {
      handleRecalculate()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stations.length, hasPrecomputedUMAP])

  const highlightedCount = scatterPoints.filter((p) => p.highlighted).length
  const totalCount = scatterPoints.length

  // Shared top bar buttons
  const domainButtons = (
    <div className="flex rounded-lg overflow-hidden border border-white/10">
      <button
        onClick={() => handleDomainChange('piezo')}
        className={`px-4 py-2 text-sm transition-colors ${
          domain === 'piezo'
            ? 'bg-accent-cyan/20 text-accent-cyan'
            : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
        }`}
      >
        Piezometry
      </button>
      <button
        onClick={() => handleDomainChange('hydro')}
        className={`px-4 py-2 text-sm transition-colors ${
          domain === 'hydro'
            ? 'bg-accent-cyan/20 text-accent-cyan'
            : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
        }`}
      >
        Hydrometry
      </button>
    </div>
  )

  const tabButtons = (
    <div className="flex rounded-lg overflow-hidden border border-white/10">
      <button
        onClick={() => setActiveTab('scatter')}
        className={`px-4 py-2 text-sm transition-colors ${
          activeTab === 'scatter'
            ? 'bg-accent-cyan/20 text-accent-cyan'
            : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
        }`}
      >
        Scatter
      </button>
      <button
        onClick={() => setActiveTab('profiling')}
        className={`px-4 py-2 text-sm transition-colors ${
          activeTab === 'profiling'
            ? 'bg-accent-cyan/20 text-accent-cyan'
            : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
        }`}
      >
        Profiling
      </button>
    </div>
  )

  // Loading state
  if (isLoading) {
    return (
      <div className="flex flex-col h-full gap-3 p-4 overflow-hidden">
        <div className="flex items-center gap-4 shrink-0">
          {domainButtons}
          {tabButtons}
        </div>

        {activeTab === 'profiling' ? (
          <ClusterProfiling domain={domain} hideUnclassified={hideUnclassified} />
        ) : (
          <div className="flex items-center justify-center flex-1">
            <div className="flex flex-col items-center gap-3">
              <div className="w-10 h-10 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin" />
              <span className="text-text-secondary text-sm">Loading embeddings...</span>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Error state
  if (isError) {
    return (
      <div className="flex flex-col h-full gap-3 p-4 overflow-hidden">
        <div className="flex items-center gap-4 shrink-0">
          {domainButtons}
          {tabButtons}
        </div>

        {activeTab === 'profiling' ? (
          <ClusterProfiling domain={domain} hideUnclassified={hideUnclassified} />
        ) : (
          <div className="flex items-center justify-center flex-1">
            <div className="bg-bg-card rounded-xl border border-white/5 p-8 flex flex-col items-center gap-4 max-w-md">
              <AlertTriangle className="w-10 h-10 text-accent-red" />
              <p className="text-text-primary text-center">BRGM database unavailable</p>
              <p className="text-text-muted text-sm text-center">
                Unable to load embeddings. Check brgm-postgres connection.
              </p>
              <button
                onClick={() => refetch()}
                className="bg-accent-cyan text-white px-4 py-2 rounded-lg text-sm hover:bg-accent-cyan/90 transition-colors"
              >
                Retry
              </button>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full gap-3 p-4 overflow-hidden">
      {/* Top bar: domain switch + tab toggle + stats */}
      <div className="flex items-center gap-4 shrink-0">
        {domainButtons}
        {tabButtons}

        <span className="text-text-muted text-sm">
          {hasActiveFilters
            ? `${highlightedCount} / ${totalCount} stations`
            : `${totalCount} stations`}
        </span>

        {subsampled && (
          <div className="flex items-center gap-1.5 bg-amber-500/10 text-amber-400 px-3 py-1 rounded-lg text-xs">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>
              {scatterPoints.length.toLocaleString()} / {subsampled.from.toLocaleString()} points
              (subsampled)
            </span>
          </div>
        )}

        {computeMutation.isError && (
          <span className="text-accent-red text-xs">
            UMAP error: {(computeMutation.error as Error)?.message ?? 'unknown error'}
          </span>
        )}
      </div>

      {/* Main content: filter sidebar + (scatter or profiling) + detail */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Filter sidebar — always visible */}
        <div className="shrink-0 overflow-y-auto">
          <FilterPanel
            domain={domain}
            stations={stationsForFilter}
            filters={filters}
            onFiltersChange={setFilters}
            colorBy={colorBy}
            onColorByChange={setColorBy}
            onStationSelect={handleStationSelect}
            hideUnclassified={hideUnclassified}
            onHideUnclassifiedChange={setHideUnclassified}
          />
        </div>

        {activeTab === 'profiling' ? (
          <div className="flex-1 min-w-0 overflow-y-auto">
            <ClusterProfiling domain={domain} hideUnclassified={hideUnclassified} />
          </div>
        ) : (
          <>
            {/* Scatter + controls */}
            <div className="flex-1 flex flex-col min-w-0 gap-2">
              {/* Empty state */}
              {scatterPoints.length === 0 && !computeMutation.isPending ? (
                <div className="flex-1 flex items-center justify-center">
                  <div className="flex flex-col items-center gap-3">
                    <p className="text-text-muted text-sm">
                      {stations.length > 0
                        ? 'Computing UMAP projection...'
                        : 'No station embeddings found.'}
                    </p>
                    {stations.length > 0 && (
                      <button
                        onClick={handleRecalculate}
                        className="bg-accent-cyan text-white px-4 py-2 rounded-lg text-sm hover:bg-accent-cyan/90 transition-colors"
                      >
                        Compute now
                      </button>
                    )}
                  </div>
                </div>
              ) : highlightedCount === 0 && hasActiveFilters ? (
                <div className="flex-1 flex items-center justify-center">
                  <p className="text-text-muted text-sm">
                    No stations match the selected filters.
                  </p>
                </div>
              ) : (
                <div className="flex-1 min-h-0">
                  <EmbeddingScatter
                    points={scatterPoints}
                    mode={mode}
                    colorBy={colorBy}
                    onPointClick={handleStationSelect}
                    loading={computeMutation.isPending}
                    className="h-full"
                  />
                </div>
              )}

              {/* Controls bar */}
              <div className="shrink-0 bg-bg-card rounded-xl border border-white/5 px-3 py-1">
                <UMAPControls
                  mode={mode}
                  onModeChange={setMode}
                  level={level}
                  onLevelChange={(l) => {
                    setLevel(l)
                    setComputedPoints(null)
                    setSubsampled(null)
                  }}
                  umapParams={umapParams}
                  onUmapParamsChange={setUmapParams}
                  clusteringParams={clusteringParams}
                  onClusteringParamsChange={setClusteringParams}
                  onRecalculate={handleRecalculate}
                  onReset={handleReset}
                  isComputing={computeMutation.isPending}
                  yearRange={yearRange}
                  onYearRangeChange={setYearRange}
                  season={season}
                  onSeasonChange={setSeason}
                />
              </div>
            </div>

            {/* Right sidebar: station detail + quality metrics */}
            <div className="shrink-0 flex flex-col gap-3 overflow-y-auto">
              {selectedStation && (
                <StationDetail
                  domain={domain}
                  stationId={selectedStation}
                  stationMeta={selectedStationMeta}
                  onClose={() => {
                    setSelectedStation(null)
                    setFilters({})
                  }}
                  onNeighborClick={handleStationSelect}
                />
              )}
              {qualityMetrics && (
                <QualityMetrics metrics={qualityMetrics as Record<string, unknown>} />
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
