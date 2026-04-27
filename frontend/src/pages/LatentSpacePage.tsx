import { useState, useMemo, useCallback, useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { AlertTriangle } from 'lucide-react'
import { useStationEmbeddings, useCachedCompute, useRecomputePca, useRecomputeViz, useRecomputeClustering, useAutoTune } from '@/hooks/useLatentSpace'
import { EmbeddingScatter } from '@/components/latent-space/EmbeddingScatter'
import { FilterPanel } from '@/components/latent-space/FilterPanel'
import { PipelineControls } from '@/components/latent-space/PipelineControls'
import { StationDetail } from '@/components/latent-space/StationDetail'
import { ClusterProfiling } from '@/components/latent-space/ClusterProfiling'
import { ClusterLegendBar } from '@/components/latent-space/ClusterLegendBar'
import { ElbowChart } from '@/components/latent-space/ElbowChart'

type Domain = 'piezo' | 'hydro'
type Mode = '2d' | '3d'

interface StationRaw {
  id: string
  umap_2d: [number, number] | null
  umap_3d: [number, number, number] | null
  cluster_id: number | null
  n_windows: number | null
  last_date: string | null
  metadata: Record<string, unknown>
}

interface CacheData {
  station_ids: string[]
  metadata: Record<string, unknown>[]
  pca: {
    n_components: number
    variance_threshold: number
    variance_explained: number
    cumvar_curve: number[]
    embedding_dim: number
  }
  viz: {
    coords_2d: [number, number][]
    params: { n_neighbors: number; min_dist: number }
    trustworthiness: number
  }
  clustering: {
    hdbscan: {
      params: { min_cluster_size: number; min_samples: number }
      labels: number[]
      n_clusters: number
      dbcv: number
      noise_ratio: number
      silhouette: number
    }
    kmeans_elbow: Array<{
      k: number
      labels: number[]
      silhouette: number
      inertia: number
    }>
  }
}

const CATEGORICAL_COLORS = [
  '#06b6d4', '#8b5cf6', '#f59e0b', '#10b981', '#ef4444',
  '#3b82f6', '#ec4899', '#14b8a6', '#f97316', '#a78bfa',
  '#84cc16', '#fb7185', '#22d3ee', '#fbbf24', '#60a5fa',
]

export default function LatentSpacePage() {
  const queryClient = useQueryClient()

  // Core state
  const [domain, setDomain] = useState<Domain>('piezo')
  const [filters, setFilters] = useState<Record<string, string | number | null>>({})
  const [colorBy, setColorBy] = useState('cluster_id')
  const [mode, setMode] = useState<Mode>('2d')
  const [selectedStation, setSelectedStation] = useState<string | null>(null)
  const [hideUnclassified, setHideUnclassified] = useState(false)
  const [activeTab, setActiveTab] = useState<'scatter' | 'profiling'>('scatter')
  const [space, setSpace] = useState<'uni' | 'multi'>('multi')
  const [highlightedSite, setHighlightedSite] = useState<string | null>(null)
  const [legendCluster, setLegendCluster] = useState<number | null>(null)
  const [onlyActive, setOnlyActive] = useState(false)

  // Pipeline params (synced from cache on load)
  const [varianceThreshold, setVarianceThreshold] = useState(0.95)
  const [vizNNeighbors, setVizNNeighbors] = useState(50)
  const [vizMinDist, setVizMinDist] = useState(0.3)
  const [hdbscanMcs, setHdbscanMcs] = useState(25)
  const [hdbscanMs, setHdbscanMs] = useState(5)

  // Clustering display (instant switch, no API call)
  const [cachedMethod, setCachedMethod] = useState<'hdbscan' | 'kmeans'>('hdbscan')
  const [cachedKmeansK, setCachedKmeansK] = useState<number | null>(null)
  const initialSyncDone = useRef(false)

  // Mode toggles
  const [level, setLevel] = useState<'stations' | 'windows'>('stations')
  const [yearRange, setYearRange] = useState<[number, number]>([2015, 2025])
  const [season, setSeason] = useState<string | null>(null)

  // Data fetching
  const { data: stationsData, isLoading, isError, refetch } = useStationEmbeddings(domain, space)
  const { data: cachedData, isLoading: isCacheLoading } = useCachedCompute(domain, space)
  const pcaMutation = useRecomputePca()
  const vizMutation = useRecomputeViz()
  const clusteringMutation = useRecomputeClustering()
  const autoTuneMutation = useAutoTune()

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

  const activeStations = useMemo(() => {
    if (!onlyActive) return stations
    return stations.filter((s) => s.last_date != null && s.last_date >= '2024-01-01')
  }, [stations, onlyActive])

  // Build lookup maps for filtering
  const stationLookup = useMemo(() => {
    const m = new Map<string, StationRaw>()
    for (const s of allStations) m.set(s.id, s)
    return m
  }, [allStations])

  const isStationVisible = useCallback(
    (id: string) => {
      const s = stationLookup.get(id)
      if (!s) return false
      if (onlyActive && (s.last_date == null || s.last_date < '2024-01-01')) return false
      if (hideUnclassified && domain === 'piezo') {
        if (!EH_KEYS.some((k) => s.metadata[k] != null && s.metadata[k] !== '')) return false
      }
      return true
    },
    [onlyActive, hideUnclassified, domain, stationLookup],
  )

  // Apply client-side filters for highlight
  const matchesFilters = useCallback(
    (station: { id: string; cluster_id?: number | null; cluster_label?: number; metadata: Record<string, unknown> }) => {
      for (const [key, value] of Object.entries(filters)) {
        if (value === null || value === '') continue
        if (key === 'cluster_id') {
          const cid = station.cluster_id ?? station.cluster_label
          if (cid !== Number(value)) return false
        } else {
          if (String(station.metadata[key] ?? '') !== String(value)) return false
        }
      }
      return true
    },
    [filters],
  )

  const hasActiveFilters = Object.values(filters).some((v) => v !== null && v !== '')

  // Derive scatter points from cache — SINGLE source of truth
  const scatterPoints = useMemo(() => {
    const cd = cachedData as CacheData | undefined
    if (!cd?.station_ids || !cd?.viz?.coords_2d) return []

    // Pick labels based on selected method
    let labels: number[]
    if (cachedMethod === 'kmeans' && cd.clustering?.kmeans_elbow?.length) {
      const match = cd.clustering.kmeans_elbow.find(e => e.k === cachedKmeansK) ?? cd.clustering.kmeans_elbow[0]
      labels = match.labels
    } else {
      labels = cd.clustering?.hdbscan?.labels ?? cd.station_ids.map(() => -1)
    }

    return cd.station_ids.map((id, i) => ({
      id,
      coords: cd.viz.coords_2d[i] as [number, number],
      cluster_label: labels[i] ?? -1,
      metadata: cd.metadata?.[i] ?? {} as Record<string, unknown>,
      highlighted: !hasActiveFilters || matchesFilters({ id, cluster_label: labels[i], metadata: cd.metadata?.[i] ?? {} }),
    })).filter(p => isStationVisible(p.id))
  }, [cachedData, cachedMethod, cachedKmeansK, hasActiveFilters, matchesFilters, isStationVisible])

  // Elbow data from cache
  const cachedElbow = useMemo(() => {
    const cd = cachedData as CacheData | undefined
    return cd?.clustering?.kmeans_elbow ?? null
  }, [cachedData])

  // Clustering diagnostics from cache
  const clusteringDiagnostics = useMemo(() => {
    const cd = cachedData as CacheData | undefined
    if (!cd?.clustering?.hdbscan) return null
    const h = cd.clustering.hdbscan
    return { dbcv: h.dbcv, noise_ratio: h.noise_ratio, n_clusters: h.n_clusters }
  }, [cachedData])

  const clusterInfo = useMemo(() => {
    const counts = new Map<number, number>()
    for (const p of scatterPoints) {
      counts.set(p.cluster_label, (counts.get(p.cluster_label) ?? 0) + 1)
    }
    return Array.from(counts.entries()).map(([id, count]) => ({
      id,
      color: id === -1 ? '#4b5563' : CATEGORICAL_COLORS[Math.abs(id) % CATEGORICAL_COLORS.length],
      count,
    }))
  }, [scatterPoints])

  // Station list for FilterPanel (always from pre-computed, not computed points)
  const stationsForFilter = useMemo(
    () =>
      activeStations.map((s) => ({
        id: s.id,
        metadata: s.metadata,
        cluster_id: s.cluster_id,
      })),
    [activeStations],
  )

  // Selected station metadata for the detail panel
  const selectedStationMeta = useMemo(() => {
    if (!selectedStation) return undefined
    const s = allStations.find((st) => st.id === selectedStation)
    return s?.metadata as Record<string, unknown> | undefined
  }, [selectedStation, allStations])

  function handleStationSelect(stationId: string) {
    setSelectedStation(stationId)
    const station = allStations.find((s) => s.id === stationId)
    if (!station) return
    const siteKey = domain === 'piezo' ? 'libelle_eh' : 'nom_cours_eau'
    const siteValue = station.metadata[siteKey]
    if (siteValue && typeof siteValue === 'string') {
      setHighlightedSite(siteValue)
    }
  }

  function handleDeselectStation() {
    setSelectedStation(null)
    setHighlightedSite(null)
  }

  function handleFilterBySite() {
    if (!selectedStation) return
    const station = allStations.find((s) => s.id === selectedStation)
    if (!station) return
    const siteKey = domain === 'piezo' ? 'libelle_eh' : 'nom_cours_eau'
    const siteValue = station.metadata[siteKey]
    if (siteValue && typeof siteValue === 'string') {
      setFilters({ [siteKey]: siteValue })
    }
  }

  // Recompute handlers
  async function handleRecomputePca() {
    try {
      await pcaMutation.mutateAsync({ domain, space, variance_threshold: varianceThreshold })
      queryClient.invalidateQueries({ queryKey: ['latent-space', 'cached', domain, space] })
    } catch { /* mutation state handles error */ }
  }

  async function handleRecomputeViz() {
    try {
      await vizMutation.mutateAsync({ domain, space, n_neighbors: vizNNeighbors, min_dist: vizMinDist })
      queryClient.invalidateQueries({ queryKey: ['latent-space', 'cached', domain, space] })
    } catch { /* mutation state handles error */ }
  }

  async function handleRecomputeClustering() {
    try {
      await clusteringMutation.mutateAsync({ domain, space, min_cluster_size: hdbscanMcs, min_samples: hdbscanMs })
      queryClient.invalidateQueries({ queryKey: ['latent-space', 'cached', domain, space] })
    } catch { /* mutation state handles error */ }
  }

  async function handleAutoTune() {
    try {
      await autoTuneMutation.mutateAsync({ domain, space })
      queryClient.invalidateQueries({ queryKey: ['latent-space', 'cached', domain, space] })
    } catch { /* mutation state handles error */ }
  }

  // Sync params from cache on load
  useEffect(() => {
    const cd = cachedData as CacheData | undefined
    if (!cd || initialSyncDone.current) return
    initialSyncDone.current = true
    if (cd.pca) setVarianceThreshold(cd.pca.variance_threshold)
    if (cd.viz?.params) {
      setVizNNeighbors(cd.viz.params.n_neighbors)
      setVizMinDist(cd.viz.params.min_dist)
    }
    if (cd.clustering?.hdbscan?.params) {
      setHdbscanMcs(cd.clustering.hdbscan.params.min_cluster_size)
      setHdbscanMs(cd.clustering.hdbscan.params.min_samples)
    }
    if (cd.clustering?.kmeans_elbow?.length) {
      const best = cd.clustering.kmeans_elbow.reduce((a, b) => b.silhouette > a.silhouette ? b : a)
      setCachedKmeansK(best.k)
    }
  }, [cachedData])

  // Handle space switch
  function handleSpaceChange(s: 'uni' | 'multi') {
    setSpace(s)
    setSelectedStation(null)
    setHighlightedSite(null)
    setLegendCluster(null)
    setOnlyActive(false)
    setCachedMethod('hdbscan')
    setCachedKmeansK(null)
    initialSyncDone.current = false
  }

  // Handle domain switch
  function handleDomainChange(d: Domain) {
    setDomain(d)
    setFilters({})
    setColorBy('cluster_id')
    setSelectedStation(null)
    setHighlightedSite(null)
    setLegendCluster(null)
    setOnlyActive(false)
    setCachedMethod('hdbscan')
    setCachedKmeansK(null)
    initialSyncDone.current = false
  }

  const highlightedCount = scatterPoints.filter((p) => p.highlighted).length
  const totalCount = scatterPoints.length

  const anyMutationPending = pcaMutation.isPending || vizMutation.isPending || clusteringMutation.isPending || autoTuneMutation.isPending

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

  const spaceButtons = (
    <div className="flex rounded-lg overflow-hidden border border-white/10">
      <button
        onClick={() => handleSpaceChange('uni')}
        className={`px-4 py-2 text-sm transition-colors ${
          space === 'uni'
            ? 'bg-accent-cyan/20 text-accent-cyan'
            : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
        }`}
      >Univariate</button>
      <button
        onClick={() => handleSpaceChange('multi')}
        className={`px-4 py-2 text-sm transition-colors ${
          space === 'multi'
            ? 'bg-accent-cyan/20 text-accent-cyan'
            : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
        }`}
      >Multivariate</button>
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
  if (isLoading || isCacheLoading) {
    return (
      <div className="flex flex-col h-full gap-3 p-4 overflow-hidden">
        <div className="flex items-center gap-4 shrink-0">
          {domainButtons}
          {spaceButtons}
          {tabButtons}
        </div>

        {activeTab === 'profiling' ? (
          <ClusterProfiling domain={domain} space={space} hideUnclassified={hideUnclassified} />
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
          {spaceButtons}
          {tabButtons}
        </div>

        {activeTab === 'profiling' ? (
          <ClusterProfiling domain={domain} space={space} hideUnclassified={hideUnclassified} />
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
      {/* Top bar: domain switch + tab toggle + stats + clustering method */}
      <div className="flex items-center gap-4 shrink-0">
        {domainButtons}
        {spaceButtons}
        {tabButtons}

        <span className="text-text-muted text-sm">
          {hasActiveFilters
            ? `${highlightedCount} / ${totalCount} stations`
            : `${totalCount} stations`}
        </span>

        {/* HDBSCAN / KMeans toggle + ElbowChart + PCA badge */}
        {cachedData && activeTab === 'scatter' && (
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg overflow-hidden border border-white/10">
              <button
                onClick={() => setCachedMethod('hdbscan')}
                className={`px-3 py-1.5 text-xs transition-colors ${
                  cachedMethod === 'hdbscan'
                    ? 'bg-accent-cyan/20 text-accent-cyan'
                    : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
                }`}
              >
                HDBSCAN {clusteringDiagnostics ? `(${clusteringDiagnostics.n_clusters})` : ''}
              </button>
              <button
                onClick={() => setCachedMethod('kmeans')}
                className={`px-3 py-1.5 text-xs transition-colors ${
                  cachedMethod === 'kmeans'
                    ? 'bg-accent-cyan/20 text-accent-cyan'
                    : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
                }`}
              >
                KMeans {cachedKmeansK ? `(k=${cachedKmeansK})` : ''}
              </button>
            </div>
            {cachedMethod === 'kmeans' && cachedElbow && (
              <ElbowChart elbow={cachedElbow} selectedK={cachedKmeansK} onSelectK={setCachedKmeansK} />
            )}
            {/* PCA diagnostics badge */}
            {(cachedData as unknown as CacheData)?.pca && (
              <span className="text-text-muted text-[10px] border-l border-white/10 pl-2">
                PCA {(cachedData as unknown as CacheData).pca.n_components}D
                ({((cachedData as unknown as CacheData).pca.variance_explained * 100).toFixed(0)}% var)
              </span>
            )}
          </div>
        )}

        {(pcaMutation.isError || vizMutation.isError || clusteringMutation.isError) && (
          <span className="text-accent-red text-xs">
            Pipeline error: {
              (pcaMutation.error as Error)?.message ??
              (vizMutation.error as Error)?.message ??
              (clusteringMutation.error as Error)?.message ??
              'unknown error'
            }
          </span>
        )}
      </div>

      {/* Main content: filter sidebar + (scatter or profiling) + detail */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Filter sidebar -- always visible */}
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
            onlyActive={onlyActive}
            onOnlyActiveChange={setOnlyActive}
          />
        </div>

        {activeTab === 'profiling' ? (
          <div className="flex-1 min-w-0 overflow-y-auto">
            <ClusterProfiling domain={domain} space={space} hideUnclassified={hideUnclassified} />
          </div>
        ) : (
          <>
            {/* Scatter + controls */}
            <div className="flex-1 flex flex-col min-w-0 gap-2">
              {/* Empty state */}
              {scatterPoints.length === 0 && !anyMutationPending ? (
                <div className="flex-1 flex items-center justify-center">
                  <div className="flex flex-col items-center gap-3">
                    <p className="text-text-muted text-sm">
                      {stations.length > 0
                        ? 'No cached projection found. Run the pipeline to compute.'
                        : 'No station embeddings found.'}
                    </p>
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
                    domain={domain}
                    highlightedSite={highlightedSite}
                    onPointClick={handleStationSelect}
                    onDeselect={handleDeselectStation}
                    loading={anyMutationPending}
                    className="h-full"
                  />
                </div>
              )}

              {/* Cluster legend */}
              {scatterPoints.length > 0 && (
                <div className="shrink-0">
                  <ClusterLegendBar
                    clusters={clusterInfo}
                    selectedCluster={legendCluster}
                    onSelectCluster={(id) => {
                      setLegendCluster(id)
                      if (id !== null) {
                        setFilters({ cluster_id: id })
                      } else {
                        const next = { ...filters }
                        delete next.cluster_id
                        setFilters(next)
                      }
                    }}
                  />
                </div>
              )}

              {/* Controls bar -- fixed height, no grow */}
              <div className="shrink-0 bg-bg-card rounded-xl border border-white/5 px-3 py-1 overflow-x-auto">
                <PipelineControls
                  pcaDims={(cachedData as CacheData | undefined)?.pca?.n_components ?? null}
                  pcaVariance={(cachedData as CacheData | undefined)?.pca?.variance_explained ?? null}
                  varianceThreshold={varianceThreshold}
                  onVarianceThresholdChange={setVarianceThreshold}
                  onRecomputePca={handleRecomputePca}
                  isRecomputingPca={pcaMutation.isPending}
                  vizNNeighbors={vizNNeighbors}
                  vizMinDist={vizMinDist}
                  onVizNNeighborsChange={setVizNNeighbors}
                  onVizMinDistChange={setVizMinDist}
                  onRecomputeViz={handleRecomputeViz}
                  isRecomputingViz={vizMutation.isPending}
                  trustworthiness={(cachedData as CacheData | undefined)?.viz?.trustworthiness ?? null}
                  hdbscanMcs={hdbscanMcs}
                  hdbscanMs={hdbscanMs}
                  onHdbscanMcsChange={setHdbscanMcs}
                  onHdbscanMsChange={setHdbscanMs}
                  onRecomputeClustering={handleRecomputeClustering}
                  isRecomputingClustering={clusteringMutation.isPending}
                  clusteringDiagnostics={clusteringDiagnostics}
                  onAutoTune={handleAutoTune}
                  isAutoTuning={autoTuneMutation.isPending}
                  mode={mode}
                  onModeChange={setMode}
                  level={level}
                  onLevelChange={setLevel}
                  yearRange={yearRange}
                  onYearRangeChange={setYearRange}
                  season={season}
                  onSeasonChange={setSeason}
                />
              </div>
            </div>

            {/* Right sidebar: station detail only */}
            {selectedStation && (
              <div className="shrink-0 w-72 overflow-y-auto">
                <StationDetail
                  domain={domain}
                  space={space}
                  stationId={selectedStation}
                  stationMeta={selectedStationMeta}
                  clusterLabel={scatterPoints.find(p => p.id === selectedStation)?.cluster_label ?? null}
                  onClose={handleDeselectStation}
                  onNeighborClick={handleStationSelect}
                  onFilterBySite={handleFilterBySite}
                />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
