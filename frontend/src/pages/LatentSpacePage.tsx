import { useState, useMemo, useCallback } from 'react'
import { AlertTriangle } from 'lucide-react'
import { useStationEmbeddings, useComputeUMAP } from '@/hooks/useLatentSpace'
import { EmbeddingScatter } from '@/components/latent-space/EmbeddingScatter'
import { FilterPanel } from '@/components/latent-space/FilterPanel'
import { UMAPControls } from '@/components/latent-space/UMAPControls'
import { StationDetail } from '@/components/latent-space/StationDetail'

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
  const [umapParams, setUmapParams] = useState({ n_neighbors: 15, min_dist: 0.1 })
  const [clusteringParams, setClusteringParams] = useState({
    method: 'hdbscan' as const,
    min_cluster_size: 10,
    min_samples: 5,
    n_clusters: 8,
    n_umap_dims: 10,
  })
  const [selectedStation, setSelectedStation] = useState<string | null>(null)
  const [yearRange, setYearRange] = useState<[number, number]>([2015, 2025])
  const [season, setSeason] = useState<string | null>(null)

  // Computed points from /compute endpoint (overrides pre-computed)
  const [computedPoints, setComputedPoints] = useState<ComputedPointRaw[] | null>(null)
  const [subsampled, setSubsampled] = useState<{ from: number } | null>(null)

  // Data fetching
  const { data: stationsData, isLoading, isError, refetch } = useStationEmbeddings(domain)
  const computeMutation = useComputeUMAP()

  // Extract raw stations from API response
  const stations = useMemo(() => {
    if (!stationsData) return []
    return (stationsData.stations ?? []) as StationRaw[]
  }, [stationsData])

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
        ...(filters.region ? { region: filters.region } : {}),
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
      }
      setComputedPoints(result.points)
      setSubsampled(result.subsampled && result.subsampled_from ? { from: result.subsampled_from } : null)
    } catch {
      // Error handled by mutation state
    }
  }

  // Handle reset
  function handleReset() {
    setComputedPoints(null)
    setSubsampled(null)
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin" />
          <span className="text-text-secondary text-sm">Chargement des embeddings...</span>
        </div>
      </div>
    )
  }

  // Error state
  if (isError) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="bg-bg-card rounded-xl border border-white/5 p-8 flex flex-col items-center gap-4 max-w-md">
          <AlertTriangle className="w-10 h-10 text-accent-red" />
          <p className="text-text-primary text-center">Base de données BRGM indisponible</p>
          <p className="text-text-muted text-sm text-center">
            Impossible de charger les embeddings. Vérifiez la connexion à brgm-postgres.
          </p>
          <button
            onClick={() => refetch()}
            className="bg-accent-cyan text-white px-4 py-2 rounded-lg text-sm hover:bg-accent-cyan/90 transition-colors"
          >
            Réessayer
          </button>
        </div>
      </div>
    )
  }

  const highlightedCount = scatterPoints.filter((p) => p.highlighted).length
  const totalCount = scatterPoints.length

  return (
    <div className="flex flex-col h-full gap-3 p-4 overflow-hidden">
      {/* Top bar: domain switch + stats */}
      <div className="flex items-center gap-4 shrink-0">
        <div className="flex rounded-lg overflow-hidden border border-white/10">
          <button
            onClick={() => handleDomainChange('piezo')}
            className={`px-4 py-2 text-sm transition-colors ${
              domain === 'piezo'
                ? 'bg-accent-cyan/20 text-accent-cyan'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
            }`}
          >
            Piézométrie
          </button>
          <button
            onClick={() => handleDomainChange('hydro')}
            className={`px-4 py-2 text-sm transition-colors ${
              domain === 'hydro'
                ? 'bg-accent-cyan/20 text-accent-cyan'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
            }`}
          >
            Hydrométrie
          </button>
        </div>

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
              (sous-échantillonné)
            </span>
          </div>
        )}

        {computeMutation.isError && (
          <span className="text-accent-red text-xs">
            Erreur UMAP: {(computeMutation.error as Error)?.message ?? 'erreur inconnue'}
          </span>
        )}
      </div>

      {/* Main content: filter sidebar + scatter + detail */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Filter sidebar */}
        <div className="shrink-0 overflow-y-auto">
          <FilterPanel
            domain={domain}
            stations={stationsForFilter}
            filters={filters}
            onFiltersChange={setFilters}
            colorBy={colorBy}
            onColorByChange={setColorBy}
          />
        </div>

        {/* Scatter + controls */}
        <div className="flex-1 flex flex-col min-w-0 gap-2">
          {/* Empty state */}
          {scatterPoints.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-text-muted text-sm">
                Aucune station avec des coordonnées UMAP pré-calculées.
              </p>
            </div>
          ) : highlightedCount === 0 && hasActiveFilters ? (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-text-muted text-sm">
                Aucune station ne correspond aux filtres sélectionnés.
              </p>
            </div>
          ) : (
            <div className="flex-1 min-h-0">
              <EmbeddingScatter
                points={scatterPoints}
                mode={mode}
                colorBy={colorBy}
                onPointClick={setSelectedStation}
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

        {/* Station detail panel */}
        {selectedStation && (
          <div className="shrink-0">
            <StationDetail
              domain={domain}
              stationId={selectedStation}
              onClose={() => setSelectedStation(null)}
            />
          </div>
        )}
      </div>
    </div>
  )
}
