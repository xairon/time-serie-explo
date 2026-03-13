import { Loader2 } from 'lucide-react'

interface ClusteringParams {
  method: 'hdbscan' | 'kmeans'
  min_cluster_size: number
  min_samples: number
  n_clusters: number
  n_umap_dims: number
}

interface UMAPControlsProps {
  mode: '2d' | '3d'
  onModeChange: (mode: '2d' | '3d') => void
  level: 'stations' | 'windows'
  onLevelChange: (level: 'stations' | 'windows') => void
  umapParams: { n_neighbors: number; min_dist: number }
  onUmapParamsChange: (params: { n_neighbors: number; min_dist: number }) => void
  clusteringParams: ClusteringParams
  onClusteringParamsChange: (params: ClusteringParams) => void
  onRecalculate: () => void
  onReset: () => void
  isComputing: boolean
  yearRange?: [number, number]
  onYearRangeChange?: (range: [number, number]) => void
  season?: string | null
  onSeasonChange?: (season: string | null) => void
}

const SEASONS = [
  { key: 'DJF', label: 'DJF' },
  { key: 'MAM', label: 'MAM' },
  { key: 'JJA', label: 'JJA' },
  { key: 'SON', label: 'SON' },
]

const inputClass =
  'bg-bg-input border border-white/10 rounded-lg px-2 py-1 text-sm w-20 text-text-primary focus:outline-none focus:border-accent-cyan/50 transition-colors'

const selectClass =
  'bg-bg-input border border-white/10 rounded-lg px-2 py-1 text-sm text-text-primary focus:outline-none focus:border-accent-cyan/50 transition-colors'

function ToggleGroup<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { key: T; label: string }[]
  value: T
  onChange: (v: T) => void
}) {
  return (
    <div className="flex rounded-lg overflow-hidden border border-white/10">
      {options.map(({ key, label }) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          className={`px-3 py-1.5 text-sm transition-colors ${
            value === key
              ? 'bg-accent-cyan/20 text-accent-cyan border-accent-cyan'
              : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

export function UMAPControls({
  mode,
  onModeChange,
  level,
  onLevelChange,
  umapParams,
  onUmapParamsChange,
  clusteringParams,
  onClusteringParamsChange,
  onRecalculate,
  onReset,
  isComputing,
  yearRange,
  onYearRangeChange,
  season,
  onSeasonChange,
}: UMAPControlsProps) {
  return (
    <div className="flex flex-wrap items-end gap-4 px-1 py-2">
      {/* Dimension toggle */}
      <div className="flex flex-col gap-1">
        <span className="text-text-muted text-xs">Dimensions</span>
        <ToggleGroup
          options={[
            { key: '2d', label: '2D' },
            { key: '3d', label: '3D' },
          ]}
          value={mode}
          onChange={onModeChange}
        />
      </div>

      {/* Level toggle */}
      <div className="flex flex-col gap-1">
        <span className="text-text-muted text-xs">Level</span>
        <ToggleGroup
          options={[
            { key: 'stations', label: 'Stations' },
            { key: 'windows', label: 'Windows' },
          ]}
          value={level}
          onChange={onLevelChange}
        />
      </div>

      {/* UMAP params */}
      <div className="flex flex-col gap-1">
        <span className="text-text-muted text-xs">n_neighbors</span>
        <input
          type="number"
          className={inputClass}
          min={2}
          max={200}
          value={umapParams.n_neighbors}
          onChange={(e) =>
            onUmapParamsChange({ ...umapParams, n_neighbors: Number(e.target.value) })
          }
        />
      </div>

      <div className="flex flex-col gap-1">
        <span className="text-text-muted text-xs">min_dist</span>
        <input
          type="number"
          className={inputClass}
          min={0}
          max={1}
          step={0.05}
          value={umapParams.min_dist}
          onChange={(e) =>
            onUmapParamsChange({ ...umapParams, min_dist: Number(e.target.value) })
          }
        />
      </div>

      {/* Clustering */}
      <div className="flex flex-col gap-1">
        <span className="text-text-muted text-xs">Clustering</span>
        <select
          className={selectClass}
          value={clusteringParams.method}
          onChange={(e) =>
            onClusteringParamsChange({
              ...clusteringParams,
              method: e.target.value as 'hdbscan' | 'kmeans',
            })
          }
        >
          <option value="hdbscan">HDBSCAN</option>
          <option value="kmeans">K-Means</option>
        </select>
      </div>

      {clusteringParams.method === 'hdbscan' ? (
        <>
          <div className="flex flex-col gap-1">
            <span className="text-text-muted text-xs">min_cluster_size</span>
            <input
              type="number"
              className={inputClass}
              min={2}
              value={clusteringParams.min_cluster_size}
              onChange={(e) =>
                onClusteringParamsChange({
                  ...clusteringParams,
                  min_cluster_size: Number(e.target.value),
                })
              }
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-text-muted text-xs">min_samples</span>
            <input
              type="number"
              className={inputClass}
              min={1}
              value={clusteringParams.min_samples}
              onChange={(e) =>
                onClusteringParamsChange({
                  ...clusteringParams,
                  min_samples: Number(e.target.value),
                })
              }
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-text-muted text-xs">n_umap_dims</span>
            <input
              type="number"
              className={inputClass}
              min={2}
              max={10}
              value={clusteringParams.n_umap_dims}
              onChange={(e) =>
                onClusteringParamsChange({
                  ...clusteringParams,
                  n_umap_dims: Number(e.target.value),
                })
              }
            />
          </div>
        </>
      ) : (
        <div className="flex flex-col gap-1">
          <span className="text-text-muted text-xs">n_clusters</span>
          <input
            type="number"
            className={inputClass}
            min={2}
            value={clusteringParams.n_clusters}
            onChange={(e) =>
              onClusteringParamsChange({
                ...clusteringParams,
                n_clusters: Number(e.target.value),
              })
            }
          />
        </div>
      )}

      {/* Windows-only: year range + season */}
      {level === 'windows' && onYearRangeChange && yearRange && (
        <>
          <div className="flex flex-col gap-1">
            <span className="text-text-muted text-xs">Start year</span>
            <input
              type="number"
              className={inputClass}
              value={yearRange[0]}
              onChange={(e) => onYearRangeChange([Number(e.target.value), yearRange[1]])}
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-text-muted text-xs">End year</span>
            <input
              type="number"
              className={inputClass}
              value={yearRange[1]}
              onChange={(e) => onYearRangeChange([yearRange[0], Number(e.target.value)])}
            />
          </div>
        </>
      )}

      {level === 'windows' && onSeasonChange && (
        <div className="flex flex-col gap-1">
          <span className="text-text-muted text-xs">Season</span>
          <div className="flex rounded-lg overflow-hidden border border-white/10">
            <button
              onClick={() => onSeasonChange(null)}
              className={`px-2 py-1.5 text-xs transition-colors ${
                season === null
                  ? 'bg-accent-cyan/20 text-accent-cyan'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
              }`}
            >
              All
            </button>
            {SEASONS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => onSeasonChange(season === key ? null : key)}
                className={`px-2 py-1.5 text-xs transition-colors ${
                  season === key
                    ? 'bg-accent-cyan/20 text-accent-cyan'
                    : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-end gap-2 ml-auto">
        <button
          onClick={onReset}
          className="px-3 py-2 text-sm text-text-secondary hover:text-text-primary border border-white/10 rounded-lg hover:bg-bg-hover transition-colors"
        >
          Reset
        </button>
        <button
          onClick={onRecalculate}
          disabled={isComputing}
          className="flex items-center gap-2 bg-accent-cyan text-white px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-60 disabled:cursor-not-allowed hover:bg-accent-cyan/90 transition-colors"
        >
          {isComputing ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Computing...
            </>
          ) : (
            'Recompute'
          )}
        </button>
      </div>
    </div>
  )
}
