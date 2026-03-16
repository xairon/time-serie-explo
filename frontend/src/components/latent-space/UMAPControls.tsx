import { Loader2, ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'

interface UMAPPreReductionParams {
  n_components: number
  n_neighbors: number
  min_dist: number
}

interface ClusteringParams {
  method: 'hdbscan' | 'kmeans'
  min_cluster_size: number
  min_samples: number
  n_clusters: number
  umap_prereduction: UMAPPreReductionParams
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
  qualityMetrics?: Record<string, unknown> | null
}

const SEASONS = [
  { key: 'DJF', label: 'DJF' },
  { key: 'MAM', label: 'MAM' },
  { key: 'JJA', label: 'JJA' },
  { key: 'SON', label: 'SON' },
]

const inputClass =
  'bg-bg-input border border-white/10 rounded px-2 py-1 text-xs w-16 text-text-primary focus:outline-none focus:border-accent-cyan/50 transition-colors'

const selectClass =
  'bg-bg-input border border-white/10 rounded px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-cyan/50 transition-colors'

function Section({ title, children, defaultOpen = false }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-l-2 border-white/10 pl-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs font-medium text-text-secondary hover:text-text-primary transition-colors"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        {title}
      </button>
      {open && <div className="flex flex-wrap items-end gap-3 mt-1.5">{children}</div>}
    </div>
  )
}

function Param({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-text-muted text-[10px]">{label}</span>
      {children}
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
  qualityMetrics,
}: UMAPControlsProps) {
  const pre = clusteringParams.umap_prereduction

  return (
    <div className="flex items-center gap-3 px-1 py-1.5 overflow-x-auto">
      {/* Quick toggles */}
      <div className="flex items-center gap-2 shrink-0">
        <div className="flex rounded overflow-hidden border border-white/10">
          {(['2d', '3d'] as const).map((m) => (
            <button
              key={m}
              onClick={() => onModeChange(m)}
              className={`px-2.5 py-1 text-xs transition-colors ${
                mode === m ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
              }`}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="flex rounded overflow-hidden border border-white/10">
          {(['stations', 'windows'] as const).map((l) => (
            <button
              key={l}
              onClick={() => onLevelChange(l)}
              className={`px-2.5 py-1 text-xs transition-colors ${
                level === l ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
              }`}
            >
              {l === 'stations' ? 'Stations' : 'Windows'}
            </button>
          ))}
        </div>
      </div>

      <div className="w-px h-6 bg-white/10 shrink-0" />

      {/* UMAP Visualization */}
      <Section title="UMAP Viz" defaultOpen>
        <Param label="n_neighbors">
          <input type="number" className={inputClass} min={2} max={200}
            value={umapParams.n_neighbors}
            onChange={(e) => onUmapParamsChange({ ...umapParams, n_neighbors: Number(e.target.value) })}
          />
        </Param>
        <Param label="min_dist">
          <input type="number" className={inputClass} min={0} max={1} step={0.05}
            value={umapParams.min_dist}
            onChange={(e) => onUmapParamsChange({ ...umapParams, min_dist: Number(e.target.value) })}
          />
        </Param>
      </Section>

      <div className="w-px h-6 bg-white/10 shrink-0" />

      {/* UMAP Pre-reduction (HDBSCAN only) */}
      {clusteringParams.method === 'hdbscan' && (
        <>
          <Section title="UMAP Pre-reduction">
            <Param label="n_components">
              <input type="number" className={inputClass} min={2} max={50}
                value={pre.n_components}
                onChange={(e) => onClusteringParamsChange({
                  ...clusteringParams,
                  umap_prereduction: { ...pre, n_components: Number(e.target.value) },
                })}
              />
            </Param>
            <Param label="n_neighbors">
              <input type="number" className={inputClass} min={2} max={100}
                value={pre.n_neighbors}
                onChange={(e) => onClusteringParamsChange({
                  ...clusteringParams,
                  umap_prereduction: { ...pre, n_neighbors: Number(e.target.value) },
                })}
              />
            </Param>
            <Param label="min_dist">
              <input type="number" className={inputClass} min={0} max={1} step={0.05}
                value={pre.min_dist}
                onChange={(e) => onClusteringParamsChange({
                  ...clusteringParams,
                  umap_prereduction: { ...pre, min_dist: Number(e.target.value) },
                })}
              />
            </Param>
          </Section>
          <div className="w-px h-6 bg-white/10 shrink-0" />
        </>
      )}

      {/* Clustering */}
      <Section title="Clustering" defaultOpen>
        <Param label="Method">
          <select className={selectClass}
            value={clusteringParams.method}
            onChange={(e) => onClusteringParamsChange({
              ...clusteringParams,
              method: e.target.value as 'hdbscan' | 'kmeans',
            })}
          >
            <option value="hdbscan">HDBSCAN</option>
            <option value="kmeans">K-Means</option>
          </select>
        </Param>

        {clusteringParams.method === 'hdbscan' ? (
          <>
            <Param label="min_cluster_size">
              <input type="number" className={inputClass} min={2}
                value={clusteringParams.min_cluster_size}
                onChange={(e) => onClusteringParamsChange({
                  ...clusteringParams,
                  min_cluster_size: Number(e.target.value),
                })}
              />
            </Param>
            <Param label="min_samples">
              <input type="number" className={inputClass} min={1}
                value={clusteringParams.min_samples}
                onChange={(e) => onClusteringParamsChange({
                  ...clusteringParams,
                  min_samples: Number(e.target.value),
                })}
              />
            </Param>
          </>
        ) : (
          <Param label="n_clusters">
            <input type="number" className={inputClass} min={2}
              value={clusteringParams.n_clusters}
              onChange={(e) => onClusteringParamsChange({
                ...clusteringParams,
                n_clusters: Number(e.target.value),
              })}
            />
          </Param>
        )}
      </Section>

      {/* Windows-only params */}
      {level === 'windows' && onYearRangeChange && yearRange && (
        <>
          <div className="w-px h-6 bg-white/10 shrink-0" />
          <Section title="Windows" defaultOpen>
            <Param label="From">
              <input type="number" className={inputClass}
                value={yearRange[0]}
                onChange={(e) => onYearRangeChange([Number(e.target.value), yearRange[1]])}
              />
            </Param>
            <Param label="To">
              <input type="number" className={inputClass}
                value={yearRange[1]}
                onChange={(e) => onYearRangeChange([yearRange[0], Number(e.target.value)])}
              />
            </Param>
            {onSeasonChange && (
              <Param label="Season">
                <div className="flex rounded overflow-hidden border border-white/10">
                  <button
                    onClick={() => onSeasonChange(null)}
                    className={`px-1.5 py-1 text-[10px] transition-colors ${
                      season === null ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:bg-bg-hover'
                    }`}
                  >All</button>
                  {SEASONS.map(({ key, label }) => (
                    <button key={key}
                      onClick={() => onSeasonChange(season === key ? null : key)}
                      className={`px-1.5 py-1 text-[10px] transition-colors ${
                        season === key ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:bg-bg-hover'
                      }`}
                    >{label}</button>
                  ))}
                </div>
              </Param>
            )}
          </Section>
        </>
      )}

      {/* Quality metrics (compact, inline after recompute) */}
      {qualityMetrics && (() => {
        const clust = (qualityMetrics as Record<string, Record<string, unknown>>).clustering as Record<string, number> | undefined
        return clust ? (
          <div className="flex items-center gap-2 shrink-0 text-[10px] text-text-muted border-l border-white/10 pl-2">
            {clust.n_clusters != null && <span>{clust.n_clusters} clusters</span>}
            {clust.silhouette != null && (
              <span className={clust.silhouette > 0.3 ? 'text-green-400' : clust.silhouette > 0.1 ? 'text-amber-400' : 'text-red-400'}>
                sil: {clust.silhouette.toFixed(2)}
              </span>
            )}
            {clust.noise_ratio != null && <span>noise: {(clust.noise_ratio * 100).toFixed(0)}%</span>}
          </div>
        ) : null
      })()}

      {/* Actions */}
      <div className="flex items-center gap-2 ml-auto shrink-0">
        <button onClick={onReset}
          className="px-2.5 py-1.5 text-xs text-text-secondary hover:text-text-primary border border-white/10 rounded hover:bg-bg-hover transition-colors"
        >Reset</button>
        <button onClick={onRecalculate} disabled={isComputing}
          className="flex items-center gap-1.5 bg-accent-cyan text-white px-3 py-1.5 rounded text-xs font-medium disabled:opacity-60 disabled:cursor-not-allowed hover:bg-accent-cyan/90 transition-colors"
        >
          {isComputing ? (
            <><Loader2 className="w-3.5 h-3.5 animate-spin" />Computing...</>
          ) : 'Recompute'}
        </button>
      </div>
    </div>
  )
}
