import { Loader2 } from 'lucide-react'

const SEASONS = [
  { key: 'DJF', label: 'DJF' },
  { key: 'MAM', label: 'MAM' },
  { key: 'JJA', label: 'JJA' },
  { key: 'SON', label: 'SON' },
]

const inputClass =
  'bg-bg-input border border-white/10 rounded px-2 py-1 text-xs w-16 text-text-primary focus:outline-none focus:border-accent-cyan/50 transition-colors'

interface PipelineControlsProps {
  // PCA
  pcaDims: number | null
  pcaVariance: number | null
  varianceThreshold: number
  onVarianceThresholdChange: (v: number) => void
  onRecomputePca: () => void
  isRecomputingPca: boolean
  // Viz
  vizNNeighbors: number
  vizMinDist: number
  onVizNNeighborsChange: (v: number) => void
  onVizMinDistChange: (v: number) => void
  onRecomputeViz: () => void
  isRecomputingViz: boolean
  trustworthiness: number | null
  // Clustering
  hdbscanMcs: number
  hdbscanMs: number
  onHdbscanMcsChange: (v: number) => void
  onHdbscanMsChange: (v: number) => void
  onRecomputeClustering: () => void
  isRecomputingClustering: boolean
  clusteringDiagnostics: { dbcv?: number; noise_ratio?: number; n_clusters?: number } | null
  // Global
  onAutoTune: () => void
  isAutoTuning: boolean
  // Mode toggles
  mode: '2d' | '3d'
  onModeChange: (m: '2d' | '3d') => void
  level: 'stations' | 'windows'
  onLevelChange: (l: 'stations' | 'windows') => void
  // Windows params (only shown when level='windows')
  yearRange?: [number, number]
  onYearRangeChange?: (range: [number, number]) => void
  season?: string | null
  onSeasonChange?: (season: string | null) => void
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <span className="text-text-muted text-[10px] font-medium uppercase tracking-wide">{children}</span>
}

function Param({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-text-muted text-[10px]">{label}</span>
      {children}
    </div>
  )
}

function Diagnostic({ children }: { children: React.ReactNode }) {
  return <span className="text-text-muted text-[10px]">{children}</span>
}

function RecomputeButton({ onClick, isLoading, label }: { onClick: () => void; isLoading: boolean; label?: string }) {
  return (
    <button
      onClick={onClick}
      disabled={isLoading}
      className="flex items-center gap-1 px-2 py-1 text-xs text-text-secondary hover:text-text-primary border border-white/10 rounded hover:bg-bg-hover transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
    >
      {isLoading ? (
        <><Loader2 className="w-3 h-3 animate-spin" />{label ?? 'Recompute'}</>
      ) : (
        label ?? 'Recompute'
      )}
    </button>
  )
}

function Divider() {
  return <div className="w-px h-6 bg-white/10 shrink-0" />
}

export function PipelineControls({
  pcaDims,
  pcaVariance,
  varianceThreshold,
  onVarianceThresholdChange,
  onRecomputePca,
  isRecomputingPca,
  vizNNeighbors,
  vizMinDist,
  onVizNNeighborsChange,
  onVizMinDistChange,
  onRecomputeViz,
  isRecomputingViz,
  trustworthiness,
  hdbscanMcs,
  hdbscanMs,
  onHdbscanMcsChange,
  onHdbscanMsChange,
  onRecomputeClustering,
  isRecomputingClustering,
  clusteringDiagnostics,
  onAutoTune,
  isAutoTuning,
  mode,
  onModeChange,
  level,
  onLevelChange,
  yearRange,
  onYearRangeChange,
  season,
  onSeasonChange,
}: PipelineControlsProps) {
  const anyLoading = isRecomputingPca || isRecomputingViz || isRecomputingClustering || isAutoTuning

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

      <Divider />

      {/* PCA Section */}
      <div className="flex flex-col gap-1 shrink-0">
        <SectionTitle>PCA</SectionTitle>
        <div className="flex items-end gap-2">
          <Param label="Variance %">
            <input
              type="number"
              className={inputClass}
              min={0.5}
              max={1}
              step={0.01}
              value={varianceThreshold}
              onChange={(e) => onVarianceThresholdChange(Number(e.target.value))}
            />
          </Param>
          <RecomputeButton onClick={onRecomputePca} isLoading={isRecomputingPca} />
        </div>
        <div className="flex items-center gap-2">
          {pcaDims != null && <Diagnostic>{pcaDims} dims</Diagnostic>}
          {pcaVariance != null && <Diagnostic>{(pcaVariance * 100).toFixed(1)}% var</Diagnostic>}
        </div>
      </div>

      <Divider />

      {/* Visualization Section */}
      <div className="flex flex-col gap-1 shrink-0">
        <SectionTitle>Visualization</SectionTitle>
        <div className="flex items-end gap-2">
          <Param label="n_neighbors">
            <input
              type="number"
              className={inputClass}
              min={2}
              max={200}
              value={vizNNeighbors}
              onChange={(e) => onVizNNeighborsChange(Number(e.target.value))}
            />
          </Param>
          <Param label="min_dist">
            <input
              type="number"
              className={inputClass}
              min={0}
              max={1}
              step={0.05}
              value={vizMinDist}
              onChange={(e) => onVizMinDistChange(Number(e.target.value))}
            />
          </Param>
          <RecomputeButton onClick={onRecomputeViz} isLoading={isRecomputingViz} />
        </div>
        <div className="flex items-center gap-2">
          {trustworthiness != null && (
            <Diagnostic>
              trust: <span className={trustworthiness > 0.9 ? 'text-green-400' : trustworthiness > 0.8 ? 'text-amber-400' : 'text-red-400'}>
                {trustworthiness.toFixed(3)}
              </span>
            </Diagnostic>
          )}
        </div>
      </div>

      <Divider />

      {/* Clustering Section */}
      <div className="flex flex-col gap-1 shrink-0">
        <SectionTitle>Clustering</SectionTitle>
        <div className="flex items-end gap-2">
          <Param label="min_cluster_size">
            <input
              type="number"
              className={inputClass}
              min={2}
              value={hdbscanMcs}
              onChange={(e) => onHdbscanMcsChange(Number(e.target.value))}
            />
          </Param>
          <Param label="min_samples">
            <input
              type="number"
              className={inputClass}
              min={1}
              value={hdbscanMs}
              onChange={(e) => onHdbscanMsChange(Number(e.target.value))}
            />
          </Param>
          <RecomputeButton onClick={onRecomputeClustering} isLoading={isRecomputingClustering} />
        </div>
        <div className="flex items-center gap-2">
          {clusteringDiagnostics?.n_clusters != null && (
            <Diagnostic>{clusteringDiagnostics.n_clusters} clusters</Diagnostic>
          )}
          {clusteringDiagnostics?.dbcv != null && (
            <Diagnostic>
              DBCV: <span className={clusteringDiagnostics.dbcv > 0.3 ? 'text-green-400' : clusteringDiagnostics.dbcv > 0.1 ? 'text-amber-400' : 'text-red-400'}>
                {clusteringDiagnostics.dbcv.toFixed(3)}
              </span>
            </Diagnostic>
          )}
          {clusteringDiagnostics?.noise_ratio != null && (
            <Diagnostic>noise: {(clusteringDiagnostics.noise_ratio * 100).toFixed(0)}%</Diagnostic>
          )}
        </div>
      </div>

      {/* Windows-only params */}
      {level === 'windows' && onYearRangeChange && yearRange && (
        <>
          <Divider />
          <div className="flex flex-col gap-1 shrink-0">
            <SectionTitle>Windows</SectionTitle>
            <div className="flex items-end gap-2">
              <Param label="From">
                <input
                  type="number"
                  className={inputClass}
                  value={yearRange[0]}
                  onChange={(e) => onYearRangeChange([Number(e.target.value), yearRange[1]])}
                />
              </Param>
              <Param label="To">
                <input
                  type="number"
                  className={inputClass}
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
                      <button
                        key={key}
                        onClick={() => onSeasonChange(season === key ? null : key)}
                        className={`px-1.5 py-1 text-[10px] transition-colors ${
                          season === key ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:bg-bg-hover'
                        }`}
                      >{label}</button>
                    ))}
                  </div>
                </Param>
              )}
            </div>
          </div>
        </>
      )}

      {/* Auto-tune button */}
      <div className="ml-auto shrink-0">
        <button
          onClick={onAutoTune}
          disabled={anyLoading}
          className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-amber-400 border border-amber-400/30 rounded hover:bg-amber-400/10 disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
        >
          {isAutoTuning ? (
            <><Loader2 className="w-3.5 h-3.5 animate-spin" />Tuning...</>
          ) : 'Auto-tune'}
        </button>
      </div>
    </div>
  )
}
