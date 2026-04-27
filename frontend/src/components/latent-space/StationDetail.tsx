import { X } from 'lucide-react'
import { useSimilarStations } from '@/hooks/useLatentSpace'
import { WindowScatter } from './WindowScatter'

interface StationDetailProps {
  domain: 'piezo' | 'hydro'
  space: string
  stationId: string | null
  stationMeta?: Record<string, unknown>
  clusterLabel?: number | null
  onClose: () => void
  onNeighborClick?: (stationId: string) => void
  onFilterBySite?: () => void
}

function m(v: unknown): string | number | null { return v as string | number | null }

function MetaLine({ label, value }: { label: string; value: string | number | null | undefined }) {
  if (value === null || value === undefined || value === '') return null
  return (
    <div className="flex justify-between gap-2 text-xs">
      <span className="text-text-muted shrink-0">{label}</span>
      <span className="text-text-primary text-right truncate">{String(value)}</span>
    </div>
  )
}

export function StationDetail({ domain, space, stationId, stationMeta, clusterLabel, onClose, onNeighborClick, onFilterBySite }: StationDetailProps) {
  const { data, isLoading } = useSimilarStations(domain, stationId, space)

  if (!stationId) return null

  const neighbors = (data as Record<string, unknown> | undefined)?.neighbors as
    | Array<{ id: string; distance: number; cluster_id: number | null }>
    | undefined

  const siteLabel = domain === 'piezo' ? 'aquifer' : 'waterway'
  const siteValue = (domain === 'piezo' ? stationMeta?.libelle_eh : stationMeta?.nom_cours_eau) as string | undefined

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-3 w-full flex flex-col gap-2.5 overflow-y-auto max-h-full">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 shrink-0">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <p className="text-text-primary text-sm font-medium leading-snug break-all">{stationId}</p>
            {clusterLabel != null && clusterLabel >= 0 && (
              <span className="shrink-0 bg-accent-cyan/20 text-accent-cyan text-[10px] px-1.5 py-0.5 rounded">
                C{clusterLabel}
              </span>
            )}
          </div>
        </div>
        <button onClick={onClose}
          className="shrink-0 p-1 text-text-muted hover:text-text-primary hover:bg-bg-hover rounded transition-colors">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Metadata */}
      {stationMeta && (
        <>
          <div className="border-t border-white/5" />
          <div className="flex flex-col gap-0.5 shrink-0">
            {domain === 'piezo' ? (
              <>
                <MetaLine label="Aquifer" value={m(stationMeta.libelle_eh)} />
                <MetaLine label="Department" value={m(stationMeta.departement)} />
                <MetaLine label="Altitude" value={m(stationMeta.altitude)} />
                <MetaLine label="Windows" value={m(stationMeta.n_windows)} />
                <MetaLine label="Days" value={m(stationMeta.n_days)} />
              </>
            ) : (
              <>
                <MetaLine label="Waterway" value={m(stationMeta.nom_cours_eau)} />
                <MetaLine label="Department" value={m(stationMeta.departement)} />
                <MetaLine label="Status" value={m(stationMeta.statut_station)} />
                <MetaLine label="Windows" value={m(stationMeta.n_windows)} />
                <MetaLine label="Days" value={m(stationMeta.n_days)} />
              </>
            )}
          </div>
        </>
      )}

      {/* Filter by site button */}
      {siteValue && onFilterBySite && (
        <button onClick={onFilterBySite}
          className="w-full py-1.5 text-[10px] text-accent-cyan border border-accent-cyan/30 rounded hover:bg-accent-cyan/10 transition-colors">
          Filter by {siteLabel}
        </button>
      )}

      {/* Window scatter */}
      <div className="border-t border-white/5" />
      <div>
        <p className="text-text-muted text-[10px] font-medium uppercase tracking-wide mb-1">Temporal evolution</p>
        <WindowScatter domain={domain} stationId={stationId} space={space} />
      </div>

      {/* Neighbors */}
      <div className="border-t border-white/5" />
      <div className="flex flex-col gap-1 min-h-0">
        <p className="text-text-muted text-[10px] font-medium uppercase tracking-wide mb-0.5 shrink-0">Similar stations</p>
        {isLoading ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-4 w-3/4 bg-white/5 rounded animate-pulse" />
            ))}
          </div>
        ) : !neighbors || neighbors.length === 0 ? (
          <p className="text-text-muted text-xs">No similar stations found.</p>
        ) : (
          <div className="flex flex-col divide-y divide-white/5">
            {neighbors.map((neighbor, idx) => (
              <button key={neighbor.id}
                className="py-1.5 flex items-center justify-between gap-2 text-left hover:bg-bg-hover rounded px-1 -mx-1 transition-colors"
                onClick={() => onNeighborClick?.(neighbor.id)}>
                <span className="text-text-primary text-[10px] font-medium truncate">
                  {idx + 1}. {neighbor.id}
                </span>
                <span className="shrink-0 text-accent-cyan text-[10px] font-mono">
                  {neighbor.distance.toFixed(4)}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
