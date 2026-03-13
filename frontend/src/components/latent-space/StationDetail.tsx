import { X } from 'lucide-react'
import { useSimilarStations } from '@/hooks/useLatentSpace'

interface StationDetailProps {
  domain: 'piezo' | 'hydro'
  stationId: string | null
  stationMeta?: Record<string, unknown>
  onClose: () => void
  onNeighborClick?: (stationId: string) => void
}

function SkeletonLine({ width = 'w-full' }: { width?: string }) {
  return <div className={`h-4 ${width} bg-white/5 rounded animate-pulse`} />
}

function MetaLine({ label, value }: { label: string; value: unknown }) {
  if (value === null || value === undefined || value === '') return null
  return (
    <div className="flex justify-between gap-2 text-xs">
      <span className="text-text-muted shrink-0">{label}</span>
      <span className="text-text-primary text-right truncate">{String(value)}</span>
    </div>
  )
}

export function StationDetail({ domain, stationId, stationMeta, onClose, onNeighborClick }: StationDetailProps) {
  const { data, isLoading } = useSimilarStations(domain, stationId)

  if (!stationId) return null

  const neighbors = (data as Record<string, unknown> | undefined)?.neighbors as
    | Array<{ id: string; distance: number; cluster_id: number | null }>
    | undefined

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4 w-full flex flex-col gap-3 overflow-hidden">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 shrink-0">
        <div className="min-w-0">
          <p className="text-text-muted text-xs mb-0.5">Selected station</p>
          <p className="text-text-primary text-sm font-medium leading-snug break-all">
            {stationId}
          </p>
        </div>
        <button
          onClick={onClose}
          className="shrink-0 p-1 text-text-muted hover:text-text-primary hover:bg-bg-hover rounded transition-colors"
          aria-label="Close"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Station metadata */}
      {stationMeta && (
        <>
          <div className="border-t border-white/5" />
          <div className="flex flex-col gap-1 shrink-0">
            {domain === 'piezo' ? (
              <>
                <MetaLine label="Aquifer" value={stationMeta.libelle_eh} />
                <MetaLine label="Medium" value={stationMeta.milieu_eh} />
                <MetaLine label="Theme" value={stationMeta.theme_eh} />
                <MetaLine label="State" value={stationMeta.etat_eh} />
                <MetaLine label="Department" value={stationMeta.departement} />
                <MetaLine label="Altitude" value={stationMeta.altitude} />
              </>
            ) : (
              <>
                <MetaLine label="Waterway" value={stationMeta.nom_cours_eau} />
                <MetaLine label="Department" value={stationMeta.departement} />
                <MetaLine label="Status" value={stationMeta.statut_station} />
              </>
            )}
          </div>
        </>
      )}

      <div className="border-t border-white/5" />

      {/* Neighbors list */}
      <div className="flex flex-col gap-1 min-h-0 overflow-y-auto">
        <p className="text-text-muted text-xs font-medium uppercase tracking-wide mb-1 shrink-0">
          Similar stations
        </p>

        {isLoading ? (
          <div className="flex flex-col gap-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex flex-col gap-1.5">
                <SkeletonLine width="w-3/4" />
                <SkeletonLine width="w-1/2" />
              </div>
            ))}
          </div>
        ) : !neighbors || neighbors.length === 0 ? (
          <p className="text-text-muted text-sm">No similar stations found.</p>
        ) : (
          <div className="flex flex-col divide-y divide-white/5">
            {neighbors.map((neighbor, idx) => (
              <button
                key={neighbor.id}
                className="py-2 flex items-center justify-between gap-2 text-left hover:bg-bg-hover rounded px-1 -mx-1 transition-colors"
                onClick={() => onNeighborClick?.(neighbor.id)}
              >
                <span className="text-text-primary text-xs font-medium truncate">
                  {idx + 1}. {neighbor.id}
                </span>
                <span className="shrink-0 text-accent-cyan text-xs font-mono">
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
