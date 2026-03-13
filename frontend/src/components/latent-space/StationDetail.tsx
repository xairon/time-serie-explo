import { X } from 'lucide-react'
import { useSimilarStations } from '@/hooks/useLatentSpace'

interface StationDetailProps {
  domain: 'piezo' | 'hydro'
  stationId: string | null
  onClose: () => void
}

function SkeletonLine({ width = 'w-full' }: { width?: string }) {
  return <div className={`h-4 ${width} bg-white/5 rounded animate-pulse`} />
}

function getSecondaryMeta(
  domain: 'piezo' | 'hydro',
  metadata: Record<string, unknown>,
): string {
  if (domain === 'piezo') {
    return [metadata['nappe'], metadata['milieu_eh']].filter(Boolean).join(' · ')
  }
  return String(metadata['nom_cours_eau'] ?? '')
}

export function StationDetail({ domain, stationId, onClose }: StationDetailProps) {
  const { data, isLoading } = useSimilarStations(domain, stationId)

  if (!stationId) return null

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4 w-72 flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-text-muted text-xs mb-0.5">Station sélectionnée</p>
          <p className="text-text-primary text-sm font-medium leading-snug break-all">
            {stationId}
          </p>
        </div>
        <button
          onClick={onClose}
          className="shrink-0 p-1 text-text-muted hover:text-text-primary hover:bg-bg-hover rounded transition-colors"
          aria-label="Fermer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="border-t border-white/5" />

      {/* Neighbors list */}
      <div className="flex flex-col gap-1">
        <p className="text-text-muted text-xs font-medium uppercase tracking-wide mb-1">
          Stations similaires
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
        ) : !data || !(data as Record<string, unknown>).neighbors || ((data as Record<string, unknown>).neighbors as unknown[]).length === 0 ? (
          <p className="text-text-muted text-sm">Aucune station similaire trouvée.</p>
        ) : (
          <div className="flex flex-col divide-y divide-white/5">
            {((data as Record<string, unknown>).neighbors as Array<{ id: string; distance: number; metadata: Record<string, unknown> }>).map(
              (neighbor, idx) => {
                const secondary = getSecondaryMeta(domain, neighbor.metadata ?? {})
                return (
                  <div key={neighbor.id} className="py-2 flex flex-col gap-0.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-text-primary text-xs font-medium truncate">
                        {idx + 1}. {neighbor.id}
                      </span>
                      <span className="shrink-0 text-accent-cyan text-xs font-mono">
                        {neighbor.distance.toFixed(4)}
                      </span>
                    </div>
                    {secondary && (
                      <span className="text-text-muted text-xs truncate">{secondary}</span>
                    )}
                  </div>
                )
              },
            )}
          </div>
        )}
      </div>
    </div>
  )
}
