import { useState } from 'react'
import { useIPSReference } from '@/hooks/useCounterfactual'

interface IPSClassification {
  month: string
  level: string
}

interface IPSPanelProps {
  modelId: string | null
  className?: string
}

const levelColors: Record<string, string> = {
  'très bas': 'bg-red-600',
  'bas': 'bg-red-400',
  'modérément bas': 'bg-orange-400',
  'autour de la moyenne': 'bg-yellow-400',
  'modérément haut': 'bg-green-400',
  'haut': 'bg-blue-400',
  'très haut': 'bg-blue-600',
}

const WINDOWS = [1, 3, 6, 12] as const

/**
 * Classify a z-score (standardized value) into an IPS level.
 */
function classifyZScore(z: number): string {
  if (z <= -2) return 'très bas'
  if (z <= -1) return 'bas'
  if (z <= -0.5) return 'modérément bas'
  if (z <= 0.5) return 'autour de la moyenne'
  if (z <= 1) return 'modérément haut'
  if (z <= 2) return 'haut'
  return 'très haut'
}

/**
 * Transform ref_stats from API into classifications for the table.
 * ref_stats is expected to be a dict keyed by month (or period label) with stat values.
 */
function transformRefStats(
  refStats: Record<string, unknown>,
  mu: number | null,
  sigma: number | null,
): IPSClassification[] {
  if (!refStats || !mu || !sigma || sigma === 0) return []

  return Object.entries(refStats).map(([month, value]) => {
    const numVal = typeof value === 'number' ? value : Number(value)
    const z = (numVal - mu) / sigma
    return {
      month,
      level: classifyZScore(z),
    }
  })
}

export function IPSPanel({ modelId, className = '' }: IPSPanelProps) {
  const [window, setWindow] = useState<(typeof WINDOWS)[number]>(3)
  const { data: ipsData, isLoading, isError } = useIPSReference(modelId, window)

  const classifications = ipsData
    ? transformRefStats(ipsData.ref_stats, ipsData.mu_target, ipsData.sigma_target)
    : []

  return (
    <div className={`space-y-4 ${className}`}>
      <h4 className="text-sm font-semibold text-text-primary">Référence IPS</h4>

      {/* Window selector */}
      <div className="flex gap-1">
        {WINDOWS.map((w) => (
          <button
            key={w}
            onClick={() => setWindow(w)}
            className={`px-2.5 py-1 text-xs rounded-lg transition-colors ${
              window === w
                ? 'bg-accent-cyan/10 text-accent-cyan border border-accent-cyan'
                : 'bg-bg-hover text-text-secondary border border-white/10 hover:text-text-primary'
            }`}
          >
            {w} mois
          </button>
        ))}
      </div>

      {/* Loading state */}
      {isLoading && modelId && (
        <div className="animate-pulse space-y-2">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-6 bg-bg-hover rounded" />
          ))}
        </div>
      )}

      {/* Error state */}
      {isError && modelId && (
        <p className="text-xs text-accent-red italic">
          Erreur lors du chargement des données IPS.
        </p>
      )}

      {/* Info about n_years and validation */}
      {ipsData && (
        <div className="flex gap-3 text-[10px] text-text-secondary">
          {ipsData.n_years != null && <span>{ipsData.n_years} années de référence</span>}
          {ipsData.mu_target != null && <span>mu={ipsData.mu_target.toFixed(3)}</span>}
          {ipsData.sigma_target != null && <span>sigma={ipsData.sigma_target.toFixed(3)}</span>}
        </div>
      )}

      {/* Classification table */}
      {!isLoading && classifications.length > 0 ? (
        <div className="overflow-x-auto rounded-lg border border-white/5">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-bg-hover">
                <th className="px-3 py-2 text-left text-text-secondary font-medium">Mois</th>
                <th className="px-3 py-2 text-left text-text-secondary font-medium">
                  Classification ({window} mois)
                </th>
              </tr>
            </thead>
            <tbody>
              {classifications.map((c) => (
                <tr key={c.month} className="border-t border-white/5">
                  <td className="px-3 py-1.5 text-text-primary">{c.month}</td>
                  <td className="px-3 py-1.5">
                    <span
                      className={`inline-block w-3 h-3 rounded-full mr-2 ${levelColors[c.level] ?? 'bg-gray-400'}`}
                    />
                    <span className="text-text-primary">{c.level}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        !isLoading && (
          <p className="text-xs text-text-secondary italic">
            {modelId
              ? 'Aucune classification IPS disponible pour ce modèle.'
              : 'Sélectionnez un modèle et lancez une analyse contrefactuelle pour obtenir les résultats IPS.'}
          </p>
        )
      )}

      {/* Legend */}
      <div>
        <p className="text-[10px] text-text-secondary uppercase mb-1">Légende</p>
        <div className="flex flex-wrap gap-2">
          {Object.entries(levelColors).map(([level, color]) => (
            <div key={level} className="flex items-center gap-1">
              <span className={`w-2.5 h-2.5 rounded-full ${color}`} />
              <span className="text-[10px] text-text-secondary">{level}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
