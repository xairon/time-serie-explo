import { useState } from 'react'

interface IPSClassification {
  month: string
  level: 'très bas' | 'bas' | 'modérément bas' | 'autour de la moyenne' | 'modérément haut' | 'haut' | 'très haut'
}

interface IPSPanelProps {
  classifications?: IPSClassification[]
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

export function IPSPanel({ classifications, className = '' }: IPSPanelProps) {
  const [window, setWindow] = useState<(typeof WINDOWS)[number]>(3)

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

      {/* Classification table */}
      {classifications && classifications.length > 0 ? (
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
        <p className="text-xs text-text-secondary italic">
          Aucune classification IPS disponible. Lancez une analyse contrefactuelle pour obtenir les
          résultats.
        </p>
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
