import { METEO_CLASS_LABELS, meteoClassColor } from '@/lib/meteo-colors'

interface Props {
  code: string
  commune?: string
  classification?: string | null
  /** Station type — drives the fallback label when classification is missing. */
  type?: 'piezo' | 'hydro'
  derniereMesure?: string | null
  onClose: () => void
}

const FR_MONTHS_LONG = [
  'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
  'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre',
]

/** Format "YYYY-MM" → "mai 2026" */
function formatMonthFR(value: string | null | undefined): string | null {
  if (!value) return null
  const parts = value.split('-')
  if (parts.length < 2) return value
  const year = parts[0]
  const monthIndex = parseInt(parts[1], 10) - 1
  if (monthIndex < 0 || monthIndex > 11) return value
  return `${FR_MONTHS_LONG[monthIndex]} ${year}`
}

function capitalize(s: string): string {
  if (!s) return s
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function StationPopup({ code, commune, classification, type, derniereMesure, onClose }: Props) {
  const classKey = classification ?? 'UNKNOWN'
  const classColor = meteoClassColor(classification as Parameters<typeof meteoClassColor>[0])
  const isUnknown = !classification || classification === 'UNKNOWN'
  // The UNKNOWN label ("Sans nappe libre étendue …") is groundwater-specific and
  // reads as nonsense on a surface-water flow station — use a neutral fallback there.
  const classLabel = isUnknown && type === 'hydro'
    ? 'Non classé / Donnée indisponible'
    : capitalize(METEO_CLASS_LABELS[classKey] ?? METEO_CLASS_LABELS.UNKNOWN)
  const mesureFormatted = formatMonthFR(derniereMesure)

  return (
    <div className="bg-white rounded-lg shadow-lg border border-slate-200 w-56 text-slate-800">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 px-3 pt-3 pb-2 border-b border-slate-100">
        <p className="font-mono text-sm font-bold text-slate-800 tracking-tight">{code}</p>
        <button
          onClick={onClose}
          aria-label="Fermer"
          className="flex-shrink-0 p-0.5 rounded hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
      </div>

      {/* Body */}
      <div className="px-3 py-2.5 space-y-1.5">

        {/* Commune */}
        {commune && (
          <p className="text-xs text-slate-600">{commune}</p>
        )}

        {/* Class badge */}
        <div className="flex items-center gap-2">
          <span
            className="w-2.5 h-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: classColor }}
            aria-hidden="true"
          />
          <span className="text-xs text-slate-700">{classLabel}</span>
        </div>

        {/* Dernière mesure */}
        {mesureFormatted && (
          <p className="text-xs text-slate-500">
            Dernière mesure :{' '}
            <span className="text-slate-700 font-medium">{mesureFormatted}</span>
          </p>
        )}

      </div>
    </div>
  )
}
