import { METEO_CLASS_COLORS, METEO_CLASS_LABELS, METEO_TREND_LABELS, meteoClassColor } from '@/lib/meteo-colors'
import type { SectorSituation } from '@/lib/observatory-types'

interface Props {
  sector: SectorSituation
  onClose: () => void
}

// Trend arrow SVG (14×14), rotated per trend value
function TrendArrow({ trend }: { trend: string | null }) {
  const rotation = trend === 'hausse' ? 0 : trend === 'stable' ? 90 : trend === 'baisse' ? 180 : null
  if (rotation === null) {
    return <span className="inline-block w-3.5 text-center text-slate-400 font-bold text-xs">–</span>
  }
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      aria-hidden="true"
      style={{ transform: `rotate(${rotation}deg)`, flexShrink: 0 }}
    >
      <path
        d="M7 2.5 L12 11 L2 11 Z"
        fill="#0f172a"
        stroke="#fff"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function capitalize(s: string): string {
  if (!s) return s
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function SectorPopup({ sector, onClose }: Props) {
  const classKey = sector.situation_class ?? 'UNKNOWN'
  const classColor = meteoClassColor(sector.situation_class)
  const classLabel = capitalize(METEO_CLASS_LABELS[classKey] ?? METEO_CLASS_LABELS.UNKNOWN)
  const trendLabel = sector.trend != null
    ? (METEO_TREND_LABELS[sector.trend] ?? 'Inconnu')
    : 'Inconnu'

  return (
    <div className="bg-white rounded-lg shadow-lg border border-slate-200 w-64 text-slate-800">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 px-3 pt-3 pb-2 border-b border-slate-100">
        <div className="min-w-0">
          <p
            className="text-sm font-semibold text-slate-800 truncate"
            title={sector.name}
          >
            {sector.name}
          </p>
          <p className="text-[11px] text-slate-400 font-mono">{sector.code}</p>
        </div>
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
      <div className="px-3 py-2.5 space-y-2">

        {/* Class badge */}
        <div className="flex items-center gap-2">
          <span
            className="w-3 h-3 rounded-full flex-shrink-0"
            style={{ backgroundColor: classColor }}
            aria-hidden="true"
          />
          <span className="text-xs text-slate-700">{classLabel}</span>
        </div>

        {/* Trend */}
        <div className="flex items-center gap-2">
          <TrendArrow trend={sector.trend ?? null} />
          <span className="text-xs text-slate-700">{trendLabel}</span>
        </div>

        {/* % sous la normale */}
        {sector.pct_below_normal != null && (
          <div className="text-xs text-slate-600">
            <span className="font-semibold">{sector.pct_below_normal.toFixed(0)} %</span>{' '}
            sous la normale
          </div>
        )}

        {/* Station counts */}
        <div className="text-xs text-slate-500">
          <span className="font-medium text-slate-700">{sector.n_eligible}</span> station
          {sector.n_eligible !== 1 ? 's' : ''} fiable{sector.n_eligible !== 1 ? 's' : ''}
          {sector.n_provisoire > 0 && (
            <span className="text-slate-400"> (+{sector.n_provisoire} provisoire{sector.n_provisoire !== 1 ? 's' : ''})</span>
          )}
        </div>

      </div>
    </div>
  )
}

// Suppress unused-import warning — METEO_CLASS_COLORS is used indirectly via meteoClassColor
void METEO_CLASS_COLORS
