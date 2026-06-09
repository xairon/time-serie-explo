import { METEO_CLASS_COLORS, METEO_CLASS_LABELS } from '@/lib/meteo-colors'

// Arrow SVG for trend indicators (upward triangle, rotated per trend)
function TrendArrow({ rotation }: { rotation: number }) {
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

// Station type rows with a colored dot placeholder
const STATION_TYPES: { label: string; color: string }[] = [
  { label: 'Piézomètre',        color: '#3b7fc8' },
  { label: 'Source',             color: '#60a3d6' },
  { label: 'Station de pluie',   color: '#10b981' },
  { label: 'Station de débit',   color: '#8b5cf6' },
  { label: 'Avec modèle',        color: '#f59e0b' },
]

// Trend rows: label + rotation (0=up, 90=right/stable, 180=down)
const TREND_ROWS: { label: string; rotation?: number }[] = [
  { label: 'en hausse', rotation: 0 },
  { label: 'stable',    rotation: 90 },
  { label: 'en baisse', rotation: 180 },
]

// Niveau swatches in wet-first order
const NIVEAU_SWATCHES: { hex: string; label: string }[] = [
  { hex: METEO_CLASS_COLORS.EXTREMEMENT_HAUT, label: METEO_CLASS_LABELS.EXTREMEMENT_HAUT },
  { hex: METEO_CLASS_COLORS.TRES_HAUT,        label: METEO_CLASS_LABELS.TRES_HAUT },
  { hex: METEO_CLASS_COLORS.HAUT,             label: METEO_CLASS_LABELS.HAUT },
  { hex: METEO_CLASS_COLORS.NORMAL,           label: METEO_CLASS_LABELS.NORMAL },
  { hex: METEO_CLASS_COLORS.BAS,              label: METEO_CLASS_LABELS.BAS },
  { hex: METEO_CLASS_COLORS.TRES_BAS,         label: METEO_CLASS_LABELS.TRES_BAS },
  { hex: METEO_CLASS_COLORS.EXTREMEMENT_BAS,  label: METEO_CLASS_LABELS.EXTREMEMENT_BAS },
  { hex: METEO_CLASS_COLORS.UNKNOWN,          label: METEO_CLASS_LABELS.UNKNOWN },
]

export function MeteoLegend() {
  return (
    <div className="absolute bottom-16 left-3 z-10 bg-white rounded-lg shadow-md border border-slate-200 p-3 w-64 max-h-[calc(100vh-200px)] overflow-y-auto">

      {/* Section: Type */}
      <div className="mb-3">
        <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
          Type
        </h4>
        <div className="space-y-1">
          {STATION_TYPES.map(({ label, color }) => (
            <div key={label} className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: color }}
                aria-hidden="true"
              />
              <span className="text-xs text-slate-700">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-slate-100 mb-3" />

      {/* Section: Evolution des niveaux */}
      <div className="mb-3">
        <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
          Evolution des niveaux
        </h4>
        <div className="space-y-1">
          {TREND_ROWS.map(({ label, rotation }) => (
            <div key={label} className="flex items-center gap-2">
              {rotation !== undefined ? (
                <TrendArrow rotation={rotation} />
              ) : (
                <span className="w-3.5 text-center text-slate-400 text-xs font-bold" aria-hidden="true">–</span>
              )}
              <span className="text-xs text-slate-700">{label}</span>
            </div>
          ))}
          {/* Inconnu row — grey dash */}
          <div className="flex items-center gap-2">
            <span
              className="inline-block w-3.5 text-center text-slate-400 font-bold text-xs leading-none flex-shrink-0"
              aria-hidden="true"
            >
              –
            </span>
            <span className="text-xs text-slate-500">Inconnu</span>
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-slate-100 mb-3" />

      {/* Section: Niveau */}
      <div>
        <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
          Niveau
        </h4>
        <div className="space-y-1">
          {NIVEAU_SWATCHES.map(({ hex, label }) => (
            <div key={hex} className="flex items-center gap-2">
              <span
                className="flex-shrink-0"
                style={{ width: 10, height: 10, backgroundColor: hex, display: 'inline-block' }}
                aria-hidden="true"
              />
              <span className="text-[11px] text-slate-700 leading-tight">{label}</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  )
}
