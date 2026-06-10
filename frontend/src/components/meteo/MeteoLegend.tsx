// frontend/src/components/meteo/MeteoLegend.tsx
import { METEO_CLASS_COLORS, METEO_CLASS_LABELS, METEO_TREND_LABELS } from '@/lib/meteo-colors'
import { TrendBadge, type TrendBadgeKind } from './TrendBadge'

const TREND_ROWS: { kind: TrendBadgeKind; label: string }[] = [
  { kind: 'hausse', label: METEO_TREND_LABELS.hausse },
  { kind: 'stable', label: METEO_TREND_LABELS.stable },
  { kind: 'baisse', label: METEO_TREND_LABELS.baisse },
  { kind: 'inconnu', label: 'Inconnu' },
]

// Wet-first vertical scale, like the original's "Level" card.
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
    <div className="space-y-2">
      {/* Évolution des niveaux */}
      <div className="bg-white rounded-lg shadow-md border border-slate-200 p-3 w-56">
        <h4 className="text-[11px] font-semibold text-slate-600 mb-1.5">Évolution des niveaux</h4>
        <div className="space-y-1">
          {TREND_ROWS.map(({ kind, label }) => (
            <div key={kind} className="flex items-center gap-2">
              <span className="inline-flex rounded-full bg-slate-100"><TrendBadge kind={kind} size={15} /></span>
              <span className="text-[11px] text-slate-700">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Niveau */}
      <div className="bg-white rounded-lg shadow-md border border-slate-200 p-3 w-56">
        <h4 className="text-[11px] font-semibold text-slate-600 mb-1.5">Niveau</h4>
        <div className="space-y-0">
          {NIVEAU_SWATCHES.map(({ hex, label }) => (
            <div key={hex} className="flex items-center gap-2">
              <span className="flex-shrink-0" style={{ width: 12, height: 14, backgroundColor: hex }} aria-hidden="true" />
              <span className="text-[11px] text-slate-700 leading-[14px]">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
