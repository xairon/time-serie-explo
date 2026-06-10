import { METEO_TREND_LABELS } from '@/lib/meteo-colors'
import { TrendBadge } from './TrendBadge'

interface SectorMetrics {
  pctBelowNormal?: number | null
  nEligible?: number
  nProvisoire?: number
}

interface Props {
  name: string
  code: string
  classLabel: string
  trend: 'hausse' | 'stable' | 'baisse' | null
  colorHex: string
  metrics?: SectorMetrics
  onClose: () => void
}

export function SectorPopup({ name, code, classLabel, trend, colorHex, metrics, onClose }: Props) {
  const trendLabel = trend != null ? (METEO_TREND_LABELS[trend] ?? 'Inconnu') : 'Inconnu'

  return (
    <div className="bg-white rounded-lg shadow-lg border border-slate-200 w-64 text-slate-800">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 px-3 pt-3 pb-2 border-b border-slate-100">
        <div className="min-w-0">
          <p
            className="text-sm font-semibold text-slate-800 truncate"
            title={name}
          >
            {name}
          </p>
          <p className="text-[11px] text-slate-400 font-mono">{code}</p>
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
            style={{ backgroundColor: colorHex }}
            aria-hidden="true"
          />
          <span className="text-xs text-slate-700">{classLabel}</span>
        </div>

        {/* Trend */}
        <div className="flex items-center gap-2">
          <TrendBadge kind={trend ?? 'inconnu'} size={16} />
          <span className="text-xs text-slate-700">{trendLabel}</span>
        </div>

        {/* % sous la normale (IPS source) */}
        {metrics?.pctBelowNormal != null && (
          <div className="text-xs text-slate-600">
            <span className="font-semibold">{metrics.pctBelowNormal.toFixed(0)} %</span>{' '}
            sous la normale
          </div>
        )}

        {/* Station counts (IPS source) */}
        {metrics?.nEligible != null && (
          <div className="text-xs text-slate-500">
            <span className="font-medium text-slate-700">{metrics.nEligible}</span> station
            {metrics.nEligible !== 1 ? 's' : ''} fiable{metrics.nEligible !== 1 ? 's' : ''}
            {(metrics.nProvisoire ?? 0) > 0 && (
              <span className="text-slate-400"> (+{metrics.nProvisoire} provisoire{metrics.nProvisoire !== 1 ? 's' : ''})</span>
            )}
          </div>
        )}

      </div>
    </div>
  )
}
