import { X, AlertTriangle, CloudRain, Globe, CalendarClock, BarChart3, HelpCircle, Database } from 'lucide-react'
import { useTranslation } from 'react-i18next'

interface OutlierDiagnostic {
  date: string
  residual: number
  residual_zscore: number
  severity: number
  category: string
  category_label: string
  secondary_tags: string[]
  explanation: string
  climate: {
    precip_mm: number | null; precip_zscore: number | null
    temp_c: number | null; temp_zscore: number | null
    etp_mm: number | null; etp_zscore: number | null
    spli: number | null; spli_class: string | null
    spi: number | null; spi_class: string | null
  }
  contributions: Record<string, number>
  observed: number
  simulated: number
  data_quality: {
    gap_days: number; coverage_pct: number
    nearest_gap_distance_days: number | null
  }
  neighbors: {
    total: number; anomalous: number
    neighbor_zscores: { code_bss: string; zscore: number }[]
  }
}

interface Props {
  outlier: OutlierDiagnostic
  onClose: () => void
}

function getCategoryMeta(t: (k: string) => string): Record<string, { color: string; icon: any; verdict: string }> {
  return {
    DATA_GAP: { color: 'red', icon: Database, verdict: t('pastas.outlier.verdictDataGap') },
    CLIMATE_EXTREME: { color: 'orange', icon: CloudRain, verdict: t('pastas.outlier.verdictClimateExtreme') },
    REGIONAL_SIGNAL: { color: 'blue', icon: Globe, verdict: t('pastas.outlier.verdictRegional') },
    SEASONAL_BIAS: { color: 'yellow', icon: CalendarClock, verdict: t('pastas.outlier.verdictSeasonal') },
    DOMINANT_CONTRIBUTION: { color: 'purple', icon: BarChart3, verdict: t('pastas.outlier.verdictDominant') },
    UNKNOWN: { color: 'gray', icon: HelpCircle, verdict: t('pastas.outlier.verdictUnknown') },
  }
}

const CATEGORY_CLASSES: Record<string, { bg: string; text: string; border: string; bgLight: string }> = {
  DATA_GAP:              { bg: 'bg-red-500/15',    text: 'text-red-400',    border: 'border-red-500/30',    bgLight: 'bg-red-500/5' },
  CLIMATE_EXTREME:       { bg: 'bg-orange-500/15', text: 'text-orange-400', border: 'border-orange-500/30', bgLight: 'bg-orange-500/5' },
  REGIONAL_SIGNAL:       { bg: 'bg-blue-500/15',   text: 'text-blue-400',   border: 'border-blue-500/30',   bgLight: 'bg-blue-500/5' },
  SEASONAL_BIAS:         { bg: 'bg-yellow-500/15', text: 'text-yellow-400', border: 'border-yellow-500/30', bgLight: 'bg-yellow-500/5' },
  DOMINANT_CONTRIBUTION: { bg: 'bg-purple-500/15', text: 'text-purple-400', border: 'border-purple-500/30', bgLight: 'bg-purple-500/5' },
  UNKNOWN:               { bg: 'bg-gray-500/15',   text: 'text-gray-400',   border: 'border-gray-500/30',   bgLight: 'bg-gray-500/5' },
}

function SeverityBar({ severity }: { severity: number }) {
  const pct = Math.round(severity * 100)
  const color = severity > 0.8 ? 'bg-red-500' : severity > 0.5 ? 'bg-orange-500' : 'bg-yellow-500'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] text-text-muted w-8 text-right">{pct}%</span>
    </div>
  )
}

function StatRow({ label, value, alert }: { label: string; value: string; alert?: boolean }) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-[10px] text-text-muted">{label}</span>
      <span className={`text-[10px] font-mono ${alert ? 'text-orange-400 font-semibold' : 'text-text-secondary'}`}>{value}</span>
    </div>
  )
}

export function OutlierDetailPanel({ outlier, onClose }: Props) {
  const { t } = useTranslation()
  const CATEGORY_META = getCategoryMeta(t)
  const { climate, contributions, data_quality, neighbors } = outlier
  const meta = CATEGORY_META[outlier.category] ?? CATEGORY_META.UNKNOWN
  const cls = CATEGORY_CLASSES[outlier.category] ?? CATEGORY_CLASSES.UNKNOWN
  const Icon = meta.icon
  const direction = outlier.residual > 0 ? t('pastas.outlier.modelOver') : t('pastas.outlier.modelUnder')
  const error = Math.abs(outlier.observed - outlier.simulated)

  return (
    <div className="mt-2 bg-bg-card border border-white/10 rounded-xl overflow-hidden">
      {/* Top bar: close + date + error summary */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-white/5">
        <div className={`p-1.5 rounded-lg ${cls.bg}`}>
          <Icon className={`w-4 h-4 ${cls.text}`} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-text-primary">
              {new Date(outlier.date).toLocaleDateString('fr-FR', { year: 'numeric', month: 'long' })}
            </span>
            <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${cls.border} ${cls.bg} ${cls.text}`}>
              {outlier.category_label}
            </span>
            {outlier.secondary_tags.map(tag => (
              <span key={tag} className="px-1.5 py-0.5 rounded-full text-[9px] border border-white/10 text-text-muted">
                {tag.replace(/_/g, ' ').toLowerCase()}
              </span>
            ))}
          </div>
          <p className="text-[11px] text-text-secondary mt-0.5">
            Le modèle {direction} de <span className="font-semibold text-text-primary">{error.toFixed(2)} m</span>
            {' '}({t('pastas.signatures.observed').toLowerCase()} {outlier.observed.toFixed(2)} m vs {t('pastas.signatures.simulated').toLowerCase()} {outlier.simulated.toFixed(2)} m)
          </p>
        </div>
        <button onClick={onClose} className="p-1.5 hover:bg-bg-hover rounded-lg transition-colors shrink-0">
          <X className="w-4 h-4 text-text-muted" />
        </button>
      </div>

      {/* Verdict — the key takeaway */}
      <div className={`px-4 py-2.5 border-b border-white/5 ${cls.bgLight}`}>
        <div className="flex items-start gap-2">
          <AlertTriangle className={`w-3.5 h-3.5 ${cls.text} mt-0.5 shrink-0`} />
          <p className="text-xs text-text-secondary leading-relaxed">{meta.verdict}</p>
        </div>
      </div>

      {/* Severity */}
      <div className="px-4 py-2 border-b border-white/5">
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-text-muted w-14">{t('pastas.outlier.severity')}</span>
          <div className="flex-1"><SeverityBar severity={outlier.severity} /></div>
          <span className="text-[10px] font-mono text-text-secondary">{outlier.residual_zscore.toFixed(1)}σ</span>
        </div>
      </div>

      {/* Evidence grid */}
      <div className="grid grid-cols-3 divide-x divide-white/5">
        {/* What happened (climate) */}
        <div className="p-3">
          <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-2">{t('pastas.outlier.whatHappened')}</div>
          {climate.precip_mm != null && (
            <StatRow label={t('pastas.outlier.precipitation')} value={`${climate.precip_mm.toFixed(0)} mm (${climate.precip_zscore != null ? (climate.precip_zscore > 0 ? '+' : '') + climate.precip_zscore.toFixed(1) + 'σ' : '—'})`} alert={climate.precip_zscore != null && Math.abs(climate.precip_zscore) > 1.5} />
          )}
          {climate.temp_c != null && (
            <StatRow label={t('pastas.outlier.temperature')} value={`${climate.temp_c.toFixed(1)} °C (${climate.temp_zscore != null ? (climate.temp_zscore > 0 ? '+' : '') + climate.temp_zscore.toFixed(1) + 'σ' : '—'})`} alert={climate.temp_zscore != null && Math.abs(climate.temp_zscore) > 1.5} />
          )}
          {climate.etp_mm != null && (
            <StatRow label={t('pastas.outlier.evapotranspiration')} value={`${climate.etp_mm.toFixed(1)} mm (${climate.etp_zscore != null ? (climate.etp_zscore > 0 ? '+' : '') + climate.etp_zscore.toFixed(1) + 'σ' : '—'})`} alert={climate.etp_zscore != null && Math.abs(climate.etp_zscore) > 1.5} />
          )}
          {climate.spli != null && <StatRow label={t('pastas.outlier.drySpli')} value={`${climate.spli.toFixed(1)} — ${climate.spli_class?.replace(/_/g, ' ').toLowerCase()}`} alert={Math.abs(climate.spli) > 1.5} />}
          {climate.spi != null && <StatRow label={t('pastas.outlier.rainSpi')} value={`${climate.spi.toFixed(1)} — ${climate.spi_class?.replace(/_/g, ' ').toLowerCase()}`} alert={Math.abs(climate.spi) > 1.5} />}
        </div>

        {/* What the model did (contributions) */}
        <div className="p-3">
          <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-2">{t('pastas.outlier.modelResponse')}</div>
          {Object.entries(contributions).map(([name, value]) => (
            <StatRow key={name} label={name} value={`${value > 0 ? '+' : ''}${value.toFixed(3)} m`} />
          ))}
          {Object.keys(contributions).length === 0 && <span className="text-[10px] text-text-muted">{t('pastas.outlier.noContribData')}</span>}
        </div>

        {/* Data reliability */}
        <div className="p-3">
          <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-2">{t('pastas.outlier.dataReliability')}</div>
          <StatRow label={t('pastas.outlier.coverage30d')} value={`${data_quality.coverage_pct.toFixed(0)}%`} alert={data_quality.coverage_pct < 90} />
          <StatRow label={t('pastas.outlier.missingDays')} value={String(data_quality.gap_days)} alert={data_quality.gap_days > 0} />
          {data_quality.nearest_gap_distance_days != null && (
            <StatRow label={t('pastas.outlier.nearestGap')} value={t('pastas.outlier.nearestGapAt', { days: data_quality.nearest_gap_distance_days })} alert={data_quality.nearest_gap_distance_days < 15} />
          )}
        </div>
      </div>

      {/* Neighbors — regional context */}
      {neighbors.total > 0 && (
        <div className="px-4 py-2.5 border-t border-white/5">
          <div className="flex items-center gap-2 mb-1.5">
            <Globe className="w-3 h-3 text-text-muted" />
            <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wide">
              {t('pastas.outlier.neighborStations', { anomalous: neighbors.anomalous, total: neighbors.total })}
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {neighbors.neighbor_zscores.map(n => {
              const isAnomalous = Math.abs(n.zscore) > 1.5
              return (
                <span
                  key={n.code_bss}
                  className={`px-2 py-0.5 rounded-md text-[10px] font-mono border ${
                    isAnomalous
                      ? 'border-red-500/30 bg-red-500/10 text-red-400'
                      : 'border-white/10 bg-white/5 text-text-muted'
                  }`}
                >
                  {n.code_bss.split('/').pop()}: {n.zscore > 0 ? '+' : ''}{n.zscore.toFixed(1)}σ
                  {isAnomalous ? ' !' : ''}
                </span>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
