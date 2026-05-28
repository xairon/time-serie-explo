import { useTranslation } from 'react-i18next'
import type { ObsPastasSummary } from '@/lib/observatory-types'

function evpColor(evp: number | null): string { if (evp == null) return '#6b7280'; if (evp >= 70) return '#10b981'; if (evp >= 50) return '#f59e0b'; return '#ef4444' }

interface Props { summary: ObsPastasSummary }

export function PastasModelCard({ summary }: Props) {
  const { t, i18n } = useTranslation()
  const localeTag = i18n.language?.startsWith('en') ? 'en-US' : 'fr-FR'
  const fmt = (v: number | null | undefined, decimals = 1): string => {
    if (v == null) return '--'
    return v.toLocaleString(localeTag, { maximumFractionDigits: decimals })
  }
  const fmtDate = (d: string | null | undefined): string => {
    if (!d) return '--'
    return new Date(d).toLocaleDateString(localeTag, { year: 'numeric', month: 'short' })
  }
  const evpLabel = (evp: number | null): string => {
    if (evp == null) return '--'
    if (evp >= 70) return t('observatory.pastas.qualityExcellent')
    if (evp >= 50) return t('observatory.pastas.qualityGood')
    if (evp >= 30) return t('observatory.pastas.qualityFair')
    return t('observatory.pastas.qualityPoor')
  }

  const metrics = [
    { label: t('observatory.pastas.qualityEvp'), value: summary.evp != null ? `${fmt(summary.evp, 1)}%` : '--', sub: evpLabel(summary.evp), color: evpColor(summary.evp) },
    { label: t('observatory.pastas.nashSutcliffe'), value: fmt(summary.nash, 3), sub: summary.kge != null ? `KGE = ${fmt(summary.kge, 3)}` : null },
    { label: t('observatory.pastas.maxResponseTime'), value: summary.tmax_days != null ? `${fmt(summary.tmax_days, 0)} j` : '--', sub: summary.cutoff_95_days != null ? t('observatory.pastas.in95Days', { days: fmt(summary.cutoff_95_days, 0) }) : null },
    { label: t('observatory.pastas.meanResponseTime'), value: summary.mean_response_time != null ? `${fmt(summary.mean_response_time, 0)} j` : '--', sub: summary.gain != null ? `Gain = ${fmt(summary.gain, 1)}` : null },
  ]
  const signatures = [
    { label: t('observatory.pastas.autocorrelation'), value: summary.autocorr_time, unit: 'j' },
    { label: t('observatory.pastas.recessionConstant'), value: summary.recession_constant, unit: 'j' },
    { label: t('observatory.pastas.seasonalityParde'), value: summary.parde_seasonality, unit: '' },
    { label: t('observatory.pastas.seasonalFluctuation'), value: summary.avg_seasonal_fluctuation, unit: 'm' },
  ].filter(s => s.value != null)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {metrics.map(m => (<div key={m.label} className="bg-bg-primary/50 rounded-lg p-3 border border-white/5"><p className="text-[11px] text-text-secondary mb-1">{m.label}</p><p className="text-lg font-semibold font-mono text-text-primary" style={m.color ? { color: m.color } : undefined}>{m.value}</p>{m.sub && <p className="text-[11px] text-text-secondary mt-0.5">{m.sub}</p>}</div>))}
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-text-secondary">
        {signatures.map(s => (<span key={s.label}>{s.label}: <span className="text-text-primary font-mono">{fmt(s.value, 1)}{s.unit ? ` ${s.unit}` : ''}</span></span>))}
        <span>{t('observatory.pastas.period')} : <span className="text-text-primary">{fmtDate(summary.series_start)} -- {fmtDate(summary.series_end)}</span></span>
        {summary.pastas_version && <span>{t('observatory.pastas.version', { version: summary.pastas_version })}</span>}
      </div>
    </div>
  )
}
