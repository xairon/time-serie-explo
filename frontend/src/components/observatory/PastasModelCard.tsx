import type { ObsPastasSummary } from '@/lib/observatory-types'

function evpColor(evp: number | null): string { if (evp == null) return '#6b7280'; if (evp >= 70) return '#10b981'; if (evp >= 50) return '#f59e0b'; return '#ef4444' }
function evpLabel(evp: number | null): string { if (evp == null) return '--'; if (evp >= 70) return 'Excellent'; if (evp >= 50) return 'Good'; if (evp >= 30) return 'Moderate'; return 'Poor' }
function fmt(v: number | null | undefined, decimals = 1): string { if (v == null) return '--'; return v.toLocaleString('en-GB', { maximumFractionDigits: decimals }) }
function fmtDate(d: string | null | undefined): string { if (!d) return '--'; return new Date(d).toLocaleDateString('en-GB', { year: 'numeric', month: 'short' }) }

interface Props { summary: ObsPastasSummary }

export function PastasModelCard({ summary }: Props) {
  const metrics = [
    { label: 'Quality (EVP)', value: summary.evp != null ? `${fmt(summary.evp, 1)}%` : '--', sub: evpLabel(summary.evp), color: evpColor(summary.evp) },
    { label: 'Nash-Sutcliffe', value: fmt(summary.nash, 3), sub: summary.kge != null ? `KGE = ${fmt(summary.kge, 3)}` : null },
    { label: 'Max response time', value: summary.tmax_days != null ? `${fmt(summary.tmax_days, 0)} d` : '--', sub: summary.cutoff_95_days != null ? `95% in ${fmt(summary.cutoff_95_days, 0)} d` : null },
    { label: 'Mean response time', value: summary.mean_response_time != null ? `${fmt(summary.mean_response_time, 0)} d` : '--', sub: summary.gain != null ? `Gain = ${fmt(summary.gain, 1)}` : null },
  ]
  const signatures = [
    { label: 'Autocorrelation', value: summary.autocorr_time, unit: 'd' },
    { label: 'Recession constant', value: summary.recession_constant, unit: 'd' },
    { label: 'Seasonality (Parde)', value: summary.parde_seasonality, unit: '' },
    { label: 'Seasonal fluctuation', value: summary.avg_seasonal_fluctuation, unit: 'm' },
  ].filter(s => s.value != null)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {metrics.map(m => (<div key={m.label} className="bg-bg-primary/50 rounded-lg p-3 border border-white/5"><p className="text-[11px] text-text-secondary mb-1">{m.label}</p><p className="text-lg font-semibold font-mono text-text-primary" style={m.color ? { color: m.color } : undefined}>{m.value}</p>{m.sub && <p className="text-[11px] text-text-secondary mt-0.5">{m.sub}</p>}</div>))}
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-text-secondary">
        {signatures.map(s => (<span key={s.label}>{s.label}: <span className="text-text-primary font-mono">{fmt(s.value, 1)}{s.unit ? ` ${s.unit}` : ''}</span></span>))}
        <span>Period: <span className="text-text-primary">{fmtDate(summary.series_start)} -- {fmtDate(summary.series_end)}</span></span>
        {summary.pastas_version && <span>Pastas v{summary.pastas_version}</span>}
      </div>
    </div>
  )
}
