// frontend/src/components/pastas/OutlierDetailPanel.tsx
import { X } from 'lucide-react'

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

const CATEGORY_COLORS: Record<string, string> = {
  DATA_GAP: 'bg-red-500/20 text-red-400 border-red-500/30',
  CLIMATE_EXTREME: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  REGIONAL_SIGNAL: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  SEASONAL_BIAS: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  DOMINANT_CONTRIBUTION: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  UNKNOWN: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
}

function CategoryBadge({ category, label, outline = false }: { category: string; label: string; outline?: boolean }) {
  const color = CATEGORY_COLORS[category] ?? CATEGORY_COLORS.UNKNOWN
  return (
    <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${color} ${outline ? 'bg-transparent' : ''}`}>
      {label}
    </span>
  )
}

function SeverityDots({ severity }: { severity: number }) {
  const filled = Math.max(1, Math.round(severity * 4))
  return (
    <span className="flex gap-0.5" title={`Severity: ${(severity * 100).toFixed(0)}%`}>
      {[1, 2, 3, 4].map(i => (
        <span key={i} className={`w-1.5 h-1.5 rounded-full ${i <= filled ? 'bg-red-400' : 'bg-white/10'}`} />
      ))}
    </span>
  )
}

export function OutlierDetailPanel({ outlier, onClose }: Props) {
  const { climate, contributions, data_quality, neighbors } = outlier

  return (
    <div className="mt-2 bg-bg-card border border-white/10 rounded-lg overflow-hidden animate-in slide-in-from-top-2 duration-200">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-white/5">
        <CategoryBadge category={outlier.category} label={outlier.category_label} />
        {outlier.secondary_tags.map(tag => (
          <CategoryBadge key={tag} category={tag} label={tag.replace(/_/g, ' ').toLowerCase()} outline />
        ))}
        <span className="text-xs text-text-primary font-medium ml-1">
          {new Date(outlier.date).toLocaleDateString('en-GB', { year: 'numeric', month: 'short' })}
        </span>
        <span className="text-xs text-text-muted">|</span>
        <span className="text-xs font-mono text-text-secondary">
          {outlier.residual > 0 ? '+' : ''}{outlier.residual.toFixed(3)}m ({outlier.residual_zscore.toFixed(1)}σ)
        </span>
        <SeverityDots severity={outlier.severity} />
        <button onClick={onClose} className="ml-auto p-1 hover:bg-bg-hover rounded transition-colors">
          <X className="w-3.5 h-3.5 text-text-muted" />
        </button>
      </div>

      {/* Explanation */}
      <div className="px-4 py-2 border-b border-white/5">
        <p className="text-xs text-text-secondary leading-relaxed">{outlier.explanation}</p>
      </div>

      {/* Context grid */}
      <div className="grid grid-cols-3 gap-px bg-white/5">
        {/* Climate column */}
        <div className="bg-bg-card p-3">
          <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-2">Climate</div>
          {climate.precip_mm != null && (
            <div className="flex items-center justify-between py-1 border-b border-white/5">
              <span className="text-[10px] text-text-muted">Precip</span>
              <span className={`text-[10px] font-mono ${climate.precip_zscore && Math.abs(climate.precip_zscore) > 1.5 ? 'text-orange-400 font-semibold' : 'text-text-secondary'}`}>
                {climate.precip_mm.toFixed(0)}mm
                {climate.precip_zscore != null && ` (${climate.precip_zscore > 0 ? '+' : ''}${climate.precip_zscore.toFixed(1)}σ)`}
              </span>
            </div>
          )}
          {climate.temp_c != null && (
            <div className="flex items-center justify-between py-1 border-b border-white/5">
              <span className="text-[10px] text-text-muted">Temp</span>
              <span className={`text-[10px] font-mono ${climate.temp_zscore && Math.abs(climate.temp_zscore) > 1.5 ? 'text-orange-400 font-semibold' : 'text-text-secondary'}`}>
                {climate.temp_c.toFixed(1)}°C
                {climate.temp_zscore != null && ` (${climate.temp_zscore > 0 ? '+' : ''}${climate.temp_zscore.toFixed(1)}σ)`}
              </span>
            </div>
          )}
          {climate.etp_mm != null && (
            <div className="flex items-center justify-between py-1 border-b border-white/5">
              <span className="text-[10px] text-text-muted">ETP</span>
              <span className={`text-[10px] font-mono ${climate.etp_zscore && Math.abs(climate.etp_zscore) > 1.5 ? 'text-orange-400 font-semibold' : 'text-text-secondary'}`}>
                {climate.etp_mm.toFixed(1)}mm
                {climate.etp_zscore != null && ` (${climate.etp_zscore > 0 ? '+' : ''}${climate.etp_zscore.toFixed(1)}σ)`}
              </span>
            </div>
          )}
          {climate.spli != null && (
            <div className="flex items-center justify-between py-1 border-b border-white/5">
              <span className="text-[10px] text-text-muted">SPLI</span>
              <span className="text-[10px] font-mono text-text-secondary">{climate.spli.toFixed(2)} — {climate.spli_class}</span>
            </div>
          )}
          {climate.spi != null && (
            <div className="flex items-center justify-between py-1">
              <span className="text-[10px] text-text-muted">SPI</span>
              <span className="text-[10px] font-mono text-text-secondary">{climate.spi.toFixed(2)} — {climate.spi_class}</span>
            </div>
          )}
        </div>

        {/* Model column */}
        <div className="bg-bg-card p-3">
          <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-2">Model</div>
          <div className="flex items-center justify-between py-1 border-b border-white/5">
            <span className="text-[10px] text-text-muted">Observed</span>
            <span className="text-[10px] font-mono text-text-secondary">{outlier.observed.toFixed(3)}m</span>
          </div>
          <div className="flex items-center justify-between py-1 border-b border-white/5">
            <span className="text-[10px] text-text-muted">Simulated</span>
            <span className="text-[10px] font-mono text-accent-cyan">{outlier.simulated.toFixed(3)}m</span>
          </div>
          {Object.entries(contributions).map(([name, value]) => (
            <div key={name} className="flex items-center justify-between py-1 border-b border-white/5 last:border-0">
              <span className="text-[10px] text-text-muted truncate mr-2">{name}</span>
              <span className="text-[10px] font-mono text-text-secondary">{value > 0 ? '+' : ''}{value.toFixed(3)}m</span>
            </div>
          ))}
        </div>

        {/* Data quality column */}
        <div className="bg-bg-card p-3">
          <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-2">Data Quality</div>
          <div className="flex items-center justify-between py-1 border-b border-white/5">
            <span className="text-[10px] text-text-muted">Coverage (±30d)</span>
            <span className={`text-[10px] font-mono ${data_quality.coverage_pct < 90 ? 'text-orange-400' : 'text-text-secondary'}`}>
              {data_quality.coverage_pct.toFixed(0)}%
            </span>
          </div>
          <div className="flex items-center justify-between py-1 border-b border-white/5">
            <span className="text-[10px] text-text-muted">Gap days</span>
            <span className={`text-[10px] font-mono ${data_quality.gap_days > 0 ? 'text-red-400' : 'text-text-secondary'}`}>
              {data_quality.gap_days}
            </span>
          </div>
          {data_quality.nearest_gap_distance_days != null && (
            <div className="flex items-center justify-between py-1">
              <span className="text-[10px] text-text-muted">Nearest gap</span>
              <span className="text-[10px] font-mono text-text-secondary">{data_quality.nearest_gap_distance_days}d</span>
            </div>
          )}
        </div>
      </div>

      {/* Neighbors */}
      {neighbors.total > 0 && (
        <div className="px-4 py-2.5 border-t border-white/5">
          <span className="text-[10px] text-text-muted">
            BDLISA neighbors: <span className={neighbors.anomalous > 0 ? 'text-blue-400 font-medium' : ''}>{neighbors.anomalous}/{neighbors.total} anomalous</span>
          </span>
          <div className="flex flex-wrap gap-1 mt-1.5">
            {neighbors.neighbor_zscores.map(n => (
              <span
                key={n.code_bss}
                className={`px-1.5 py-0.5 rounded text-[9px] font-mono border ${
                  Math.abs(n.zscore) > 1.5
                    ? 'border-red-500/30 bg-red-500/10 text-red-400'
                    : 'border-white/10 bg-white/5 text-text-muted'
                }`}
              >
                {n.code_bss.split('/').pop()}: {n.zscore > 0 ? '+' : ''}{n.zscore.toFixed(1)}σ
              </span>
            ))}
          </div>
        </div>
      )}
      {neighbors.total === 0 && (
        <div className="px-4 py-2 border-t border-white/5">
          <span className="text-[10px] text-text-muted">No BDLISA neighbors found</span>
        </div>
      )}
    </div>
  )
}
