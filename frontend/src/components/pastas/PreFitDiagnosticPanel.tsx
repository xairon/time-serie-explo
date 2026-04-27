import { BarChart3, Clock, TrendingDown, Scissors, Waves, Calendar } from 'lucide-react'
import type { DiagnoseResult, DiagnosticRecommendation } from '@/lib/types'
import { InfoTip } from './InfoTip'

interface Props {
  diagnosis: DiagnoseResult | undefined
  isLoading: boolean
  onApplyRecommendation: (rec: DiagnosticRecommendation) => void
  mode: 'guided' | 'expert'
}

const STATUS_COLORS: Record<string, string> = {
  green: '#10b981',
  orange: '#f59e0b',
  red: '#ef4444',
}

const INDICATORS = [
  { key: 'coverage' as const, label: 'Coverage', Icon: BarChart3, tip: 'Percentage of days with actual measurements between the first and last observation. Green > 80%, orange 50-80%, red < 50%. Low coverage means the model must interpolate over long missing periods.' },
  { key: 'gaps' as const, label: 'Gaps', Icon: Clock, tip: 'Largest consecutive gap without data (in days). Green < 30 days, orange 30-180 days, red > 180 days. Gaps > 6 months can distort the calibration — consider trimming the period to start after the gap.' },
  { key: 'trend' as const, label: 'Trend', Icon: TrendingDown, tip: 'Mann-Kendall test for monotonic trend in the water level. If a significant trend is detected (p < 0.05), consider adding a LinearTrend stress to the Pastas model so residuals stay stationary.' },
  { key: 'breakpoints' as const, label: 'Breakpoints', Icon: Scissors, tip: 'Pettitt change-point test — detects a single abrupt shift in the mean level (e.g. new pumping well, land-use change). If detected, the period before the break may need to be excluded.' },
  { key: 'seasonality' as const, label: 'Seasonality', Icon: Waves, tip: 'Autocorrelation at lag 12 months (ACF₁₂) — measures the strength of the annual cycle. Green > 0.3 (strong, the model can capture it), orange 0.1-0.3, red < 0.1 (weak or no annual signal — the aquifer may be deep or heavily influenced by pumping).' },
  { key: 'record_length' as const, label: 'Record', Icon: Calendar, tip: 'Total duration of available data in years. Green ≥ 15 years (robust calibration), orange 5-15 years, red < 5 years. Short records limit the model\'s ability to capture low-frequency variability and drought cycles.' },
]

function SkeletonCard() {
  return (
    <div className="bg-bg-card rounded-lg border border-white/5 p-3 animate-pulse">
      <div className="flex items-start gap-2">
        <div className="w-4 h-4 rounded bg-white/10 mt-0.5 shrink-0" />
        <div className="flex-1 space-y-2">
          <div className="h-3 w-16 bg-white/10 rounded" />
          <div className="h-3 w-24 bg-white/10 rounded" />
        </div>
        <div className="w-2 h-2 rounded-full bg-white/10 mt-1 shrink-0" />
      </div>
    </div>
  )
}

export function PreFitDiagnosticPanel({ diagnosis, isLoading, onApplyRecommendation, mode }: Props) {
  if (isLoading) {
    return (
      <div className="space-y-3">
        <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
          Pre-fit Diagnostics
        </div>
        <div className="grid grid-cols-3 gap-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
        <div className="space-y-2 animate-pulse">
          <div className="h-3 w-32 bg-white/10 rounded" />
          <div className="h-8 bg-white/5 rounded" />
        </div>
      </div>
    )
  }

  if (!diagnosis) {
    return null
  }

  const recommendations = diagnosis.recommendations ?? []

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
        Pre-fit Diagnostics
      </div>

      <div className="grid grid-cols-3 gap-2">
        {INDICATORS.map(({ key, label, Icon, tip }) => {
          const indicator = diagnosis[key]
          if (!indicator) return null
          const dotColor = STATUS_COLORS[indicator.status] ?? STATUS_COLORS.orange

          return (
            <div
              key={key}
              className="bg-bg-card rounded-lg border border-white/5 p-3 flex items-start gap-2"
            >
              <Icon className="w-4 h-4 text-text-muted mt-0.5 shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium text-text-secondary leading-tight flex items-center gap-1">
                  {label}
                  <InfoTip text={tip} />
                </div>
                {indicator.detail && (
                  <div className="text-xs text-text-muted mt-0.5 leading-tight truncate" title={indicator.detail}>
                    {indicator.detail}
                  </div>
                )}
              </div>
              <div
                className="w-2 h-2 rounded-full mt-1 shrink-0"
                style={{ backgroundColor: dotColor }}
              />
            </div>
          )
        })}
      </div>

      {recommendations.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-semibold text-text-muted uppercase tracking-wide">
            Recommendations
          </div>
          <div className="space-y-1.5">
            {recommendations.map((rec, i) => (
              <div
                key={i}
                className="flex items-start justify-between gap-3 bg-bg-card rounded-lg border border-white/5 px-3 py-2"
              >
                <p className="text-xs text-text-secondary leading-snug">{rec.message}</p>
                {mode === 'guided' ? (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20 shrink-0">
                    Auto-applied
                  </span>
                ) : (
                  <button
                    onClick={() => onApplyRecommendation(rec)}
                    className="text-xs px-2 py-0.5 rounded-full border border-accent-cyan/40 text-accent-cyan hover:bg-accent-cyan/10 transition-colors shrink-0"
                  >
                    Apply
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
