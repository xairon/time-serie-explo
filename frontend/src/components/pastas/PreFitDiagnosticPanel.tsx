import { BarChart3, Clock, TrendingDown, Scissors, Waves, Calendar } from 'lucide-react'
import type { DiagnoseResult, DiagnosticRecommendation } from '@/lib/types'

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
  { key: 'coverage' as const, label: 'Coverage', Icon: BarChart3 },
  { key: 'gaps' as const, label: 'Gaps', Icon: Clock },
  { key: 'trend' as const, label: 'Trend', Icon: TrendingDown },
  { key: 'breakpoints' as const, label: 'Breakpoints', Icon: Scissors },
  { key: 'seasonality' as const, label: 'Seasonality', Icon: Waves },
  { key: 'record_length' as const, label: 'Record', Icon: Calendar },
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
        {INDICATORS.map(({ key, label, Icon }) => {
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
                <div className="text-xs font-medium text-text-secondary leading-tight">{label}</div>
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
