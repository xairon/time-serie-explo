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
  { key: 'coverage' as const, label: 'Couverture', Icon: BarChart3, tip: 'Pourcentage de jours avec des mesures réelles entre la première et la dernière observation. Vert > 80%, orange 50-80%, rouge < 50%. Une couverture faible signifie que le modèle doit interpoler sur de longues périodes manquantes.' },
  { key: 'gaps' as const, label: 'Lacunes', Icon: Clock, tip: 'Plus grande lacune consécutive sans données (en jours). Vert < 30 jours, orange 30-180 jours, rouge > 180 jours. Les lacunes > 6 mois peuvent biaiser la calibration — envisager de tronquer la période pour commencer après la lacune.' },
  { key: 'trend' as const, label: 'Tendance', Icon: TrendingDown, tip: 'Test de Mann-Kendall pour la tendance monotone du niveau d\'eau. Si une tendance significative est détectée (p < 0,05), envisager d\'ajouter un stress LinearTrend au modèle Pastas pour que les résidus restent stationnaires.' },
  { key: 'breakpoints' as const, label: 'Ruptures', Icon: Scissors, tip: 'Test de rupture de Pettitt — détecte un saut abrupt unique de la moyenne (ex. nouveau forage de pompage, changement d\'usage des sols). Si détecté, la période avant la rupture peut nécessiter une exclusion.' },
  { key: 'seasonality' as const, label: 'Saisonnalité', Icon: Waves, tip: 'Autocorrélation au décalage de 12 mois (ACF₁₂) — mesure la force du cycle annuel. Vert > 0,3 (fort, le modèle peut le capturer), orange 0,1-0,3, rouge < 0,1 (signal annuel faible ou absent — l\'aquifère peut être profond ou fortement influencé par le pompage).' },
  { key: 'record_length' as const, label: 'Chronique', Icon: Calendar, tip: 'Durée totale des données disponibles en années. Vert ≥ 15 ans (calibration robuste), orange 5-15 ans, rouge < 5 ans. Des chroniques courtes limitent la capacité du modèle à capturer la variabilité basse fréquence et les cycles de sécheresse.' },
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
          Diagnostics pré-calibration
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
        Diagnostics pré-calibration
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
            Recommandations
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
                    Appliqué automatiquement
                  </span>
                ) : (
                  <button
                    onClick={() => onApplyRecommendation(rec)}
                    className="text-xs px-2 py-0.5 rounded-full border border-accent-cyan/40 text-accent-cyan hover:bg-accent-cyan/10 transition-colors shrink-0"
                  >
                    Appliquer
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
