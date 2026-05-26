import { useState } from 'react'
import {
  TrendingUp,
  Activity,
  Clock,
  Sun,
  Beaker,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
} from 'lucide-react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import {
  useFeatureImportance,
  usePermutationImportance,
  useShapAnalysis,
  useGradientAnalysis,
  useLagImportance,
  useResidualAnalysis,
  useSeasonalityAnalysis,
} from '@/hooks/useForecasting'
import type { ExplainResult } from '@/lib/types'
import { InfoTip } from '@/components/pastas/InfoTip'

interface Props {
  modelId: string
  className?: string
}

type Section = 'drivers' | 'quality' | 'behavior' | 'expert'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function LoadingSkeleton() {
  return <div className="h-[200px] bg-bg-hover rounded-lg animate-pulse" />
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="text-center py-6">
      <p className="text-xs text-red-400 mb-2">{message}</p>
      <button onClick={onRetry} className="text-xs text-accent-cyan hover:underline">Réessayer</button>
    </div>
  )
}

function extractImportance(data: ExplainResult): { features: string[]; values: number[] } | null {
  if (!data.feature_importance) return null
  const entries = Object.entries(data.feature_importance)
    .filter(([, v]) => v != null)
    .map(([k, v]) => [k, v as number] as const)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
  if (entries.length === 0) return null
  return { features: entries.map(([k]) => k), values: entries.map(([, v]) => v) }
}

const HYDRO_LABELS: Record<string, string> = {
  total_precipitation: 'Précipitations',
  potential_evaporation: 'Évapotranspiration (ETP)',
  temperature_2m: 'Température',
  niveau_nappe_eau: 'Niveau de nappe (autorégressif)',
}

function hydroLabel(name: string): string {
  return HYDRO_LABELS[name] ?? name.replace(/_/g, ' ')
}

function influenceColor(pct: number): string {
  if (pct > 40) return '#06b6d4'
  if (pct > 15) return '#eab308'
  return '#6b7280'
}

function SectionHeader({ icon: Icon, title, tip, open, onToggle }: {
  icon: React.ElementType; title: string; tip: string; open: boolean; onToggle: () => void
}) {
  return (
    <button onClick={onToggle} className="w-full flex items-center gap-2 py-2 group">
      <Icon className="w-4 h-4 text-accent-cyan shrink-0" />
      <span className="text-sm font-semibold text-text-primary">{title}</span>
      <InfoTip text={tip} />
      <ChevronDown className={`w-3.5 h-3.5 text-text-muted ml-auto transition-transform ${open ? '' : '-rotate-90'}`} />
    </button>
  )
}

// ---------------------------------------------------------------------------
// Section 1: What drives predictions
// ---------------------------------------------------------------------------

function DriversSection({ modelId }: { modelId: string }) {
  const mutation = useFeatureImportance()
  if (!mutation.data && !mutation.isPending && !mutation.isError) mutation.mutate(modelId)
  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError) return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const importance = extractImportance(mutation.data)
  if (!importance) return <p className="text-xs text-text-muted py-4">Aucune donnée d'importance des variables disponible.</p>

  const total = importance.values.reduce((s, v) => s + Math.abs(v), 0)
  const features = importance.features.map(hydroLabel)
  const pcts = importance.values.map(v => total > 0 ? Math.abs(v) / total * 100 : 0)
  const colors = pcts.map(influenceColor)

  return (
    <div className="space-y-3">
      <div className="h-[220px]">
        <Plot
          data={[{
            type: 'bar', orientation: 'h' as const,
            y: features, x: importance.values.map(Math.abs),
            marker: { color: colors },
            hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
          }]}
          layout={{
            ...darkLayout,
            xaxis: { ...darkLayout.xaxis, title: { text: 'Force de corrélation' } },
            yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const },
            margin: { t: 5, r: 20, b: 35, l: 160 },
          }}
          config={plotlyConfig}
          useResizeHandler
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      <div className="bg-bg-hover rounded-lg p-3 space-y-1.5">
        {importance.features.slice(0, 4).map((feat, i) => {
          const pct = pcts[i]
          const label = hydroLabel(feat)
          const isTarget = feat === 'niveau_nappe_eau'
          return (
            <p key={feat} className="text-xs text-text-secondary">
              <span className="font-medium" style={{ color: colors[i] }}>{label}</span>
              {' '}représente <span className="text-text-primary font-mono">{pct.toFixed(0)}%</span> du signal.
              {isTarget && ' Le niveau de nappe est fortement autocorrélé — les niveaux récents prédisent les niveaux futurs.'}
              {feat === 'total_precipitation' && pct > 30 && ' Les précipitations sont un facteur majeur — typique des aquifères superficiels ou alluviaux.'}
              {feat === 'potential_evaporation' && pct > 20 && ' L\'évapotranspiration réduit significativement la recharge pendant les mois chauds.'}
              {feat === 'temperature_2m' && pct > 30 && ' Forte influence de la température — peut indiquer des effets thermiques ou une corrélation avec l\'évapotranspiration.'}
            </p>
          )
        })}
      </div>

      <div className="flex gap-4 text-[10px] text-text-muted">
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded" style={{ backgroundColor: '#06b6d4' }} /> Forte (&gt;40%)</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded" style={{ backgroundColor: '#eab308' }} /> Modérée (15-40%)</span>
        <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded" style={{ backgroundColor: '#6b7280' }} /> Faible (&lt;15%)</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 2: Model quality
// ---------------------------------------------------------------------------

function QualitySection({ modelId }: { modelId: string }) {
  const mutation = useResidualAnalysis()
  if (!mutation.data && !mutation.isPending && !mutation.isError) mutation.mutate(modelId)
  if (mutation.isPending) return <LoadingSkeleton />
  if (mutation.isError) return <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate(modelId)} />
  if (!mutation.data) return null

  const d = mutation.data
  const balanced = d.bias_status === 'balanced' || d.bias_status === 'equilibre'
  const normal = d.normality_pvalue != null ? d.normality_pvalue >= 0.05 : null
  const acfOk = d.acf_lag1 != null ? Math.abs(d.acf_lag1) < 0.3 : null
  const direction = d.mean_error < 0 ? 'surestime' : 'sous-estime'

  return (
    <div className="space-y-3">
      <div className={`flex items-center gap-3 p-3 rounded-lg border ${balanced ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-amber-500/10 border-amber-500/30'}`}>
        {balanced ? <CheckCircle className="w-6 h-6 text-emerald-400 shrink-0" /> : <AlertTriangle className="w-6 h-6 text-amber-400 shrink-0" />}
        <div>
          <p className={`text-sm font-semibold ${balanced ? 'text-emerald-400' : 'text-amber-400'}`}>
            {balanced ? 'Prévisions non biaisées' : 'Biais systématique détecté'}
          </p>
          <p className="text-xs text-text-muted">
            {balanced
              ? 'Les erreurs de prévision sont centrées autour de zéro — pas de sur/sous-estimation systématique.'
              : `Le modèle ${direction} le niveau de nappe de ${Math.abs(d.mean_error).toFixed(3)} m en moyenne.`}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">Erreur moyenne</span>
            <InfoTip text="Erreur moyenne de prévision (biais). Une valeur proche de zéro signifie qu'il n'y a pas de sur- ou sous-estimation systématique. Positif = le modèle sous-estime les niveaux." iconSize={10} />
          </div>
          <p className="text-base font-bold font-mono text-text-primary">{d.mean_error.toFixed(3)} <span className="text-xs text-text-muted font-normal">m</span></p>
        </div>
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">Erreur typique</span>
            <InfoTip text="Écart-type des erreurs de prévision — magnitude typique des erreurs. 95% des erreurs sont dans ±2σ. Comparez avec l'amplitude naturelle du niveau de nappe pour évaluer la signification." iconSize={10} />
          </div>
          <p className="text-base font-bold font-mono text-text-primary">±{d.std_error.toFixed(3)} <span className="text-xs text-text-muted font-normal">m</span></p>
        </div>
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">Erreurs normales</span>
            <InfoTip text="Test de normalité D'Agostino-Pearson sur les résidus. 'Oui' signifie que les erreurs sont aléatoires et bien réparties. 'Non' signifie que le modèle manque certains motifs — cherchez les regroupements dans le graphique d'erreurs ci-dessous." iconSize={10} />
          </div>
          <p className="text-base font-bold">
            {normal === null ? <span className="text-text-muted">?</span> : normal ? <span className="text-emerald-400">Oui</span> : <span className="text-amber-400">Non</span>}
          </p>
        </div>
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-1">
            <span className="text-[10px] text-text-muted uppercase">Indépendantes</span>
            <InfoTip text="Autocorrélation au lag 1 (ACF₁). Faible (< 0.3) signifie que les erreurs successives sont indépendantes — bon. Élevée signifie que les erreurs sont corrélées : si le modèle se trompe aujourd'hui, il se trompera probablement demain aussi, suggérant des dynamiques manquantes." iconSize={10} />
          </div>
          <p className="text-base font-bold">
            {acfOk === null ? <span className="text-text-muted">?</span> : acfOk ? <span className="text-emerald-400">Oui</span> : <span className="text-amber-400">Non</span>}
          </p>
          {d.acf_lag1 != null && <p className="text-[10px] text-text-muted mt-0.5">ACF₁ = {d.acf_lag1.toFixed(3)}</p>}
        </div>
      </div>

      {d.residuals && d.dates && (
        <div>
          <p className="text-xs text-text-muted mb-1">Erreur au cours du temps — idéalement répartie uniformément autour de zéro</p>
          <div className="h-[200px]">
            <Plot
              data={[{
                type: 'scatter', mode: 'markers',
                x: d.dates, y: d.residuals,
                marker: { color: '#f43f5e', size: 3, opacity: 0.6 },
                hovertemplate: '%{x|%d/%m/%Y}<br>Erreur : %{y:.4f} m<extra></extra>',
              }]}
              layout={{
                ...darkLayout,
                margin: { t: 5, r: 20, b: 30, l: 50 },
                height: 200,
                xaxis: { ...darkLayout.xaxis },
                yaxis: { ...darkLayout.yaxis, title: { text: 'Erreur (m)' } },
                shapes: [{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0, y1: 0, line: { color: 'rgba(255,255,255,0.15)', dash: 'dash', width: 1 } }],
              }}
              config={plotlyConfig}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 3: Aquifer behavior (temporal memory + seasonality)
// ---------------------------------------------------------------------------

function BehaviorSection({ modelId }: { modelId: string }) {
  const lagMutation = useLagImportance()
  const seasonMutation = useSeasonalityAnalysis()

  if (!lagMutation.data && !lagMutation.isPending && !lagMutation.isError) lagMutation.mutate(modelId)
  if (!seasonMutation.data && !seasonMutation.isPending && !seasonMutation.isError) seasonMutation.mutate(modelId)

  const lagLoading = lagMutation.isPending
  const seasonLoading = seasonMutation.isPending

  const lag = lagMutation.data
  const season = seasonMutation.data

  const memoryHorizon = lag?.significant_lags?.length
    ? Math.max(...lag.significant_lags)
    : null

  const periodLabels: Record<number, string> = { 7: 'Hebdomadaire', 30: 'Mensuel', 90: 'Trimestriel', 365: 'Annuel' }

  return (
    <div className="space-y-4">
      {/* Response time KPI */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="bg-bg-hover rounded-lg p-3 border border-accent-cyan/20">
          <div className="flex items-center gap-1 mb-1">
            <Clock className="w-3.5 h-3.5 text-accent-cyan" />
            <span className="text-[10px] text-text-muted uppercase">Mémoire de l'aquifère</span>
            <InfoTip text="Nombre de jours de données passées qui influencent significativement le niveau actuel. Mémoire longue = aquifère lent et profond. Mémoire courte = réponse rapide (alluvial, superficiel). Dérivée de l'analyse d'autocorrélation (ACF)." iconSize={10} />
          </div>
          {lagLoading ? (
            <div className="h-6 bg-white/5 rounded animate-pulse mt-1" />
          ) : memoryHorizon != null ? (
            <p className="text-lg font-bold text-accent-cyan">{memoryHorizon} <span className="text-xs text-text-muted font-normal">jours</span></p>
          ) : (
            <p className="text-sm text-text-muted">—</p>
          )}
          {memoryHorizon != null && (
            <p className="text-[10px] text-text-muted mt-0.5">
              {memoryHorizon > 180 ? 'Aquifère profond ou captif avec un temps de réponse très long.' :
               memoryHorizon > 60 ? 'Réponse modérée — typique des aquifères sédimentaires ou semi-captifs.' :
               'Réponse rapide — typique des aquifères alluviaux ou superficiels.'}
            </p>
          )}
        </div>

        <div className="bg-bg-hover rounded-lg p-3 border border-amber-500/20">
          <div className="flex items-center gap-1 mb-1">
            <Sun className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[10px] text-text-muted uppercase">Cycles détectés</span>
            <InfoTip text="Motifs périodiques significatifs dans le signal du niveau de nappe, détectés par analyse spectrale FFT. Les cycles annuels sont attendus pour la plupart des aquifères. Les cycles mensuels peuvent indiquer des influences marégraphiques ou de pompage." iconSize={10} />
          </div>
          {seasonLoading ? (
            <div className="h-6 bg-white/5 rounded animate-pulse mt-1" />
          ) : season?.detected_periods?.length ? (
            <div className="flex flex-wrap gap-1.5 mt-1">
              {season.detected_periods.map(p => (
                <span key={p} className="px-2 py-0.5 bg-amber-500/15 text-amber-400 text-xs rounded-md font-medium">
                  {periodLabels[p] ?? `${p}d`}
                  {season.period_strengths?.[String(p)] != null && (
                    <span className="text-amber-400/60 ml-1">({season.period_strengths[String(p)].toFixed(0)}x)</span>
                  )}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-sm text-text-muted mt-1">Aucun détecté</p>
          )}
        </div>
      </div>

      {/* ACF chart */}
      {lag && (
        <div className="h-[200px]">
          <Plot
            data={[{
              type: 'bar',
              x: lag.lags, y: lag.autocorrelations,
              marker: { color: lag.lags.map(l => lag.significant_lags?.includes(l) ? '#06b6d4' : 'rgba(6,182,212,0.15)') },
              hovertemplate: 'Jour -%{x}<br>Autocorrélation : %{y:.3f}<extra></extra>',
            }]}
            layout={{
              ...darkLayout,
              height: 200,
              xaxis: { ...darkLayout.xaxis, title: { text: 'Décalage (jours)' } },
              yaxis: { ...darkLayout.yaxis, title: { text: 'ACF' } },
              margin: { t: 5, r: 20, b: 35, l: 50 },
            }}
            config={plotlyConfig}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}

      {/* Variance decomposition */}
      {season?.variance_trend != null && season.variance_seasonal != null && season.variance_residual != null && (
        <div className="bg-bg-hover rounded-lg p-3">
          <div className="flex items-center gap-1 mb-2">
            <span className="text-xs font-medium text-text-secondary">Décomposition du signal</span>
            <InfoTip text="La décomposition STL sépare le niveau de nappe en 3 composantes : tendance long terme (changement climatique, occupation des sols), cycle saisonnier (recharge/décharge annuelle) et bruit résiduel (événements imprévisibles). Un aquifère bien modélisé a un bruit résiduel faible." iconSize={10} />
          </div>
          <div className="h-6 rounded-lg overflow-hidden flex">
            {[
              { pct: season.variance_trend!, color: '#06b6d4', label: 'Tendance' },
              { pct: season.variance_seasonal!, color: '#8b5cf6', label: 'Saisonnier' },
              { pct: season.variance_residual!, color: '#f43f5e', label: 'Bruit' },
            ].map(({ pct, color, label }) => (
              <div key={label} className="flex items-center justify-center text-[9px] font-semibold"
                style={{ width: `${pct}%`, backgroundColor: color, color: pct > 10 ? '#0f172a' : 'transparent' }}>
                {pct > 10 ? `${label} ${pct.toFixed(0)}%` : ''}
              </div>
            ))}
          </div>
          <div className="flex gap-3 mt-1.5 text-[10px] text-text-muted">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded" style={{ backgroundColor: '#06b6d4' }} />Tendance {season.variance_trend!.toFixed(0)}%</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded" style={{ backgroundColor: '#8b5cf6' }} />Saisonnier {season.variance_seasonal!.toFixed(0)}%</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded" style={{ backgroundColor: '#f43f5e' }} />Bruit {season.variance_residual!.toFixed(0)}%</span>
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 4: Expert tools (collapsed by default)
// ---------------------------------------------------------------------------

type ExpertMethod = 'permutation' | 'shap' | 'gradients'

function ExpertSection({ modelId }: { modelId: string }) {
  const [method, setMethod] = useState<ExpertMethod>('permutation')

  const METHODS: { key: ExpertMethod; label: string; tip: string }[] = [
    { key: 'permutation', label: 'Permutation', tip: 'Mesure la dégradation de la prévision lorsque chaque variable est mélangée aléatoirement. Indépendant du modèle, fiable mais lent.' },
    { key: 'shap', label: 'SHAP', tip: 'Valeurs de Shapley issues de la théorie des jeux — distribue équitablement la prévision entre toutes les variables d\'entrée. Tient compte des interactions entre variables.' },
    { key: 'gradients', label: 'Gradients Intégrés', tip: 'Suit le chemin du gradient depuis une référence (entrée nulle) jusqu\'à l\'entrée réelle. Montre les pas de temps et variables auxquels le réseau de neurones est le plus sensible.' },
  ]

  return (
    <div className="space-y-3">
      <div className="flex gap-1">
        {METHODS.map(m => (
          <button key={m.key} onClick={() => setMethod(m.key)} title={m.tip}
            className={`px-2.5 py-1 text-[11px] font-medium rounded transition-colors ${method === m.key ? 'bg-white/10 text-text-primary' : 'text-text-muted hover:text-text-primary hover:bg-white/5'}`}>
            {m.label}
          </button>
        ))}
      </div>
      {method === 'permutation' && <PermutationView modelId={modelId} />}
      {method === 'shap' && <ShapView modelId={modelId} />}
      {method === 'gradients' && <GradientsView modelId={modelId} />}
    </div>
  )
}

function PermutationView({ modelId }: { modelId: string }) {
  const m = usePermutationImportance()
  if (!m.data && !m.isPending && !m.isError) m.mutate({ model_id: modelId, n_permutations: 3 })
  if (m.isPending) return <LoadingSkeleton />
  if (m.isError) return <ErrorState message={(m.error as Error).message} onRetry={() => m.mutate({ model_id: modelId })} />
  const imp = m.data ? extractImportance(m.data) : null
  if (!imp) return <p className="text-xs text-text-muted py-4">Aucune donnée.</p>
  return (
    <div className="space-y-2">
      <div className="h-[220px]">
        <Plot
          data={[{ type: 'bar', orientation: 'h' as const, y: imp.features.map(hydroLabel), x: imp.values, marker: { color: '#f59e0b' } }]}
          layout={{ ...darkLayout, margin: { t: 5, r: 20, b: 35, l: 160 }, xaxis: { ...darkLayout.xaxis, title: { text: 'Importance' } }, yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const } }}
          config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
        />
      </div>
      <p className="text-[10px] text-text-muted">Plus élevé = le modèle s'appuie davantage sur cette variable. Si mélanger une variable ne change presque pas les prévisions, le modèle n'en a pas besoin.</p>
    </div>
  )
}

function ShapView({ modelId }: { modelId: string }) {
  const m = useShapAnalysis()
  if (!m.data && !m.isPending && !m.isError) m.mutate({ model_id: modelId })
  if (m.isPending) return <LoadingSkeleton />
  if (m.isError) return <ErrorState message={(m.error as Error).message} onRetry={() => m.mutate({ model_id: modelId })} />
  const imp = m.data ? extractImportance(m.data) : null
  if (!imp) return <p className="text-xs text-text-muted py-4">Aucune donnée.</p>
  return (
    <div className="space-y-2">
      <div className="h-[220px]">
        <Plot
          data={[{ type: 'bar', orientation: 'h' as const, y: imp.features.map(hydroLabel), x: imp.values, marker: { color: imp.values.map(v => v >= 0 ? '#8b5cf6' : '#f43f5e') } }]}
          layout={{ ...darkLayout, margin: { t: 5, r: 20, b: 35, l: 160 }, xaxis: { ...darkLayout.xaxis, title: { text: 'Valeur SHAP moyenne' } }, yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const } }}
          config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
        />
      </div>
      <p className="text-[10px] text-text-muted">SHAP positif = pousse la prévision vers le haut, négatif = vers le bas. Le signe et la magnitude indiquent la direction et la force de l'influence de chaque variable.</p>
    </div>
  )
}

function GradientsView({ modelId }: { modelId: string }) {
  const m = useGradientAnalysis()
  if (!m.data && !m.isPending && !m.isError) m.mutate({ model_id: modelId, method: 'integrated_gradients' })
  if (m.isPending) return <LoadingSkeleton />
  if (m.isError) return <ErrorState message={(m.error as Error).message} onRetry={() => m.mutate({ model_id: modelId, method: 'integrated_gradients' })} />
  if (!m.data) return null

  const hasTemporal = m.data.temporal_importance && m.data.temporal_importance.length > 0
  const imp = extractImportance(m.data)

  return (
    <div className="space-y-3">
      {hasTemporal && (
        <div className="h-[200px]">
          <Plot
            data={[{
              type: 'scatter', mode: 'lines',
              x: m.data.temporal_importance!.map((_, i) => i - m.data.temporal_importance!.length),
              y: m.data.temporal_importance!,
              line: { color: '#10b981', width: 1.5 },
              fill: 'tozeroy' as const,
              fillcolor: 'rgba(16,185,129,0.1)',
              hovertemplate: 'Jour %{x}<br>Attribution : %{y:.4f}<extra></extra>',
            }]}
            layout={{
              ...darkLayout, height: 200,
              margin: { t: 5, r: 20, b: 35, l: 50 },
              xaxis: { ...darkLayout.xaxis, title: { text: 'Jours avant la prévision' } },
              yaxis: { ...darkLayout.yaxis, title: { text: 'Sensibilité' } },
            }}
            config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}
      {imp && (
        <div className="h-[180px]">
          <Plot
            data={[{ type: 'bar', orientation: 'h' as const, y: imp.features.map(hydroLabel), x: imp.values, marker: { color: '#10b981' } }]}
            layout={{ ...darkLayout, margin: { t: 5, r: 20, b: 35, l: 160 }, xaxis: { ...darkLayout.xaxis, title: { text: 'Attribution' } }, yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const } }}
            config={plotlyConfig} useResizeHandler style={{ width: '100%', height: '100%' }}
          />
        </div>
      )}
      <p className="text-[10px] text-text-muted">Graphique temporel : à quels jours passés le modèle est le plus sensible. Graphique des variables : quelles variables contribuent le plus au signal du gradient.</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function ExplainabilityPanel({ modelId, className = '' }: Props) {
  const [openSections, setOpenSections] = useState<Set<Section>>(new Set(['drivers', 'quality', 'behavior']))

  const toggle = (s: Section) => setOpenSections(prev => {
    const next = new Set(prev)
    next.has(s) ? next.delete(s) : next.add(s)
    return next
  })

  const sections: { key: Section; icon: React.ElementType; title: string; tip: string }[] = [
    { key: 'drivers', icon: TrendingUp, title: 'Ce qui pilote le niveau de nappe', tip: 'Importance des variables basée sur la corrélation — mesure la force de corrélation entre chaque variable météorologique (précipitations, température, ETP) et le niveau de nappe. Plus élevé = le modèle s\'appuie davantage sur cette variable pour ses prévisions.' },
    { key: 'quality', icon: Activity, title: 'Qualité des prévisions', tip: 'Analyse des résidus — vérifie si les erreurs de prévision sont aléatoires (bon) ou systématiques (mauvais). Examine le biais, l\'amplitude des erreurs, la normalité et l\'indépendance temporelle des résidus.' },
    { key: 'behavior', icon: Clock, title: 'Réponse de l\'aquifère', tip: 'Combine l\'analyse de la mémoire temporelle (ACF — sur quelle durée le niveau de nappe dépend de lui-même) et la détection de saisonnalité (FFT — quels cycles périodiques existent dans le signal).' },
    { key: 'expert', icon: Beaker, title: 'Méthodes expertes', tip: 'Méthodes avancées d\'interprétabilité ML : Permutation Importance (indépendant du modèle), valeurs SHAP (théorie des jeux), Gradients Intégrés (sensibilité du réseau de neurones). Elles offrent des perspectives alternatives sur l\'importance des variables.' },
  ]

  return (
    <div className={`bg-bg-card rounded-xl border border-white/5 p-4 space-y-1 ${className}`}>
      <h3 className="text-sm font-semibold text-text-primary mb-2">Explicabilité du modèle</h3>

      {sections.map(({ key, icon, title, tip }) => (
        <div key={key} className="border-t border-white/5 pt-1">
          <SectionHeader icon={icon} title={title} tip={tip} open={openSections.has(key)} onToggle={() => toggle(key)} />
          {openSections.has(key) && (
            <div className="pb-3 pt-1">
              {key === 'drivers' && <DriversSection modelId={modelId} />}
              {key === 'quality' && <QualitySection modelId={modelId} />}
              {key === 'behavior' && <BehaviorSection modelId={modelId} />}
              {key === 'expert' && <ExpertSection modelId={modelId} />}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
