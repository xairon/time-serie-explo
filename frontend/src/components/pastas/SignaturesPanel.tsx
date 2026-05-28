import { useState, useCallback, useEffect, useMemo } from 'react'
import { ChevronDown, Settings2 } from 'lucide-react'
import Plot from 'react-plotly.js'

interface Props {
  signatures: {
    observed: Record<string, number>
    simulated: Record<string, number>
    categories?: Record<string, string[]>
  }
}

const SIGNATURE_TOOLTIPS: Record<string, string> = {
  cv_period_mean: 'Coefficient de variation du niveau moyen par période — mesure la variabilité du niveau moyen entre mois/années.',
  magnitude: 'Plage divisée par le niveau minimum — amplitude globale relative au niveau minimum.',
  interannual_variation: 'Variabilité d\'une année à l\'autre du niveau moyen — élevé = grandes différences annuelles.',
  mean_annual_maximum: 'Moyenne des pics annuels — niveau de hautes eaux typique.',
  parde_seasonality: 'Indice de Pardé — force du schéma saisonnier (1 = pas de saisonnalité, plus haut = plus fort).',
  avg_seasonal_fluctuation: 'Amplitude moyenne intra-annuelle (haut - bas dans chaque année).',
  date_min: 'Date moyenne du minimum annuel (jour de l\'année, moyenne circulaire).',
  date_max: 'Date moyenne du maximum annuel (jour de l\'année, moyenne circulaire).',
  cv_date_min: 'Régularité de la date des basses eaux annuelles ? CV bas = calendrier prévisible.',
  cv_date_max: 'Régularité de la date des hautes eaux annuelles ? CV bas = calendrier prévisible.',
  rise_rate: 'Variation positive moyenne par jour — vitesse de remontée de la nappe (m/j).',
  fall_rate: 'Variation négative moyenne par jour — vitesse de baisse de la nappe (m/j).',
  cv_rise_rate: 'Variabilité de la vitesse de remontée — CV élevé = événements de recharge erratiques.',
  cv_fall_rate: 'Variabilité de la vitesse de baisse — CV élevé = drainage irrégulier.',
  high_pulse_count: "Nombre moyen d'événements de hautes eaux par an (au-dessus du 75e percentile).",
  high_pulse_duration: 'Durée moyenne des événements de hautes eaux (jours).',
  low_pulse_count: "Nombre moyen d'événements de basses eaux par an (en dessous du 25e percentile).",
  low_pulse_duration: 'Durée moyenne des événements de basses eaux (jours).',
  reversals_avg: "Nombre annuel moyen de changements de direction (hausse↔baisse) — mesure la rapidité de réaction.",
  reversals_cv: 'Variabilité du nombre annuel de retournements — comportement régulier vs erratique.',
  colwell_constancy: 'Indice de Colwell : constance du niveau ? 1 = parfaitement constant.',
  colwell_contingency: 'Indice de Colwell : prévisibilité du schéma saisonnier ? Plus élevé = plus prévisible.',
  recession_constant: 'Taux de décroissance exponentielle en période de tarissement — vitesse de drainage de l\'aquifère.',
  recovery_constant: 'Taux de remontée exponentielle en période de récupération — réactivité à la recharge.',
  duration_curve_slope: 'Pente de la courbe de durée du niveau (plage 30-70e percentile) — variabilité du régime.',
  duration_curve_ratio: 'Ratio entre les percentiles bas et hauts sur la courbe de durée.',
  richards_pathlength: 'Longueur de chemin totale de la série (somme des |variations|) — « mouvement » global.',
  baselevel_index: 'Indice de niveau de base — ratio du débit de base au débit total, adapté aux niveaux.',
  baselevel_stability: 'Stabilité du niveau de base dans le temps.',
  bimodality_coefficient: 'Si la distribution du niveau a deux modes (saison sèche/humide).',
  autocorr_time: 'Nombre de jours avant que l\'autocorrélation passe sous le seuil — mémoire/inertie de l\'aquifère.',
}

const CATEGORY_LABELS: Record<string, string> = {
  variability: 'Variabilité',
  seasonality: 'Saisonnalité et calendrier',
  dynamics: 'Dynamique de montée/descente',
  pulses: 'Événements d\'impulsion',
  reversals: 'Retournements',
  predictability: 'Prévisibilité',
  recession: 'Tarissement et récupération',
  duration_curve: 'Courbe de durée',
  structure: 'Propriétés structurelles',
}

const RADAR_PRESETS: Record<string, { label: string; keys: string[] }> = {
  core: {
    label: 'Validation principale',
    keys: [
      'parde_seasonality', 'avg_seasonal_fluctuation',
      'rise_rate', 'fall_rate',
      'recession_constant', 'recovery_constant',
      'autocorr_time', 'duration_curve_slope',
    ],
  },
  seasonal: {
    label: 'Saisonnier',
    keys: [
      'parde_seasonality', 'avg_seasonal_fluctuation',
      'date_min', 'date_max', 'cv_date_min', 'cv_date_max',
      'high_pulse_count', 'low_pulse_count',
    ],
  },
  dynamics: {
    label: 'Dynamique',
    keys: [
      'rise_rate', 'fall_rate', 'cv_rise_rate', 'cv_fall_rate',
      'recession_constant', 'recovery_constant',
      'reversals_avg', 'autocorr_time',
    ],
  },
}

const STORAGE_KEY = 'pastas_radar_config'

interface RadarConfig {
  preset: string | null
  custom: string[]
}

function loadRadarConfig(): RadarConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return JSON.parse(raw) as RadarConfig
  } catch { /* ignore */ }
  return { preset: 'core', custom: [] }
}

function saveRadarConfig(config: RadarConfig) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
  } catch { /* ignore */ }
}

function formatLabel(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

export function SignaturesPanel({ signatures }: Props) {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [radarConfig, setRadarConfig] = useState<RadarConfig>(loadRadarConfig)
  const [showConfig, setShowConfig] = useState(false)

  useEffect(() => { saveRadarConfig(radarConfig) }, [radarConfig])

  const allKeys = useMemo(() => Object.keys(signatures.observed), [signatures.observed])

  const categories = signatures.categories ?? { all: allKeys }
  const categoryEntries = Object.entries(categories).filter(
    ([, sigs]) => sigs.some(s => s in signatures.observed)
  )

  // Radar keys: from preset or custom selection
  const radarKeys = useMemo(() => {
    if (allKeys.length === 0) return []
    if (radarConfig.preset === 'all') return allKeys.filter(k => k in signatures.observed)
    if (radarConfig.preset && RADAR_PRESETS[radarConfig.preset]) {
      return RADAR_PRESETS[radarConfig.preset].keys.filter(k => k in signatures.observed)
    }
    if (radarConfig.custom.length > 0) {
      return radarConfig.custom.filter(k => k in signatures.observed)
    }
    return RADAR_PRESETS.core.keys.filter(k => k in signatures.observed)
  }, [radarConfig, allKeys, signatures.observed])

  const obsValues = radarKeys.map(k => signatures.observed[k] ?? 0)
  const simValues = radarKeys.map(k => signatures.simulated[k] ?? 0)
  const maxVals = radarKeys.map((_, i) => Math.max(Math.abs(obsValues[i]), Math.abs(simValues[i]), 1e-6))
  const obsNorm = obsValues.map((v, i) => v / maxVals[i])
  const simNorm = simValues.map((v, i) => v / maxVals[i])
  const labels = radarKeys.map(formatLabel)

  const toggleCustomSig = useCallback((key: string) => {
    setRadarConfig(prev => {
      const base = prev.preset && prev.preset !== 'all' && RADAR_PRESETS[prev.preset]
        ? RADAR_PRESETS[prev.preset].keys.filter(k => k in signatures.observed)
        : prev.custom.length > 0 ? prev.custom : allKeys
      const current = new Set(prev.custom.length > 0 ? prev.custom : base)
      if (current.has(key)) current.delete(key)
      else current.add(key)
      return { preset: null, custom: [...current] }
    })
  }, [allKeys, signatures.observed])

  const selectPreset = useCallback((preset: string) => {
    setRadarConfig({ preset, custom: [] })
  }, [])

  const isInRadar = useCallback((key: string) => {
    return radarKeys.includes(key)
  }, [radarKeys])

  if (allKeys.length === 0) return null

  return (
    <div className="space-y-3">
      {/* Radar preset bar */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-text-muted uppercase tracking-wide shrink-0">Radar :</span>
        <div className="flex flex-wrap gap-1">
          {Object.entries(RADAR_PRESETS).map(([key, { label }]) => (
            <button
              key={key}
              onClick={() => selectPreset(key)}
              className={`px-2 py-0.5 rounded-md text-[10px] font-medium border transition-colors ${
                radarConfig.preset === key
                  ? 'bg-accent-cyan/20 border-accent-cyan/30 text-accent-cyan'
                  : 'border-white/10 text-text-muted hover:text-text-secondary'
              }`}
            >
              {label}
            </button>
          ))}
          <button
            onClick={() => selectPreset('all')}
            className={`px-2 py-0.5 rounded-md text-[10px] font-medium border transition-colors ${
              radarConfig.preset === 'all'
                ? 'bg-accent-cyan/20 border-accent-cyan/30 text-accent-cyan'
                : 'border-white/10 text-text-muted hover:text-text-secondary'
            }`}
          >
            Tout ({allKeys.length})
          </button>
          {!radarConfig.preset && (
            <span className="px-2 py-0.5 rounded-md text-[10px] font-medium border bg-purple-500/15 border-purple-500/25 text-purple-400">
              Personnalisé ({radarKeys.length})
            </span>
          )}
        </div>
        <button
          onClick={() => setShowConfig(!showConfig)}
          className={`p-1 rounded transition-colors ${showConfig ? 'bg-accent-cyan/10 text-accent-cyan' : 'text-text-muted hover:text-text-secondary'}`}
          title="Configurer les signatures du radar"
        >
          <Settings2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Signature picker */}
      {showConfig && (
        <div className="bg-bg-card rounded-lg border border-white/5 p-3 space-y-2">
          {categoryEntries.map(([cat, sigs]) => {
            const validSigs = sigs.filter(s => s in signatures.observed)
            if (validSigs.length === 0) return null
            return (
              <div key={cat}>
                <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1">
                  {CATEGORY_LABELS[cat] ?? cat}
                </div>
                <div className="flex flex-wrap gap-1">
                  {validSigs.map(k => (
                    <button
                      key={k}
                      onClick={() => toggleCustomSig(k)}
                      className={`px-1.5 py-0.5 rounded text-[10px] border transition-colors ${
                        isInRadar(k)
                          ? 'bg-accent-cyan/15 border-accent-cyan/30 text-accent-cyan'
                          : 'border-white/10 text-text-muted hover:text-text-secondary'
                      }`}
                      title={SIGNATURE_TOOLTIPS[k]}
                    >
                      {formatLabel(k)}
                    </button>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Radar chart */}
      {radarKeys.length >= 3 && (
        <div className="bg-bg-card rounded-lg border border-white/5 p-3">
          <Plot
            data={[
              {
                type: 'scatterpolar',
                r: [...obsNorm, obsNorm[0]],
                theta: [...labels, labels[0]],
                name: 'Observé',
                fill: 'toself',
                fillcolor: 'rgba(96,165,250,0.1)',
                line: { color: '#60a5fa' },
              },
              {
                type: 'scatterpolar',
                r: [...simNorm, simNorm[0]],
                theta: [...labels, labels[0]],
                name: 'Simulé',
                fill: 'toself',
                fillcolor: 'rgba(249,115,22,0.1)',
                line: { color: '#f97316' },
              },
            ]}
            layout={{
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#9ca3af', size: 9 },
              margin: { t: 30, r: 80, b: 30, l: 80 },
              height: 400,
              polar: {
                bgcolor: 'transparent',
                radialaxis: { visible: true, gridcolor: 'rgba(255,255,255,0.05)', range: [0, 1] },
                angularaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
              },
              legend: { orientation: 'h', y: -0.05, font: { size: 10 } },
            }}
            useResizeHandler
            className="w-full"
            config={{ displayModeBar: false }}
          />
        </div>
      )}

      {/* Table grouped by category */}
      <div className="bg-bg-card rounded-lg border border-white/5 overflow-hidden">
        {/* Category filter for table */}
        <div className="flex flex-wrap gap-1 px-3 pt-2 pb-1">
          <button
            onClick={() => setSelectedCategory(null)}
            className={`px-2 py-0.5 rounded-md text-[10px] font-medium border transition-colors ${
              selectedCategory === null
                ? 'bg-accent-cyan/20 border-accent-cyan/30 text-accent-cyan'
                : 'border-white/10 text-text-muted hover:text-text-secondary'
            }`}
          >
            Tout
          </button>
          {categoryEntries.map(([cat]) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(selectedCategory === cat ? null : cat)}
              className={`px-2 py-0.5 rounded-md text-[10px] font-medium border transition-colors ${
                selectedCategory === cat
                  ? 'bg-accent-cyan/20 border-accent-cyan/30 text-accent-cyan'
                  : 'border-white/10 text-text-muted hover:text-text-secondary'
              }`}
            >
              {CATEGORY_LABELS[cat] ?? cat}
            </button>
          ))}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-text-muted border-b border-white/5">
                <th className="text-left px-3 py-2">Signature</th>
                <th className="text-right px-3 py-2">Observé</th>
                <th className="text-right px-3 py-2">Simulé</th>
                <th className="text-right px-3 py-2" title="Ratio simulé / observé. Proche de 1,0 = bonne reproduction. Vert = écart < 10%.">
                  Ratio <span className="opacity-50 cursor-help">ⓘ</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {categoryEntries
                .filter(([cat]) => !selectedCategory || cat === selectedCategory)
                .map(([cat, sigs]) => {
                  const validSigs = sigs.filter(s => s in signatures.observed)
                  if (validSigs.length === 0) return null
                  return (
                    <CategoryGroup
                      key={cat}
                      category={CATEGORY_LABELS[cat] ?? cat}
                      sigKeys={validSigs}
                      observed={signatures.observed}
                      simulated={signatures.simulated}
                      radarKeys={radarKeys}
                    />
                  )
                })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function CategoryGroup({ category, sigKeys, observed, simulated, radarKeys }: {
  category: string
  sigKeys: string[]
  observed: Record<string, number>
  simulated: Record<string, number>
  radarKeys: string[]
}) {
  const [collapsed, setCollapsed] = useState(false)
  return (
    <>
      <tr className="bg-bg-hover/50 cursor-pointer" onClick={() => setCollapsed(!collapsed)}>
        <td colSpan={4} className="px-3 py-1.5 text-[10px] font-semibold text-text-muted uppercase tracking-wider">
          <div className="flex items-center gap-1">
            <ChevronDown className={`w-3 h-3 transition-transform ${collapsed ? '-rotate-90' : ''}`} />
            {category}
          </div>
        </td>
      </tr>
      {!collapsed && sigKeys.map(k => {
        const obs = observed[k]
        const sim = simulated[k] ?? 0
        const ratio = obs !== 0 ? sim / obs : null
        const ratioGood = ratio !== null && Math.abs(ratio - 1) < 0.1
        const inRadar = radarKeys.includes(k)
        return (
          <tr key={k} className="border-b border-white/5 hover:bg-bg-hover" title={SIGNATURE_TOOLTIPS[k] ?? ''}>
            <td className="px-3 py-1.5 text-text-secondary cursor-help">
              <span className="flex items-center gap-1.5">
                {inRadar && <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan shrink-0" title="Affiché dans le radar" />}
                {formatLabel(k)}
                {SIGNATURE_TOOLTIPS[k] && <span className="opacity-30">ⓘ</span>}
              </span>
            </td>
            <td className="px-3 py-1.5 text-right font-mono text-text-primary">{obs.toFixed(4)}</td>
            <td className="px-3 py-1.5 text-right font-mono text-text-primary">{sim.toFixed(4)}</td>
            <td className={`px-3 py-1.5 text-right font-mono ${ratioGood ? 'text-green-400' : 'text-text-muted'}`}>
              {ratio !== null ? ratio.toFixed(3) : '—'}
            </td>
          </tr>
        )
      })}
    </>
  )
}
