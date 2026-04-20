import Plot from 'react-plotly.js'

interface Props {
  diagnostics: Record<string, unknown>
}

const TESTS = [
  {
    key: 'durbin_watson',
    label: 'Durbin-Watson',
    format: (v: number) => v.toFixed(2),
    good: (v: number) => v > 1.5 && v < 2.5,
    tooltip: 'Mesure l\'autocorrélation des résidus. Idéal ≈ 2.0. Proche de 0 = forte autocorrélation positive (le modèle manque de la structure).',
  },
  {
    key: 'jarque_bera_pvalue',
    label: 'Jarque-Bera p',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => v > 0.05,
    tooltip: 'Test de normalité des résidus. p > 0.05 = résidus gaussiens (bien). p ≈ 0 = résidus non-normaux, courant avec de longues séries.',
  },
  {
    key: 'shapiro_wilk_pvalue',
    label: 'Shapiro-Wilk p',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => v > 0.05,
    tooltip: 'Autre test de normalité, plus puissant. p > 0.05 = résidus gaussiens. Souvent en échec sur de longues séries même si le modèle est correct.',
  },
  {
    key: 'ljung_box_p_lag10',
    label: 'Ljung-Box p (lag 10)',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => v > 0.05,
    tooltip: 'Teste si les résidus sont indépendants (pas d\'autocorrélation). p > 0.05 = bruit blanc. p ≈ 0 = il reste de la structure non captée.',
  },
  {
    key: 'skewness',
    label: 'Asymétrie',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => Math.abs(v) < 0.5,
    tooltip: 'Asymétrie de la distribution des résidus. |valeur| < 0.5 = distribution symétrique (bien).',
  },
  {
    key: 'kurtosis',
    label: 'Kurtosis',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => Math.abs(v) < 1,
    tooltip: 'Aplatissement de la distribution. |valeur| < 1 = forme proche de la gaussienne. Valeur élevée = queues lourdes (valeurs extrêmes).',
  },
]

export function DiagnosticsPanel({ diagnostics }: Props) {
  const qqTheoretical = diagnostics.qq_theoretical as number[] | undefined
  const qqSample = diagnostics.qq_sample as number[] | undefined
  const pacfValues = diagnostics.pacf_values as number[] | undefined
  const confBound = diagnostics.confidence_bound as number | undefined
  const histCounts = diagnostics.hist_counts as number[] | undefined
  const histBins = diagnostics.hist_bins as number[] | undefined

  const chartBase = {
    paper_bgcolor: 'transparent' as const,
    plot_bgcolor: 'transparent' as const,
    font: { color: '#9ca3af', size: 9 },
    height: 200,
    showlegend: false,
  }

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
        Diagnostics des résidus
      </div>
      <p className="text-xs text-text-muted">
        Ces tests vérifient si les erreurs du modèle se comportent comme du bruit aléatoire.
        Vert = OK, rouge = le modèle pourrait être amélioré (normal avec un modèle simple).
      </p>

      <div className="flex flex-wrap gap-2">
        {TESTS.map(t => {
          const value = diagnostics[t.key] as number | undefined
          if (value === undefined || value === null) return null
          const isGood = t.good(value)
          return (
            <div
              key={t.key}
              title={t.tooltip}
              className={`px-2 py-1 rounded-md text-xs border cursor-help ${
                isGood
                  ? 'border-green-500/30 bg-green-500/10 text-green-400'
                  : 'border-red-500/30 bg-red-500/10 text-red-400'
              }`}
            >
              {t.label}: {t.format(value)} {isGood ? '✓' : '✗'}
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-2 gap-3">
        {qqTheoretical && qqSample && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <Plot
              data={[
                { x: qqTheoretical, y: qqSample, type: 'scatter', mode: 'markers',
                  marker: { color: '#60a5fa', size: 3 } },
                { x: [Math.min(...qqTheoretical), Math.max(...qqTheoretical)],
                  y: [Math.min(...qqTheoretical), Math.max(...qqTheoretical)],
                  type: 'scatter', mode: 'lines',
                  line: { color: '#ef4444', dash: 'dash' } },
              ]}
              layout={{
                ...chartBase,
                title: { text: 'QQ Plot (normalité)', font: { size: 11 } },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                xaxis: { title: { text: 'Théorique' }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: { text: 'Observé' }, gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        )}

        {pacfValues && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <Plot
              data={[
                { y: pacfValues, type: 'bar', marker: { color: '#60a5fa' } },
                ...(confBound ? [
                  { y: Array(pacfValues.length).fill(confBound), type: 'scatter' as const, mode: 'lines' as const,
                    line: { color: '#ef4444', dash: 'dash' as const, width: 1 } },
                  { y: Array(pacfValues.length).fill(-confBound), type: 'scatter' as const, mode: 'lines' as const,
                    line: { color: '#ef4444', dash: 'dash' as const, width: 1 } },
                ] : []),
              ]}
              layout={{
                ...chartBase,
                title: { text: 'Autocorrélation partielle (PACF)', font: { size: 11 } },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                xaxis: { title: { text: 'Lag (jours)' }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        )}

        {histCounts && histBins && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2 col-span-2">
            <Plot
              data={[{
                x: histBins.slice(0, -1).map((b, i) => (b + histBins[i + 1]) / 2),
                y: histCounts,
                type: 'bar',
                marker: { color: 'rgba(96,165,250,0.4)', line: { color: '#60a5fa', width: 1 } },
              }]}
              layout={{
                ...chartBase,
                title: { text: 'Distribution des résidus', font: { size: 11 } },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                height: 180,
                xaxis: { title: { text: 'Résidu (m)' }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: { text: 'Nombre' }, gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        )}
      </div>
    </div>
  )
}
