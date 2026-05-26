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
    tooltip: 'Autocorrélation des résidus. L\'idéal est autour de 2,0 (pas de corrélation entre erreurs successives). Proche de 0 = le modèle manque une structure régulière du signal. Proche de 4 = sur-correction. Sans modèle de bruit (ArNoiseModel), DW < 1,5 est attendu et n\'indique pas un mauvais modèle.',
  },
  {
    key: 'jarque_bera_pvalue',
    label: 'Jarque-Bera p',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => v > 0.05,
    tooltip: 'Test de normalité des résidus. p > 0,05 = erreurs gaussiennes (bon signe, bruit aléatoire). p proche de 0 = distribution non normale, fréquent sur des chroniques longues même avec un bon modèle.',
  },
  {
    key: 'shapiro_wilk_pvalue',
    label: 'Shapiro-Wilk p',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => v > 0.05,
    tooltip: 'Autre test de normalité, plus puissant. p > 0,05 = résidus gaussiens. Échoue souvent sur des chroniques longues même si le modèle est correct — interpréter avec le QQ-plot.',
  },
  {
    key: 'ljung_box_p_lag10',
    label: 'Ljung-Box p (lag 10)',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => v > 0.05,
    tooltip: 'Les résidus sont-ils indépendants ? p > 0,05 = bruit blanc (le modèle a tout capturé). p proche de 0 = il reste une structure non modélisée (saisonnalité résiduelle, tendance, etc.).',
  },
  {
    key: 'skewness',
    label: 'Asymétrie',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => Math.abs(v) < 1.0,
    tooltip: 'Asymétrie de la distribution des erreurs. Proche de 0 = erreurs symétriques (le modèle ne sur- ou sous-estime pas systématiquement). > 0 = queue à droite (sous-estime les pics). < 0 = queue à gauche.',
  },
  {
    key: 'kurtosis',
    label: 'Aplatissement',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => Math.abs(v) < 3,
    tooltip: 'Aplatissement de la distribution. Proche de 0 = forme gaussienne. Élevé = queues lourdes (erreurs extrêmes plus fréquentes qu\'attendu — événements mal modélisés).',
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
      <p className="text-xs text-text-muted leading-relaxed">
        Ces tests vérifient si les erreurs du modèle (observé moins simulé) se comportent comme un bruit aléatoire.
        Si oui, le modèle a capturé toute la structure du signal.
        Sinon, il reste de l'information inexploitée (saisonnalité résiduelle, tendance, etc.).
        <span className="text-green-400">Vert</span> = OK, <span className="text-red-400">rouge</span> = marge d'amélioration (fréquent avec un modèle simple).
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
        {qqTheoretical && qqSample && (() => {
          const n = qqTheoretical.length
          const i25 = Math.floor(n * 0.25)
          const i75 = Math.floor(n * 0.75)
          const slope = (qqTheoretical[i75] !== qqTheoretical[i25])
            ? (qqSample[i75] - qqSample[i25]) / (qqTheoretical[i75] - qqTheoretical[i25])
            : 1
          const intercept = qqSample[i25] - slope * qqTheoretical[i25]
          const xMin = qqTheoretical[0]
          const xMax = qqTheoretical[n - 1]
          return (
            <div className="bg-bg-card rounded-lg border border-white/5 p-2">
              <p className="text-[9px] text-text-muted px-1 mb-0.5">
                Chaque point = un quantile des résidus vs la distribution gaussienne théorique. Si les points suivent la diagonale rouge, les erreurs sont gaussiennes. Les écarts aux extrêmes = événements extrêmes mal modélisés.
              </p>
              <Plot
                data={[
                  { x: qqTheoretical, y: qqSample, type: 'scatter', mode: 'markers',
                    marker: { color: '#60a5fa', size: 3 } },
                  { x: [xMin, xMax],
                    y: [intercept + slope * xMin, intercept + slope * xMax],
                    type: 'scatter', mode: 'lines',
                    line: { color: '#ef4444', dash: 'dash' } },
                ]}
                layout={{
                  ...chartBase,
                  title: { text: 'QQ-plot', font: { size: 11 } },
                  margin: { t: 25, r: 10, b: 30, l: 40 },
                  xaxis: { title: { text: 'Théorique' }, gridcolor: 'rgba(255,255,255,0.05)' },
                  yaxis: { title: { text: 'Observé' }, gridcolor: 'rgba(255,255,255,0.05)' },
                }}
                useResizeHandler className="w-full"
                config={{ displayModeBar: false }}
              />
            </div>
          )
        })()}

        {pacfValues && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <p className="text-[9px] text-text-muted px-1 mb-0.5">
              Corrélation partielle des résidus à chaque décalage. Les barres dépassant les lignes rouges (seuil de significativité) indiquent une structure résiduelle à ce décalage.
              Idéal : toutes les barres dans la zone de confiance (bruit blanc).
            </p>
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
                xaxis: { title: { text: 'Décalage (jours)' }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        )}

        {histCounts && histBins && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2 col-span-2">
            <p className="text-[9px] text-text-muted px-1 mb-0.5">
              Distribution des erreurs (observé moins simulé). Centrée sur 0 = pas de biais systématique. Forme en cloche = erreurs gaussiennes. Asymétrique = le modèle sur- ou sous-estime de manière récurrente.
            </p>
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
