import Plot from 'react-plotly.js'
import { useTranslation } from 'react-i18next'

interface Props {
  diagnostics: Record<string, unknown>
}

function getTests(t: (k: string) => string) {
  return [
    { key: 'durbin_watson', label: t('pastas.diagnostics.durbinWatson'), format: (v: number) => v.toFixed(2), good: (v: number) => v > 1.5 && v < 2.5, tooltip: t('pastas.diagnostics.dwTip') },
    { key: 'jarque_bera_pvalue', label: t('pastas.diagnostics.jbP'), format: (v: number) => v.toFixed(3), good: (v: number) => v > 0.05, tooltip: t('pastas.diagnostics.jbTip') },
    { key: 'shapiro_wilk_pvalue', label: t('pastas.diagnostics.swP'), format: (v: number) => v.toFixed(3), good: (v: number) => v > 0.05, tooltip: t('pastas.diagnostics.swTip') },
    { key: 'ljung_box_p_lag10', label: t('pastas.diagnostics.ljungP'), format: (v: number) => v.toFixed(3), good: (v: number) => v > 0.05, tooltip: t('pastas.diagnostics.ljungTip') },
    { key: 'skewness', label: t('pastas.diagnostics.skewness'), format: (v: number) => v.toFixed(3), good: (v: number) => Math.abs(v) < 1.0, tooltip: t('pastas.diagnostics.skewTip') },
    { key: 'kurtosis', label: t('pastas.diagnostics.kurtosis'), format: (v: number) => v.toFixed(3), good: (v: number) => Math.abs(v) < 3, tooltip: t('pastas.diagnostics.kurtTip') },
  ]
}

export function DiagnosticsPanel({ diagnostics }: Props) {
  const { t } = useTranslation()
  const TESTS = getTests(t)
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
        {t('pastas.diagnostics.title')}
      </div>
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
                {t('pastas.diagnostics.qqDesc')}
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
                  title: { text: t('pastas.diagnostics.qqTitle'), font: { size: 11 } },
                  margin: { t: 25, r: 10, b: 30, l: 40 },
                  xaxis: { title: { text: t('pastas.diagnostics.qqTheoretical') }, gridcolor: 'rgba(255,255,255,0.05)' },
                  yaxis: { title: { text: t('pastas.diagnostics.qqObserved') }, gridcolor: 'rgba(255,255,255,0.05)' },
                }}
                useResizeHandler className="w-full"
                config={{ displayModeBar: false }}
              />
            </div>
          )
        })()}

        {pacfValues && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <p className="text-[9px] text-text-muted px-1 mb-0.5">{t('pastas.diagnostics.pacfDesc')}</p>
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
                title: { text: t('pastas.diagnostics.pacfTitle'), font: { size: 11 } },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                xaxis: { title: { text: t('pastas.diagnostics.lagDays') }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        )}

        {histCounts && histBins && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2 col-span-2">
            <p className="text-[9px] text-text-muted px-1 mb-0.5">{t('pastas.diagnostics.histDesc')}</p>
            <Plot
              data={[{
                x: histBins.slice(0, -1).map((b, i) => (b + histBins[i + 1]) / 2),
                y: histCounts,
                type: 'bar',
                marker: { color: 'rgba(96,165,250,0.4)', line: { color: '#60a5fa', width: 1 } },
              }]}
              layout={{
                ...chartBase,
                title: { text: t('pastas.diagnostics.histTitle'), font: { size: 11 } },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                height: 180,
                xaxis: { title: { text: t('pastas.diagnostics.residual') }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: { text: t('pastas.diagnostics.count') }, gridcolor: 'rgba(255,255,255,0.05)' },
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
