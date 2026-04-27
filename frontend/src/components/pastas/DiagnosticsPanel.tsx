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
    tooltip: 'Residual autocorrelation. Ideal is around 2.0 (no correlation between successive errors). Close to 0 = the model misses a regular structure in the signal. Close to 4 = over-correction. Without a noise model (ArNoiseModel), DW < 1.5 is expected and does not indicate a bad model.',
  },
  {
    key: 'jarque_bera_pvalue',
    label: 'Jarque-Bera p',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => v > 0.05,
    tooltip: 'Residual normality test. p > 0.05 = Gaussian errors (good sign, random noise). p near 0 = non-normal distribution, common on long series even with a good model.',
  },
  {
    key: 'shapiro_wilk_pvalue',
    label: 'Shapiro-Wilk p',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => v > 0.05,
    tooltip: 'Another normality test, more powerful. p > 0.05 = Gaussian residuals. Often fails on long series even if the model is correct — interpret alongside the QQ plot.',
  },
  {
    key: 'ljung_box_p_lag10',
    label: 'Ljung-Box p (lag 10)',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => v > 0.05,
    tooltip: 'Are residuals independent? p > 0.05 = white noise (the model captured everything). p near 0 = unmodeled structure remains (residual seasonality, trend, etc.).',
  },
  {
    key: 'skewness',
    label: 'Skewness',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => Math.abs(v) < 1.0,
    tooltip: 'Error distribution asymmetry. Close to 0 = symmetric errors (the model does not systematically over- or underestimate). > 0 = right tail (underestimates peaks). < 0 = left tail.',
  },
  {
    key: 'kurtosis',
    label: 'Kurtosis',
    format: (v: number) => v.toFixed(3),
    good: (v: number) => Math.abs(v) < 3,
    tooltip: 'Distribution flatness. Close to 0 = Gaussian shape. High = heavy tails (extreme errors more frequent than expected — poorly modeled events).',
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
        Residual Diagnostics
      </div>
      <p className="text-xs text-text-muted leading-relaxed">
        These tests check whether the model errors (observed minus simulated) behave like random noise.
        If so, the model has captured all the signal structure.
        Otherwise, unexploited information remains (residual seasonality, trend, etc.).
        <span className="text-green-400">Green</span> = OK, <span className="text-red-400">red</span> = room for improvement (common with a simple model).
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
                Each point = a residual quantile vs the theoretical Gaussian distribution. If points follow the red diagonal, errors are Gaussian. Deviations at the extremes = poorly modeled extreme events.
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
                  title: { text: 'QQ Plot', font: { size: 11 } },
                  margin: { t: 25, r: 10, b: 30, l: 40 },
                  xaxis: { title: { text: 'Theoretical' }, gridcolor: 'rgba(255,255,255,0.05)' },
                  yaxis: { title: { text: 'Observed' }, gridcolor: 'rgba(255,255,255,0.05)' },
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
              Partial correlation of residuals at each lag. Bars exceeding the red lines (significance threshold) indicate residual structure at that lag.
              Ideal: all bars within the confidence zone (white noise).
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
                title: { text: 'Partial Autocorrelation (PACF)', font: { size: 11 } },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                xaxis: { title: { text: 'Lag (days)' }, gridcolor: 'rgba(255,255,255,0.05)' },
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
              Error distribution (observed minus simulated). Centered on 0 = no systematic bias. Bell-shaped = Gaussian errors. Asymmetric = the model recurrently overestimates or underestimates.
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
                title: { text: 'Residual Distribution', font: { size: 11 } },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                height: 180,
                xaxis: { title: { text: 'Residual (m)' }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: { text: 'Count' }, gridcolor: 'rgba(255,255,255,0.05)' },
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
