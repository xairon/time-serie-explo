import Plot from 'react-plotly.js'

interface Props {
  diagnostics: Record<string, unknown>
}

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

      {/* Test badges */}
      <div className="flex flex-wrap gap-2">
        <TestBadge label="Durbin-Watson" value={diagnostics.durbin_watson as number}
          format={v => v.toFixed(2)} good={v => v > 1.5 && v < 2.5} />
        <TestBadge label="Jarque-Bera p" value={diagnostics.jarque_bera_pvalue as number}
          format={v => v.toFixed(3)} good={v => v > 0.05} />
        <TestBadge label="Shapiro-Wilk p" value={diagnostics.shapiro_wilk_pvalue as number}
          format={v => v.toFixed(3)} good={v => v > 0.05} />
        <TestBadge label="Ljung-Box p (lag 10)" value={diagnostics.ljung_box_p_lag10 as number}
          format={v => v.toFixed(3)} good={v => v > 0.05} />
        <TestBadge label="Skewness" value={diagnostics.skewness as number}
          format={v => v.toFixed(3)} good={v => Math.abs(v) < 0.5} />
        <TestBadge label="Kurtosis" value={diagnostics.kurtosis as number}
          format={v => v.toFixed(3)} good={v => Math.abs(v) < 1} />
      </div>

      <div className="grid grid-cols-2 gap-3">
        {/* QQ Plot */}
        {qqTheoretical && qqSample && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <Plot
              data={[
                { x: qqTheoretical, y: qqSample, type: 'scatter', mode: 'markers',
                  marker: { color: '#60a5fa', size: 3 }, name: 'Residuals' },
                { x: [Math.min(...qqTheoretical), Math.max(...qqTheoretical)],
                  y: [Math.min(...qqTheoretical), Math.max(...qqTheoretical)],
                  type: 'scatter', mode: 'lines',
                  line: { color: '#ef4444', dash: 'dash' }, name: '1:1' },
              ]}
              layout={{
                ...chartBase,
                title: { text: 'QQ Plot', font: { size: 11 } },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                xaxis: { title: { text: 'Theoretical' }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: { text: 'Sample' }, gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        )}

        {/* PACF */}
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
                title: { text: 'Partial Autocorrelation', font: { size: 11 } },
                margin: { t: 25, r: 10, b: 30, l: 40 },
                xaxis: { title: { text: 'Lag' }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        )}

        {/* Histogram — full width */}
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

function TestBadge({ label, value, format, good }: {
  label: string; value: number | undefined
  format: (v: number) => string; good: (v: number) => boolean
}) {
  if (value === undefined || value === null) return null
  const isGood = good(value)
  return (
    <div className={`px-2 py-1 rounded-md text-xs border ${
      isGood ? 'border-green-500/30 bg-green-500/10 text-green-400'
             : 'border-red-500/30 bg-red-500/10 text-red-400'
    }`}>
      {label}: {format(value)} {isGood ? '✓' : '✗'}
    </div>
  )
}
