import Plot from 'react-plotly.js'
import type { Data } from 'plotly.js-dist-min'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface FeatureFingerprintsProps {
  fingerprints: Record<string, unknown>[]
}

const COLORS = [
  '#06b6d4', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981',
  '#ec4899', '#3b82f6', '#f97316', '#14b8a6', '#a855f7',
]

const FEATURE_LABELS: Record<string, string> = {
  mean: 'Mean',
  std: 'Std',
  trend: 'Trend',
  seasonality: 'Seasonality',
  autocorr_365: 'Autocorr 365d',
  wet_dry_ratio: 'Wet/Dry Ratio',
}

const FEATURE_ORDER = ['mean', 'std', 'trend', 'seasonality', 'autocorr_365', 'wet_dry_ratio']

export function FeatureFingerprints({ fingerprints }: FeatureFingerprintsProps) {
  const fps = fingerprints as {
    cluster_id: number
    features: Record<string, number>
    features_raw: Record<string, number>
  }[]

  if (fps.length === 0) return null

  const theta = FEATURE_ORDER.map((f) => FEATURE_LABELS[f] ?? f)
  const thetaClosed = [...theta, theta[0]]

  const traces: Data[] = fps.map((fp, i) => {
    const r = FEATURE_ORDER.map((f) => fp.features[f] ?? 0)
    const rClosed = [...r, r[0]]
    const rawVals = FEATURE_ORDER.map((f) => fp.features_raw[f]?.toFixed(3) ?? 'N/A')
    const rawClosed = [...rawVals, rawVals[0]]

    return {
      type: 'scatterpolar',
      r: rClosed,
      theta: thetaClosed,
      fill: 'toself',
      fillcolor: COLORS[i % COLORS.length] + '1A',
      line: { color: COLORS[i % COLORS.length], width: 2 },
      name: `Cluster ${fp.cluster_id}`,
      customdata: rawClosed,
      hovertemplate: '%{theta}: %{r:.2f}<br>Raw: %{customdata}<extra>Cluster ' + fp.cluster_id + '</extra>',
    } as Data
  })

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4">
      <h3 className="text-text-primary text-sm font-medium mb-1">Feature Fingerprints</h3>
      <p className="text-text-muted text-xs mb-3">Radar chart comparing 6 time-series features across clusters (normalized 0-1). Hover for raw values.</p>
      <Plot
        data={traces}
        layout={{
          ...darkLayout,
          margin: { l: 60, r: 60, t: 30, b: 30 },
          height: 400,
          polar: {
            bgcolor: 'transparent',
            radialaxis: {
              range: [0, 1],
              gridcolor: 'rgba(255,255,255,0.08)',
              tickfont: { size: 9 },
            },
            angularaxis: {
              gridcolor: 'rgba(255,255,255,0.08)',
              tickfont: { size: 10 },
            },
          },
          legend: {
            ...darkLayout.legend,
            orientation: 'h',
            y: -0.1,
            font: { size: 10 },
          },
          showlegend: true,
        }}
        config={plotlyConfig}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
