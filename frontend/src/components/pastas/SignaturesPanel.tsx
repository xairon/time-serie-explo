import Plot from 'react-plotly.js'

interface Props {
  signatures: { observed: Record<string, number>; simulated: Record<string, number> }
}

export function SignaturesPanel({ signatures }: Props) {
  const keys = Object.keys(signatures.observed)
  if (keys.length === 0) return null

  const obsValues = keys.map(k => signatures.observed[k])
  const simValues = keys.map(k => signatures.simulated[k] ?? 0)

  // Normalize each axis to [0, 1] for radar comparability
  const maxVals = keys.map((_, i) => Math.max(Math.abs(obsValues[i]), Math.abs(simValues[i]), 1e-6))
  const obsNorm = obsValues.map((v, i) => v / maxVals[i])
  const simNorm = simValues.map((v, i) => v / maxVals[i])

  // Readable labels
  const labels = keys.map(k =>
    k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
  )

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
        Hydrological Signatures
      </div>

      <div className="bg-bg-card rounded-lg border border-white/5 p-3">
        <Plot
          data={[
            {
              type: 'scatterpolar',
              r: [...obsNorm, obsNorm[0]],
              theta: [...labels, labels[0]],
              name: 'Observed',
              fill: 'toself',
              fillcolor: 'rgba(96,165,250,0.1)',
              line: { color: '#60a5fa' },
            },
            {
              type: 'scatterpolar',
              r: [...simNorm, simNorm[0]],
              theta: [...labels, labels[0]],
              name: 'Simulated',
              fill: 'toself',
              fillcolor: 'rgba(249,115,22,0.1)',
              line: { color: '#f97316' },
            },
          ]}
          layout={{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#9ca3af', size: 9 },
            margin: { t: 30, r: 60, b: 30, l: 60 },
            height: 350,
            polar: {
              bgcolor: 'transparent',
              radialaxis: { visible: true, gridcolor: 'rgba(255,255,255,0.05)', range: [0, 1] },
              angularaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
            },
            legend: { orientation: 'h', y: -0.1, font: { size: 10 } },
          }}
          useResizeHandler
          className="w-full"
          config={{ displayModeBar: false }}
        />
      </div>

      {/* Raw values table */}
      <div className="bg-bg-card rounded-lg border border-white/5 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-text-muted border-b border-white/5">
                <th className="text-left px-3 py-2">Signature</th>
                <th className="text-right px-3 py-2">Observed</th>
                <th className="text-right px-3 py-2">Simulated</th>
                <th className="text-right px-3 py-2">Ratio</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((k, i) => {
                const ratio = obsValues[i] !== 0 ? simValues[i] / obsValues[i] : null
                return (
                  <tr key={k} className="border-b border-white/5 hover:bg-bg-hover">
                    <td className="px-3 py-1.5 text-text-secondary">{labels[i]}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-text-primary">
                      {obsValues[i].toFixed(4)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-text-primary">
                      {simValues[i].toFixed(4)}
                    </td>
                    <td
                      className={`px-3 py-1.5 text-right font-mono ${
                        ratio !== null && Math.abs(ratio - 1) < 0.1
                          ? 'text-green-400'
                          : 'text-text-muted'
                      }`}
                    >
                      {ratio !== null ? ratio.toFixed(3) : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
