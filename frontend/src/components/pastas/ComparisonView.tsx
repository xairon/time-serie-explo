import Plot from 'react-plotly.js'
import type { PastasCompareModel } from '@/lib/types'

const COLORS = ['#60a5fa', '#f97316', '#34d399', '#a78bfa', '#f43f5e']
const METRIC_KEYS = ['nse', 'kge', 'evp', 'rmse', 'rsq']
const HIGHER_IS_BETTER = new Set(['nse', 'kge', 'evp', 'rsq'])

interface Props {
  models: PastasCompareModel[]
}

export function ComparisonView({ models }: Props) {
  if (models.length === 0) return null

  return (
    <div className="space-y-4">
      {/* Metrics table */}
      <div className="bg-bg-card rounded-lg border border-white/5 overflow-hidden">
        <div className="px-3 py-2 border-b border-white/5">
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">Metrics Comparison</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-text-muted border-b border-white/5">
                <th className="text-left px-3 py-2">Metric</th>
                {models.map((m, i) => (
                  <th key={m.run_id} className="text-right px-3 py-2">
                    <span style={{ color: COLORS[i] }}>{m.name}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRIC_KEYS.map(key => {
                const values = models.map(m => m.metrics[key])
                const valid = values.filter(v => v != null && isFinite(v)) as number[]
                const best = valid.length > 0
                  ? (HIGHER_IS_BETTER.has(key) ? Math.max(...valid) : Math.min(...valid))
                  : null

                return (
                  <tr key={key} className="border-b border-white/5">
                    <td className="px-3 py-1.5 text-text-secondary uppercase">{key}</td>
                    {values.map((v, i) => (
                      <td key={i} className={`px-3 py-1.5 text-right font-mono ${
                        v != null && v === best ? 'text-green-400 font-semibold' : 'text-text-primary'
                      }`}>
                        {v != null ? v.toFixed(4) : '—'}
                      </td>
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Config comparison */}
      <div className="bg-bg-card rounded-lg border border-white/5 overflow-hidden">
        <div className="px-3 py-2 border-b border-white/5">
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">Configuration</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-text-muted border-b border-white/5">
                <th className="text-left px-3 py-2">Param</th>
                {models.map((m, i) => (
                  <th key={m.run_id} className="text-right px-3 py-2" style={{ color: COLORS[i] }}>{m.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {['recharge_type', 'response_type', 'noise_type', 'solver_type'].map(key => (
                <tr key={key} className="border-b border-white/5">
                  <td className="px-3 py-1.5 text-text-secondary">{key.replace('_type', '')}</td>
                  {models.map((m, i) => (
                    <td key={i} className="px-3 py-1.5 text-right text-text-primary">{m.params[key] ?? '—'}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Overlay plot */}
      <div className="bg-bg-card rounded-lg border border-white/5 p-3">
        <div className="text-xs font-semibold text-text-secondary mb-1 uppercase tracking-wide">Simulation Overlay</div>
        <Plot
          data={[
            // Observed from first model (they should all share the same obs)
            {
              x: models[0].observed.index,
              y: models[0].observed.values,
              type: 'scatter',
              mode: 'lines',
              name: 'Observed',
              line: { color: '#6b7280', width: 1 },
            },
            // Each model's simulation
            ...models.map((m, i) => ({
              x: m.simulated.index,
              y: m.simulated.values,
              type: 'scatter' as const,
              mode: 'lines' as const,
              name: m.name,
              line: { color: COLORS[i], width: 2 },
            })),
          ]}
          layout={{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#9ca3af', size: 10 },
            margin: { t: 10, r: 20, b: 30, l: 50 },
            height: 300,
            xaxis: { gridcolor: 'rgba(255,255,255,0.03)' },
            yaxis: { title: { text: 'Head (m)' }, gridcolor: 'rgba(255,255,255,0.05)' },
            legend: { orientation: 'h', y: -0.15, font: { size: 10 } },
          }}
          useResizeHandler
          className="w-full"
          config={{ displayModeBar: false }}
        />
      </div>
    </div>
  )
}
