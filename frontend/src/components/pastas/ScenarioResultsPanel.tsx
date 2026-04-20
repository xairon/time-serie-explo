import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { PastasScenarioResponse } from '@/lib/types'
import type { Layout } from 'plotly.js-dist-min'

interface ScenarioResultsPanelProps {
  result: PastasScenarioResponse
}

const chartLayout: Partial<Layout> = {
  ...darkLayout,
  margin: { t: 20, r: 20, b: 40, l: 60 },
  height: 240,
}

export function ScenarioResultsPanel({ result }: ScenarioResultsPanelProps) {
  const { baseline, scenario, delta, warnings } = result

  return (
    <div className="space-y-4">
      {warnings.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
          <p className="text-xs font-semibold text-yellow-400 mb-1">Warnings</p>
          {warnings.map((w, i) => (
            <p key={i} className="text-xs text-yellow-300">
              {w}
            </p>
          ))}
        </div>
      )}

      {/* Baseline vs Scenario */}
      <div className="bg-bg-primary rounded-lg border border-white/5 p-3">
        <p className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">
          Baseline vs Scenario
        </p>
        <Plot
          data={[
            {
              x: baseline.index,
              y: baseline.values,
              type: 'scatter',
              mode: 'lines',
              name: 'Baseline',
              line: { color: '#6b7280', width: 1.5 },
            },
            {
              x: scenario.index,
              y: scenario.values,
              type: 'scatter',
              mode: 'lines',
              name: 'Scenario',
              line: { color: '#22d3ee', width: 2 },
            },
          ]}
          layout={chartLayout}
          config={plotlyConfig}
          style={{ width: '100%' }}
        />
      </div>

      {/* Delta */}
      {delta?.index?.length > 0 && (
        <div className="bg-bg-primary rounded-lg border border-white/5 p-3">
          <p className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">
            Delta (scenario − baseline)
          </p>
          <Plot
            data={[
              {
                x: delta.index,
                y: delta.values,
                type: 'scatter',
                mode: 'lines',
                fill: 'tozeroy',
                name: 'Delta',
                line: { color: '#f59e0b', width: 1.5 },
                fillcolor: 'rgba(245,158,11,0.15)',
              },
            ]}
            layout={{ ...chartLayout, height: 180 }}
            config={plotlyConfig}
            style={{ width: '100%' }}
          />
        </div>
      )}
    </div>
  )
}
