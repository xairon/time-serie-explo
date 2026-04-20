import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { PastasFitResponse } from '@/lib/types'
import type { Layout } from 'plotly.js-dist-min'
import { ContributionsChart } from '@/components/pastas/ContributionsChart'

interface FitResultsPanelProps {
  result: PastasFitResponse
}

function MetricCard({ label, value }: { label: string; value: number | null | undefined }) {
  const display =
    value === null || value === undefined ? '—' : Number.isFinite(value) ? value.toFixed(4) : '—'
  return (
    <div className="bg-bg-primary rounded-lg p-3 border border-white/5">
      <div className="text-xs text-text-muted mb-1">{label}</div>
      <div className="text-lg font-semibold text-accent-cyan">{display}</div>
    </div>
  )
}

const chartLayout: Partial<Layout> = {
  ...darkLayout,
  margin: { t: 20, r: 20, b: 40, l: 60 },
  height: 220,
}

export function FitResultsPanel({ result }: FitResultsPanelProps) {
  const {
    metrics,
    parameters,
    observed,
    simulated,
    residuals,
    contributions,
    step_response,
    acf,
    warnings,
    validation_metrics,
    cal_period,
    val_period,
  } = result

  const hasStepResponse =
    step_response?.index?.length > 0 && step_response?.values?.length > 0
  const acfLags = acf?.lags as number[] | undefined
  const acfValues = acf?.acf as number[] | undefined
  const hasAcf = Array.isArray(acfLags) && Array.isArray(acfValues) && acfLags.length > 0

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

      {/* Calibration metric cards */}
      <div>
        {(cal_period || validation_metrics) && (
          <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-2 flex items-center gap-2">
            Calibration
            {cal_period && (
              <span className="text-text-muted font-normal normal-case">
                ({cal_period[0]} → {cal_period[1]})
              </span>
            )}
          </div>
        )}
        <div className="grid grid-cols-3 gap-3">
          <MetricCard label="NSE" value={metrics['nse']} />
          <MetricCard label="KGE" value={metrics['kge']} />
          <MetricCard label="EVP (%)" value={metrics['evp']} />
          <MetricCard label="RMSE" value={metrics['rmse']} />
          <MetricCard label="R²" value={metrics['rsq']} />
          <MetricCard
            label="Ljung-Box p"
            value={metrics['ljung_box_pvalue'] ?? metrics['ljung_box_p']}
          />
        </div>
      </div>

      {/* Validation metric cards */}
      {validation_metrics && (
        <div>
          <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-2 flex items-center gap-2">
            Validation
            {val_period && (
              <span className="text-text-muted font-normal normal-case">
                ({val_period[0]} → {val_period[1]})
              </span>
            )}
          </div>
          <div className="grid grid-cols-3 gap-3">
            <MetricCard label="NSE (val)" value={validation_metrics['nse']} />
            <MetricCard label="KGE (val)" value={validation_metrics['kge']} />
            <MetricCard label="RMSE (val)" value={validation_metrics['rmse']} />
          </div>
        </div>
      )}

      {/* Parameters table */}
      {parameters.length > 0 && (
        <div className="bg-bg-primary rounded-lg border border-white/5 overflow-hidden">
          <div className="px-3 py-2 border-b border-white/5">
            <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
              Parameters
            </span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-text-muted border-b border-white/5">
                  <th className="text-left px-3 py-2">Name</th>
                  <th className="text-right px-3 py-2">Optimal</th>
                  <th className="text-right px-3 py-2">Std err</th>
                  <th className="text-right px-3 py-2">Initial</th>
                </tr>
              </thead>
              <tbody>
                {parameters.map((p) => (
                  <tr key={p.name} className="border-b border-white/5 hover:bg-bg-hover">
                    <td className="px-3 py-2 text-text-primary font-mono">{p.name}</td>
                    <td className="px-3 py-2 text-right text-accent-cyan">
                      {p.optimal.toFixed(6)}
                    </td>
                    <td className="px-3 py-2 text-right text-text-secondary">
                      {p.stderr !== null ? p.stderr.toFixed(6) : '—'}
                    </td>
                    <td className="px-3 py-2 text-right text-text-muted">
                      {p.initial.toFixed(6)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Observed vs Simulated */}
      {observed?.index?.length > 0 && (
        <div className="bg-bg-primary rounded-lg border border-white/5 p-3">
          <p className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">
            Observed vs Simulated
          </p>
          <Plot
            data={[
              {
                x: observed.index,
                y: observed.values,
                type: 'scatter',
                mode: 'lines',
                name: 'Observed',
                line: { color: '#6b7280', width: 1 },
              },
              {
                x: simulated.index,
                y: simulated.values,
                type: 'scatter',
                mode: 'lines',
                name: 'Simulated',
                line: { color: '#22d3ee', width: 2 },
              },
            ]}
            layout={{
              ...chartLayout,
              shapes:
                cal_period && val_period
                  ? [
                      {
                        type: 'line' as const,
                        x0: cal_period[1],
                        x1: cal_period[1],
                        y0: 0,
                        y1: 1,
                        yref: 'paper' as const,
                        line: { color: '#f97316', width: 2, dash: 'dash' as const },
                      },
                    ]
                  : [],
            }}
            config={plotlyConfig}
            style={{ width: '100%' }}
          />
        </div>
      )}

      {/* Stress contributions */}
      {contributions && Object.keys(contributions).length > 0 && (
        <ContributionsChart contributions={contributions} observed={observed} />
      )}

      {/* Residuals */}
      {residuals?.index?.length > 0 && (
        <div className="bg-bg-primary rounded-lg border border-white/5 p-3">
          <p className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">
            Residuals
          </p>
          <Plot
            data={[
              {
                x: residuals.index,
                y: residuals.values,
                type: 'bar',
                name: 'Residuals',
                marker: { color: '#f59e0b', opacity: 0.7 },
              },
            ]}
            layout={{ ...chartLayout, height: 160 }}
            config={plotlyConfig}
            style={{ width: '100%' }}
          />
        </div>
      )}

      {/* Step response */}
      {hasStepResponse && (
        <div className="bg-bg-primary rounded-lg border border-white/5 p-3">
          <p className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">
            Step response
          </p>
          <Plot
            data={[
              {
                x: step_response.index,
                y: step_response.values,
                type: 'scatter',
                mode: 'lines',
                name: 'Step response',
                line: { color: '#a78bfa', width: 2 },
              },
            ]}
            layout={{ ...chartLayout, height: 180 }}
            config={plotlyConfig}
            style={{ width: '100%' }}
          />
        </div>
      )}

      {/* ACF */}
      {hasAcf && (
        <div className="bg-bg-primary rounded-lg border border-white/5 p-3">
          <p className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">
            ACF (residuals)
          </p>
          <Plot
            data={[
              {
                x: acfLags,
                y: acfValues,
                type: 'bar',
                name: 'ACF',
                marker: { color: '#34d399', opacity: 0.8 },
              },
            ]}
            layout={{ ...chartLayout, height: 160 }}
            config={plotlyConfig}
            style={{ width: '100%' }}
          />
        </div>
      )}
    </div>
  )
}
