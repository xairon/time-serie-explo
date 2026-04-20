import { useState } from 'react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { PastasFitResponse } from '@/lib/types'
import type { Layout } from 'plotly.js-dist-min'
import { ContributionsChart } from '@/components/pastas/ContributionsChart'
import { usePastasDiagnostics, usePastasSignatures } from '@/hooks/usePastas'
import { DiagnosticsPanel } from './DiagnosticsPanel'
import { ResponsePanel } from './ResponsePanel'
import { SignaturesPanel } from './SignaturesPanel'

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
  const [showDiagnostics, setShowDiagnostics] = useState(false)
  const [showSignatures, setShowSignatures] = useState(false)
  const { data: diagnosticsData } = usePastasDiagnostics(showDiagnostics ? result.run_id : null)
  const { data: signaturesData } = usePastasSignatures(showSignatures ? result.run_id : null)

  const {
    metrics,
    parameters,
    observed,
    simulated,
    residuals,
    contributions,
    step_response,
    block_response,
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

      {/* Metrics — side by side when validation is active */}
      {validation_metrics ? (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-bg-primary/50 rounded-lg border border-white/5 p-3">
            <div className="text-xs font-semibold text-accent-cyan uppercase tracking-wide mb-2 flex items-center gap-2">
              Entraînement
              {cal_period && (
                <span className="text-text-muted font-normal normal-case text-[10px]">
                  {cal_period[0]} → {cal_period[1]}
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <MetricCard label="NSE" value={metrics['nse']} />
              <MetricCard label="KGE" value={metrics['kge']} />
              <MetricCard label="EVP (%)" value={metrics['evp']} />
              <MetricCard label="RMSE" value={metrics['rmse']} />
              <MetricCard label="R²" value={metrics['rsq']} />
              <MetricCard label="MAE" value={metrics['mae']} />
            </div>
          </div>
          <div className="bg-bg-primary/50 rounded-lg border border-orange-500/20 p-3">
            <div className="text-xs font-semibold text-orange-400 uppercase tracking-wide mb-2 flex items-center gap-2">
              Test (données inédites)
              {val_period && (
                <span className="text-text-muted font-normal normal-case text-[10px]">
                  {val_period[0]} → {val_period[1]}
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <MetricCard label="NSE" value={validation_metrics['nse']} />
              <MetricCard label="KGE" value={validation_metrics['kge']} />
              <MetricCard label="EVP (%)" value={validation_metrics['evp']} />
              <MetricCard label="RMSE" value={validation_metrics['rmse']} />
              <MetricCard label="R²" value={validation_metrics['rsq']} />
              <MetricCard label="MAE" value={validation_metrics['mae']} />
            </div>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-3">
          <MetricCard label="NSE" value={metrics['nse']} />
          <MetricCard label="KGE" value={metrics['kge']} />
          <MetricCard label="EVP (%)" value={metrics['evp']} />
          <MetricCard label="RMSE" value={metrics['rmse']} />
          <MetricCard label="R²" value={metrics['rsq']} />
          <MetricCard label="MAE" value={metrics['mae']} />
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

      {/* Observed vs Simulated — full period or side-by-side train/test */}
      {observed?.index?.length > 0 && (
        val_period && cal_period ? (
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-bg-primary rounded-lg border border-accent-cyan/20 p-3">
              <p className="text-xs font-semibold text-accent-cyan mb-2 uppercase tracking-wide">
                Entraînement ({cal_period[0]?.slice(0,4)}–{cal_period[1]?.slice(0,4)})
              </p>
              <Plot
                data={[
                  { x: observed.index, y: observed.values, type: 'scatter', mode: 'lines', name: 'Observé', line: { color: '#6b7280', width: 1 } },
                  { x: simulated.index, y: simulated.values, type: 'scatter', mode: 'lines', name: 'Simulé', line: { color: '#22d3ee', width: 2 } },
                ]}
                layout={{ ...chartLayout, xaxis: { range: [cal_period[0], cal_period[1]], gridcolor: 'rgba(255,255,255,0.03)' } }}
                config={plotlyConfig}
                style={{ width: '100%' }}
              />
            </div>
            <div className="bg-bg-primary rounded-lg border border-orange-500/20 p-3">
              <p className="text-xs font-semibold text-orange-400 mb-2 uppercase tracking-wide">
                Test ({val_period[0]?.slice(0,4)}–{val_period[1]?.slice(0,4)})
              </p>
              <Plot
                data={[
                  { x: observed.index, y: observed.values, type: 'scatter', mode: 'lines', name: 'Observé', line: { color: '#6b7280', width: 1 } },
                  { x: simulated.index, y: simulated.values, type: 'scatter', mode: 'lines', name: 'Simulé', line: { color: '#f97316', width: 2 } },
                ]}
                layout={{ ...chartLayout, xaxis: { range: [val_period[0], val_period[1]], gridcolor: 'rgba(255,255,255,0.03)' } }}
                config={plotlyConfig}
                style={{ width: '100%' }}
              />
            </div>
          </div>
        ) : (
          <div className="bg-bg-primary rounded-lg border border-white/5 p-3">
            <p className="text-xs font-semibold text-text-secondary mb-2 uppercase tracking-wide">
              Observé vs Simulé
            </p>
            <Plot
              data={[
                { x: observed.index, y: observed.values, type: 'scatter', mode: 'lines', name: 'Observé', line: { color: '#6b7280', width: 1 } },
                { x: simulated.index, y: simulated.values, type: 'scatter', mode: 'lines', name: 'Simulé', line: { color: '#22d3ee', width: 2 } },
              ]}
              layout={chartLayout}
              config={plotlyConfig}
              style={{ width: '100%' }}
            />
          </div>
        )
      )}

      {/* Stress contributions */}
      {contributions && Object.keys(contributions).length > 0 && (
        <ContributionsChart contributions={contributions} observed={observed} simulated={simulated} />
      )}

      {/* Response function panel */}
      {(hasStepResponse || block_response?.values?.length > 0) && (
        <ResponsePanel
          stepResponse={step_response}
          blockResponse={block_response}
          parameters={parameters}
          responseType=""
        />
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

      {/* Detailed diagnostics (collapsible) */}
      <div>
        <button
          onClick={() => setShowDiagnostics(!showDiagnostics)}
          className="text-xs text-accent-cyan hover:text-accent-cyan/80 transition-colors"
        >
          {showDiagnostics ? '▼ Hide diagnostics' : '▶ Show detailed diagnostics'}
        </button>
        {showDiagnostics && diagnosticsData && (
          <div className="mt-3">
            <DiagnosticsPanel diagnostics={diagnosticsData} />
          </div>
        )}
      </div>

      {/* Hydrological signatures (collapsible) */}
      <div>
        <button
          onClick={() => setShowSignatures(!showSignatures)}
          className="text-xs text-accent-cyan hover:text-accent-cyan/80 transition-colors"
        >
          {showSignatures ? '▼ Hide signatures' : '▶ Show hydrological signatures'}
        </button>
        {showSignatures && signaturesData && (
          <div className="mt-3">
            <SignaturesPanel signatures={signaturesData} />
          </div>
        )}
      </div>
    </div>
  )
}
