import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { PastasFitResponse } from '@/lib/types'
import type { Layout } from 'plotly.js-dist-min'
import { ContributionsChart } from '@/components/pastas/ContributionsChart'
import { usePastasDiagnostics, usePastasSignatures } from '@/hooks/usePastas'
import { DiagnosticsPanel } from './DiagnosticsPanel'
import { ResponsePanel } from './ResponsePanel'
import { SignaturesPanel } from './SignaturesPanel'

// --- Accordion section ---

function Section({ title, defaultOpen = true, children }: {
  title: string; defaultOpen?: boolean; children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="bg-bg-primary rounded-lg border border-white/5 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-bg-hover transition-colors"
      >
        <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">{title}</span>
        <ChevronDown className={`w-4 h-4 text-text-muted transition-transform ${open ? '' : '-rotate-90'}`} />
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  )
}

// --- Metric definitions ---

const METRIC_DEFS: Record<string, {
  label: string; tooltip: string
  format: (v: number) => string
  quality: (v: number) => 'good' | 'ok' | 'poor'
}> = {
  nse: {
    label: 'NSE', tooltip: 'Nash-Sutcliffe Efficiency. 1 = parfait, 0 = aussi bon que la moyenne, <0 = pire.',
    format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor',
  },
  kge: {
    label: 'KGE', tooltip: 'Kling-Gupta Efficiency. Combine corrélation, biais et variabilité. >0.7 = bon.',
    format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor',
  },
  evp: {
    label: 'EVP (%)', tooltip: 'Explained Variance Percentage. 100% = variance totalement expliquée.',
    format: v => v.toFixed(1), quality: v => v > 70 ? 'good' : v > 40 ? 'ok' : 'poor',
  },
  rmse: {
    label: 'RMSE', tooltip: 'Root Mean Square Error (m). Plus c\'est bas, mieux c\'est.',
    format: v => v.toFixed(4), quality: () => 'ok',
  },
  rsq: {
    label: 'R²', tooltip: 'Coefficient de détermination. 1 = corrélation parfaite.',
    format: v => v.toFixed(3), quality: v => v > 0.7 ? 'good' : v > 0.4 ? 'ok' : 'poor',
  },
  mae: {
    label: 'MAE', tooltip: 'Mean Absolute Error (m). Erreur moyenne en valeur absolue.',
    format: v => v.toFixed(4), quality: () => 'ok',
  },
}

const Q_COLORS = { good: 'text-green-400', ok: 'text-accent-cyan', poor: 'text-red-400' }
const Q_BORDERS = { good: 'border-green-500/20', ok: 'border-white/5', poor: 'border-red-500/20' }

function MetricCard({ metricKey, value }: { metricKey: string; value: number | null | undefined }) {
  const def = METRIC_DEFS[metricKey]
  if (!def) return null
  const hasValue = value !== null && value !== undefined && Number.isFinite(value)
  const quality = hasValue ? def.quality(value!) : 'ok'
  return (
    <div className={`bg-bg-card rounded-lg p-2.5 border ${Q_BORDERS[quality]}`} title={def.tooltip}>
      <div className="text-[10px] text-text-muted mb-0.5 flex items-center gap-1">
        {def.label}<span className="cursor-help opacity-50">ⓘ</span>
      </div>
      <div className={`text-base font-semibold ${hasValue ? Q_COLORS[quality] : 'text-text-muted'}`}>
        {hasValue ? def.format(value!) : '—'}
      </div>
    </div>
  )
}

function MetricGrid({ metrics, title, period, borderColor }: {
  metrics: Record<string, number>; title?: string; period?: string[] | null; borderColor?: string
}) {
  return (
    <div className={`rounded-lg border ${borderColor ?? 'border-white/5'} p-3`}>
      {title && (
        <div className={`text-xs font-semibold uppercase tracking-wide mb-2 flex items-center gap-2 ${
          borderColor?.includes('orange') ? 'text-orange-400' : 'text-accent-cyan'
        }`}>
          {title}
          {period && <span className="text-text-muted font-normal normal-case text-[10px]">{period[0]} → {period[1]}</span>}
        </div>
      )}
      <div className="grid grid-cols-3 gap-2">
        {['nse', 'kge', 'evp', 'rmse', 'rsq', 'mae'].map(k => (
          <MetricCard key={k} metricKey={k} value={metrics[k]} />
        ))}
      </div>
    </div>
  )
}

const chartLayout: Partial<Layout> = {
  ...darkLayout,
  margin: { t: 20, r: 20, b: 40, l: 60 },
  height: 220,
}

// --- Main component ---

interface FitResultsPanelProps {
  result: PastasFitResponse
}

export function FitResultsPanel({ result }: FitResultsPanelProps) {
  const { data: diagnosticsData } = usePastasDiagnostics(result.run_id)
  const { data: signaturesData } = usePastasSignatures(result.run_id)

  const {
    metrics, parameters, observed, simulated, residuals,
    contributions, step_response, block_response,
    warnings, validation_metrics, cal_period, val_period,
  } = result

  const hasStepResponse = step_response?.index?.length > 0 && step_response?.values?.length > 0

  return (
    <div className="space-y-3">
      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
          <p className="text-xs font-semibold text-yellow-400 mb-1">Avertissements</p>
          {warnings.map((w, i) => <p key={i} className="text-xs text-yellow-300">{w}</p>)}
        </div>
      )}

      {/* 1. Metrics */}
      <Section title="Métriques de performance">
        {validation_metrics ? (
          <div className="grid grid-cols-2 gap-3">
            <MetricGrid metrics={metrics} title="Entraînement" period={cal_period} borderColor="border-accent-cyan/20" />
            <MetricGrid metrics={validation_metrics} title="Test (données inédites)" period={val_period} borderColor="border-orange-500/20" />
          </div>
        ) : (
          <MetricGrid metrics={metrics} />
        )}
      </Section>

      {/* 2. Time series */}
      <Section title="Séries temporelles — Observé vs Simulé">
        {observed?.index?.length > 0 && (
          val_period && cal_period ? (
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs font-semibold text-accent-cyan mb-1">
                  Entraînement ({cal_period[0]?.slice(0,4)}–{cal_period[1]?.slice(0,4)})
                </p>
                <Plot
                  data={[
                    { x: observed.index, y: observed.values, type: 'scatter', mode: 'lines', name: 'Observé', line: { color: '#6b7280', width: 1 } },
                    { x: simulated.index, y: simulated.values, type: 'scatter', mode: 'lines', name: 'Simulé', line: { color: '#22d3ee', width: 2 } },
                  ]}
                  layout={{ ...chartLayout, xaxis: { range: [cal_period[0], cal_period[1]], gridcolor: 'rgba(255,255,255,0.03)' } }}
                  config={plotlyConfig} style={{ width: '100%' }}
                />
              </div>
              <div>
                <p className="text-xs font-semibold text-orange-400 mb-1">
                  Test ({val_period[0]?.slice(0,4)}–{val_period[1]?.slice(0,4)})
                </p>
                <Plot
                  data={[
                    { x: observed.index, y: observed.values, type: 'scatter', mode: 'lines', name: 'Observé', line: { color: '#6b7280', width: 1 } },
                    { x: simulated.index, y: simulated.values, type: 'scatter', mode: 'lines', name: 'Simulé', line: { color: '#f97316', width: 2 } },
                  ]}
                  layout={{ ...chartLayout, xaxis: { range: [val_period[0], val_period[1]], gridcolor: 'rgba(255,255,255,0.03)' } }}
                  config={plotlyConfig} style={{ width: '100%' }}
                />
              </div>
            </div>
          ) : (
            <Plot
              data={[
                { x: observed.index, y: observed.values, type: 'scatter', mode: 'lines', name: 'Observé', line: { color: '#6b7280', width: 1 } },
                { x: simulated.index, y: simulated.values, type: 'scatter', mode: 'lines', name: 'Simulé', line: { color: '#22d3ee', width: 2 } },
              ]}
              layout={chartLayout} config={plotlyConfig} style={{ width: '100%' }}
            />
          )
        )}

      </Section>

      {/* 3. Contributions */}
      {contributions && Object.keys(contributions).length > 0 && (
        <Section title="Décomposition des contributions">
          <ContributionsChart contributions={contributions} />
        </Section>
      )}

      {/* 4. Response function */}
      {(hasStepResponse || block_response?.values?.length > 0) && (
        <Section title="Fonction de réponse">
          <ResponsePanel stepResponse={step_response} blockResponse={block_response} parameters={parameters} responseType="" />
        </Section>
      )}

      {/* 5. Residuals & diagnostics */}
      <Section title="Résidus & diagnostics">
        {residuals?.index?.length > 0 && (() => {
          const vals = residuals.values.filter(v => Number.isFinite(v))
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length
          const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length)
          const threshold = 2 * std
          return (
            <div className="mb-3">
              <p className="text-xs text-text-muted mb-1">Barres rouges = erreur supérieure à 2 écarts-types</p>
              <Plot
                data={[{
                  x: residuals.index, y: residuals.values, type: 'bar', name: 'Résidus',
                  marker: { color: residuals.values.map(v => Math.abs(v) > threshold ? 'rgba(239,68,68,0.7)' : 'rgba(245,158,11,0.5)') },
                }]}
                layout={{
                  ...chartLayout, height: 160,
                  shapes: [
                    { type: 'line', x0: residuals.index[0], x1: residuals.index[residuals.index.length - 1], y0: threshold, y1: threshold, line: { color: 'rgba(239,68,68,0.3)', dash: 'dot', width: 1 } },
                    { type: 'line', x0: residuals.index[0], x1: residuals.index[residuals.index.length - 1], y0: -threshold, y1: -threshold, line: { color: 'rgba(239,68,68,0.3)', dash: 'dot', width: 1 } },
                  ],
                }}
                config={plotlyConfig} style={{ width: '100%' }}
              />
            </div>
          )
        })()}
        {diagnosticsData && <DiagnosticsPanel diagnostics={diagnosticsData} />}
        {!diagnosticsData && <p className="text-xs text-text-muted">Chargement des diagnostics...</p>}
      </Section>

      {/* 6. Parameters */}
      {parameters.length > 0 && (
        <Section title="Paramètres du modèle" defaultOpen={false}>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-text-muted border-b border-white/5">
                  <th className="text-left px-3 py-2">Nom</th>
                  <th className="text-right px-3 py-2">Optimal</th>
                  <th className="text-right px-3 py-2">Std err</th>
                  <th className="text-right px-3 py-2">Initial</th>
                </tr>
              </thead>
              <tbody>
                {parameters.map(p => (
                  <tr key={p.name} className="border-b border-white/5 hover:bg-bg-hover">
                    <td className="px-3 py-2 text-text-primary font-mono">{p.name}</td>
                    <td className="px-3 py-2 text-right text-accent-cyan">{p.optimal.toFixed(6)}</td>
                    <td className="px-3 py-2 text-right text-text-secondary">{p.stderr !== null ? p.stderr.toFixed(6) : '—'}</td>
                    <td className="px-3 py-2 text-right text-text-muted">{p.initial.toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* 7. Signatures */}
      <Section title="Signatures hydrologiques" defaultOpen={false}>
        {signaturesData ? (
          <SignaturesPanel signatures={signaturesData} />
        ) : (
          <p className="text-xs text-text-muted">Chargement des signatures...</p>
        )}
      </Section>
    </div>
  )
}
