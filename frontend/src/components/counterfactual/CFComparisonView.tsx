import { useState, useMemo, useCallback } from 'react'
import Plot from 'react-plotly.js'
import { CheckCircle, AlertTriangle, XCircle, ChevronDown, ChevronRight, ShieldCheck, Thermometer } from 'lucide-react'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { CounterfactualResult, PastasValidationResult } from '@/lib/types'
import { api } from '@/lib/api'

const THETA_LABELS: Record<string, string> = {
  s_P_DJF: 'Precipitations hiver (DJF)',
  s_P_MAM: 'Precipitations printemps (MAM)',
  s_P_JJA: 'Precipitations ete (JJA)',
  s_P_SON: 'Precipitations automne (SON)',
  delta_T: 'Temperature (\u00b0C)',
  delta_etp: 'ETP residuelle',
  delta_s: 'Decalage temporel (jours)',
}

function interpretTheta(theta: Record<string, number>): string[] {
  const lines: string[] = []
  for (const [key, season] of [
    ['s_P_DJF', 'hiver'], ['s_P_MAM', 'printemps'], ['s_P_JJA', 'ete'], ['s_P_SON', 'automne'],
  ] as const) {
    const v = theta[key]
    if (v != null && Math.abs(v - 1) > 0.05) {
      const pct = Math.round((v - 1) * 100)
      lines.push(`Precipitations ${season} : ${pct > 0 ? '+' : ''}${pct}%`)
    }
  }
  if (theta.delta_T != null && Math.abs(theta.delta_T) > 0.1)
    lines.push(`Temperature : ${theta.delta_T > 0 ? '+' : ''}${theta.delta_T.toFixed(1)}\u00b0C`)
  if (theta.delta_etp != null && Math.abs(theta.delta_etp) > 0.01)
    lines.push(`ETP residuelle : ${theta.delta_etp > 0 ? '+' : ''}${theta.delta_etp.toFixed(3)}`)
  if (theta.delta_s != null && Math.abs(theta.delta_s) > 1)
    lines.push(`Decalage temporel : ${theta.delta_s > 0 ? '+' : ''}${Math.round(theta.delta_s)} jours`)
  return lines
}

interface CFComparisonViewProps {
  results: Record<string, CounterfactualResult | null>
  streaming: Record<string, boolean>
  modelId: string
}

export function CFComparisonView({ results, streaming, modelId }: CFComparisonViewProps) {
  const [showTheta, setShowTheta] = useState(false)
  const [pastasResults, setPastasResults] = useState<Record<string, PastasValidationResult | null>>({})
  const [pastasLoading, setPastasLoading] = useState<Record<string, boolean>>({})

  const physcf = results.physcf?.result ?? null
  const comet = results.comet?.result ?? null
  const physcfError = results.physcf?.error ?? null
  const cometError = results.comet?.error ?? null
  const anyRunning = streaming.physcf || streaming.comet
  const anyResult = physcf || comet

  // Metrics extraction
  const physcfMetrics = physcf?.metrics as Record<string, unknown> | undefined
  const cometMetrics = comet?.metrics as Record<string, unknown> | undefined

  // Overlay chart: GT + predictions + CF PhysCF + CF COMET
  const traces = useMemo(() => {
    if (!anyResult) return []
    const ref = physcf ?? comet
    if (!ref) return []

    const t: Plotly.Data[] = []

    // Ground truth (original)
    t.push({
      x: ref.dates,
      y: ref.original,
      name: 'Observe (GT)',
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgba(255,255,255,0.8)', width: 2 },
    })

    // CF PhysCF
    if (physcf) {
      t.push({
        x: physcf.dates,
        y: physcf.counterfactual,
        name: 'CF PhysCF',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#f97316', width: 2 },
      })
    }

    // CF COMET
    if (comet) {
      t.push({
        x: comet.dates,
        y: comet.counterfactual,
        name: 'CF COMET',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#a78bfa', width: 2 },
      })
    }

    return t
  }, [physcf, comet, anyResult])

  const layout = useMemo<Partial<Plotly.Layout>>(() => ({
    ...darkLayout,
    height: 320,
    margin: { t: 30, b: 40, l: 50, r: 20 },
    xaxis: { ...darkLayout.xaxis, type: 'date' },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Niveau (m)', standoff: 8 } },
    legend: { orientation: 'h' as const, y: 1.12, x: 0, font: { size: 11, color: '#9ca3af' } },
    showlegend: true,
  }), [])

  // Pastas validation for both methods
  const handlePastasValidate = useCallback(async () => {
    for (const method of ['physcf', 'comet'] as const) {
      const r = results[method]
      if (!r?.task_id || !modelId) continue
      setPastasLoading((prev) => ({ ...prev, [method]: true }))
      try {
        const resp = await api.counterfactual.pastasValidate({ model_id: modelId, cf_task_id: r.task_id })
        setPastasResults((prev) => ({ ...prev, [method]: resp }))
      } catch (err) {
        setPastasResults((prev) => ({
          ...prev,
          [method]: { model_id: modelId, cf_task_id: r.task_id, gamma: 1.5, accepted: false, rmse_cf: 0, rmse_0: 0, epsilon: 0, status: 'error', message: (err as Error).message },
        }))
      }
      setPastasLoading((prev) => ({ ...prev, [method]: false }))
    }
  }, [results, modelId])

  const anyPastasLoading = pastasLoading.physcf || pastasLoading.comet
  const hasPastas = pastasResults.physcf || pastasResults.comet

  // Loading state
  if (anyRunning && !anyResult) {
    return (
      <div className="bg-bg-card rounded-xl border border-white/5 p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-bg-hover rounded w-1/3" />
          <div className="h-8 bg-bg-hover rounded w-2/3" />
          <div className="flex gap-4">
            {streaming.physcf && (
              <span className="text-xs text-amber-400 flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                PhysCF en cours...
              </span>
            )}
            {streaming.comet && (
              <span className="text-xs text-amber-400 flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                COMET en cours...
              </span>
            )}
          </div>
        </div>
      </div>
    )
  }

  if (!anyResult && !physcfError && !cometError) {
    return (
      <div className="bg-bg-card rounded-xl border border-white/5 p-6 flex items-center justify-center">
        <p className="text-text-secondary text-sm">
          Configurez et lancez une analyse contrefactuelle pour voir les resultats.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Errors */}
      {physcfError && (
        <div className="bg-bg-card rounded-xl border border-red-500/20 p-3 flex items-center gap-2">
          <XCircle className="w-4 h-4 text-red-400 shrink-0" />
          <span className="text-sm text-red-400">PhysCF : {physcfError}</span>
        </div>
      )}
      {cometError && (
        <div className="bg-bg-card rounded-xl border border-red-500/20 p-3 flex items-center gap-2">
          <XCircle className="w-4 h-4 text-red-400 shrink-0" />
          <span className="text-sm text-red-400">COMET : {cometError}</span>
        </div>
      )}

      {/* Status indicators */}
      {anyResult && (
        <div className="flex gap-2">
          {streaming.physcf && (
            <span className="text-xs text-amber-400 flex items-center gap-1 bg-amber-500/10 px-2 py-1 rounded">
              <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              PhysCF en cours...
            </span>
          )}
          {streaming.comet && (
            <span className="text-xs text-amber-400 flex items-center gap-1 bg-amber-500/10 px-2 py-1 rounded">
              <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              COMET en cours...
            </span>
          )}
        </div>
      )}

      {/* 1. Overlay chart */}
      {traces.length > 0 && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-4">
          <h4 className="text-sm font-semibold text-text-primary mb-2">Comparaison des contrefactuels</h4>
          <Plot
            data={traces}
            layout={layout}
            config={plotlyConfig}
            useResizeHandler
            style={{ width: '100%', height: 320 }}
          />
        </div>
      )}

      {/* 2. Metrics comparison table */}
      {anyResult && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-4">
          <h4 className="text-sm font-semibold text-text-primary mb-3">Metriques comparees</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left text-text-secondary py-2 pr-4 font-medium"></th>
                  <th className="text-center text-orange-400 py-2 px-3 font-medium">PhysCF</th>
                  <th className="text-center text-purple-400 py-2 px-3 font-medium">COMET</th>
                </tr>
              </thead>
              <tbody className="text-text-primary font-mono text-xs">
                <MetricRow
                  label="Convergence"
                  physcf={physcfMetrics?.converged === true ? 'Oui' : physcf ? 'Partielle' : '\u2014'}
                  comet={cometMetrics?.converged === true ? 'Oui' : comet ? 'Partielle' : '\u2014'}
                  highlight
                />
                <MetricRow
                  label="Loss finale"
                  physcf={formatNum(physcfMetrics?.best_loss)}
                  comet={formatNum(cometMetrics?.best_loss)}
                />
                <MetricRow
                  label="Iterations"
                  physcf={formatInt(physcfMetrics?.n_iter)}
                  comet={formatInt(cometMetrics?.n_iter)}
                />
                <MetricRow
                  label="Temps (s)"
                  physcf={formatNum(physcfMetrics?.wall_clock_s, 1)}
                  comet={formatNum(cometMetrics?.wall_clock_s, 1)}
                />
                <MetricRow
                  label="Parametres"
                  physcf={physcf ? '7' : '\u2014'}
                  comet={comet ? `${comet.counterfactual.length}\u00d73` : '\u2014'}
                />
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 3. PhysCF interpretation */}
      {physcf?.theta && Object.keys(physcf.theta).length > 0 && (
        <div className="bg-bg-card rounded-xl border border-white/5">
          <button
            onClick={() => setShowTheta(!showTheta)}
            className="w-full px-4 py-3 flex items-center gap-2 text-xs text-text-secondary uppercase hover:text-text-primary transition-colors"
          >
            {showTheta ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            <Thermometer className="w-3.5 h-3.5" />
            Interpretation physique (PhysCF)
          </button>
          {showTheta && (
            <div className="px-4 pb-4 space-y-3">
              {/* Interpretation */}
              {(() => {
                const interp = interpretTheta(physcf.theta)
                if (interp.length === 0) return null
                return (
                  <ul className="space-y-1">
                    {interp.map((line, i) => (
                      <li key={i} className="text-sm text-text-primary flex items-start gap-2">
                        <span className="text-accent-cyan mt-0.5">&gt;</span>
                        {line}
                      </li>
                    ))}
                  </ul>
                )
              })()}
              {/* Raw theta values */}
              <div className="border-t border-white/5 pt-2 space-y-1">
                {Object.entries(physcf.theta).map(([key, val]) => (
                  <div key={key} className="flex items-center justify-between text-xs">
                    <span className="text-text-secondary">{THETA_LABELS[key] ?? key}</span>
                    <span className="text-text-primary font-mono">
                      {typeof val === 'number' ? val.toFixed(4) : String(val)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* 4. Pastas dual validation */}
      {anyResult && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              <ShieldCheck className="w-4 h-4" />
              Validation Pastas (TFN)
            </h4>
            {!hasPastas && (
              <button
                onClick={handlePastasValidate}
                disabled={anyPastasLoading}
                className="text-xs bg-accent-indigo/20 text-accent-indigo px-3 py-1.5 rounded-lg hover:bg-accent-indigo/30 disabled:opacity-50 transition-colors"
              >
                {anyPastasLoading ? 'Validation...' : 'Valider les deux methodes'}
              </button>
            )}
          </div>

          {anyPastasLoading && (
            <div className="animate-pulse flex gap-3">
              <div className="h-3 bg-bg-hover rounded w-1/3" />
              <div className="h-3 bg-bg-hover rounded w-1/4" />
            </div>
          )}

          {hasPastas && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left text-text-secondary py-2 pr-4 font-medium"></th>
                    <th className="text-center text-orange-400 py-2 px-3 font-medium">PhysCF</th>
                    <th className="text-center text-purple-400 py-2 px-3 font-medium">COMET</th>
                  </tr>
                </thead>
                <tbody className="text-text-primary font-mono text-xs">
                  <tr className="border-b border-white/5">
                    <td className="py-2 pr-4 text-text-secondary">Verdict</td>
                    <PastasCell result={pastasResults.physcf} />
                    <PastasCell result={pastasResults.comet} />
                  </tr>
                  <MetricRow
                    label="RMSE CF"
                    physcf={pastasResults.physcf?.status !== 'error' ? pastasResults.physcf?.rmse_cf.toFixed(4) ?? '\u2014' : '\u2014'}
                    comet={pastasResults.comet?.status !== 'error' ? pastasResults.comet?.rmse_cf.toFixed(4) ?? '\u2014' : '\u2014'}
                  />
                  <MetricRow
                    label="RMSE baseline"
                    physcf={pastasResults.physcf?.status !== 'error' ? pastasResults.physcf?.rmse_0.toFixed(4) ?? '\u2014' : '\u2014'}
                    comet={pastasResults.comet?.status !== 'error' ? pastasResults.comet?.rmse_0.toFixed(4) ?? '\u2014' : '\u2014'}
                  />
                  <MetricRow
                    label={`Seuil \u03b5 (\u03b3=${pastasResults.physcf?.gamma ?? pastasResults.comet?.gamma ?? 1.5})`}
                    physcf={pastasResults.physcf?.status !== 'error' ? pastasResults.physcf?.epsilon.toFixed(4) ?? '\u2014' : '\u2014'}
                    comet={pastasResults.comet?.status !== 'error' ? pastasResults.comet?.epsilon.toFixed(4) ?? '\u2014' : '\u2014'}
                  />
                </tbody>
              </table>
            </div>
          )}

          {!hasPastas && !anyPastasLoading && (
            <p className="text-[10px] text-text-secondary/50">
              Validation independante par modele hydrologique Pastas (Transfer Function Noise).
              Compare les predictions CF du TFT avec celles d'un modele physique independant.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function MetricRow({ label, physcf, comet, highlight }: { label: string; physcf: string; comet: string; highlight?: boolean }) {
  return (
    <tr className="border-b border-white/5">
      <td className="py-2 pr-4 text-text-secondary">{label}</td>
      <td className={`text-center py-2 px-3 ${highlight ? 'font-semibold' : ''}`}>{physcf}</td>
      <td className={`text-center py-2 px-3 ${highlight ? 'font-semibold' : ''}`}>{comet}</td>
    </tr>
  )
}

function PastasCell({ result }: { result: PastasValidationResult | null | undefined }) {
  if (!result) return <td className="text-center py-2 px-3 text-text-secondary">{'\u2014'}</td>
  if (result.status === 'error') return <td className="text-center py-2 px-3 text-red-400">Erreur</td>
  return (
    <td className="text-center py-2 px-3">
      <span className={`inline-flex items-center gap-1 ${result.accepted ? 'text-emerald-400' : 'text-amber-400'}`}>
        {result.accepted ? <CheckCircle className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
        {result.accepted ? 'Valide' : 'Rejete'}
      </span>
    </td>
  )
}

function formatNum(v: unknown, decimals = 4): string {
  if (typeof v === 'number') return v.toFixed(decimals)
  return '\u2014'
}

function formatInt(v: unknown): string {
  if (typeof v === 'number') return String(Math.round(v))
  return '\u2014'
}
