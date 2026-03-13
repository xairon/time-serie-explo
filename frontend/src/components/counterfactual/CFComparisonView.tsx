import { useState, useMemo, useCallback } from 'react'
import Plot from 'react-plotly.js'
import { CheckCircle, AlertTriangle, XCircle, ChevronDown, ChevronRight, ShieldCheck, Thermometer, Shuffle } from 'lucide-react'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { CounterfactualResult, PastasValidationResult } from '@/lib/types'
import { api } from '@/lib/api'

const THETA_LABELS: Record<string, string> = {
  s_P_DJF: 'Winter precipitation (DJF)',
  s_P_MAM: 'Spring precipitation (MAM)',
  s_P_JJA: 'Summer precipitation (JJA)',
  s_P_SON: 'Autumn precipitation (SON)',
  delta_T: 'Temperature (\u00b0C)',
  delta_etp: 'Residual ETP',
  delta_s: 'Time lag (days)',
}

function interpretTheta(theta: Record<string, number>): string[] {
  const lines: string[] = []
  for (const [key, season] of [
    ['s_P_DJF', 'winter'], ['s_P_MAM', 'spring'], ['s_P_JJA', 'summer'], ['s_P_SON', 'autumn'],
  ] as const) {
    const v = theta[key]
    if (v != null && Math.abs(v - 1) > 0.05) {
      const pct = Math.round((v - 1) * 100)
      lines.push(`${season.charAt(0).toUpperCase() + season.slice(1)} precipitation: ${pct > 0 ? '+' : ''}${pct}%`)
    }
  }
  if (theta.delta_T != null && Math.abs(theta.delta_T) > 0.1)
    lines.push(`Temperature: ${theta.delta_T > 0 ? '+' : ''}${theta.delta_T.toFixed(1)}\u00b0C`)
  if (theta.delta_etp != null && Math.abs(theta.delta_etp) > 0.01)
    lines.push(`Residual ETP: ${theta.delta_etp > 0 ? '+' : ''}${theta.delta_etp.toFixed(3)}`)
  if (theta.delta_s != null && Math.abs(theta.delta_s) > 1)
    lines.push(`Time lag: ${theta.delta_s > 0 ? '+' : ''}${Math.round(theta.delta_s)} days`)
  return lines
}

interface CFComparisonViewProps {
  results: Record<string, CounterfactualResult | null>
  streaming: Record<string, boolean>
  modelId: string
  /** Ground truth (observed) values for the prediction window */
  gtDates?: string[]
  gtValues?: number[]
}

export function CFComparisonView({ results, streaming, modelId, gtDates, gtValues }: CFComparisonViewProps) {
  const [showTheta, setShowTheta] = useState(false)
  const [pastasResults, setPastasResults] = useState<Record<string, PastasValidationResult | null>>({})
  const [pastasLoading, setPastasLoading] = useState<Record<string, boolean>>({})

  const physcf = results.physcf?.result ?? null
  const comte = results.comte?.result ?? null
  const physcfError = results.physcf?.error ?? null
  const comteError = results.comte?.error ?? null
  const anyRunning = streaming.physcf || streaming.comte
  const anyResult = physcf || comte

  // Metrics extraction
  const physcfMetrics = physcf?.metrics as Record<string, unknown> | undefined
  const comteMetrics = comte?.metrics as Record<string, unknown> | undefined
  const comteInfo = comte?.comte_info

  // Common metrics computed identically for both methods
  const commonMetrics = useMemo(() => {
    const compute = (r: typeof physcf) => {
      if (!r) return null
      const orig = r.original
      const cf = r.counterfactual
      const n = Math.min(orig.length, cf.length)
      if (n === 0) return null
      let sumDiff = 0, sumSqDiff = 0, maxDev = 0, sumOrig = 0, sumCf = 0
      for (let i = 0; i < n; i++) {
        const d = cf[i] - orig[i]
        sumDiff += d
        sumSqDiff += d * d
        maxDev = Math.max(maxDev, Math.abs(d))
        sumOrig += orig[i]
        sumCf += cf[i]
      }
      return {
        meanShift: sumDiff / n,
        rmse: Math.sqrt(sumSqDiff / n),
        maxDeviation: maxDev,
        meanOrig: sumOrig / n,
        meanCf: sumCf / n,
      }
    }
    return { physcf: compute(physcf), comte: compute(comte) }
  }, [physcf, comte])

  // Overlay chart: GT + predictions + CF PhysCF + CF CoMTE
  const traces = useMemo(() => {
    if (!anyResult) return []
    const ref = physcf ?? comte
    if (!ref) return []

    const t: Plotly.Data[] = []

    // Ground truth (observed values)
    if (gtDates?.length && gtValues?.length) {
      t.push({
        x: gtDates,
        y: gtValues,
        name: 'Observe (GT)',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#10b981', width: 2 },
      })
    }

    // Model prediction without perturbation (factual baseline for CF comparison)
    t.push({
      x: ref.dates,
      y: ref.original,
      name: 'Modele (sans perturbation)',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#06b6d4', width: 2, dash: 'dash' },
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

    // CF CoMTE
    if (comte) {
      t.push({
        x: comte.dates,
        y: comte.counterfactual,
        name: 'CF CoMTE',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#a78bfa', width: 2 },
      })
    }

    return t
  }, [physcf, comte, anyResult, gtDates, gtValues])

  const layout = useMemo<Partial<Plotly.Layout>>(() => ({
    ...darkLayout,
    height: 320,
    margin: { t: 30, b: 40, l: 50, r: 20 },
    xaxis: { ...darkLayout.xaxis, type: 'date' },
    yaxis: { ...darkLayout.yaxis, title: { text: 'Level (m)', standoff: 8 } },
    legend: { orientation: 'h' as const, y: 1.12, x: 0, font: { size: 11, color: '#9ca3af' } },
    showlegend: true,
  }), [])

  // Pastas validation for both methods
  const handlePastasValidate = useCallback(async () => {
    for (const method of ['physcf', 'comte'] as const) {
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

  const anyPastasLoading = pastasLoading.physcf || pastasLoading.comte
  const hasPastas = pastasResults.physcf || pastasResults.comte

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
                PhysCF running...
              </span>
            )}
            {streaming.comte && (
              <span className="text-xs text-amber-400 flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                CoMTE running...
              </span>
            )}
          </div>
        </div>
      </div>
    )
  }

  if (!anyResult && !physcfError && !comteError) {
    return (
      <div className="bg-bg-card rounded-xl border border-white/5 p-6 flex items-center justify-center">
        <p className="text-text-secondary text-sm">
          Configure and run a counterfactual analysis to see results.
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
      {comteError && (
        <div className="bg-bg-card rounded-xl border border-red-500/20 p-3 flex items-center gap-2">
          <XCircle className="w-4 h-4 text-red-400 shrink-0" />
          <span className="text-sm text-red-400">CoMTE : {comteError}</span>
        </div>
      )}

      {/* Status indicators */}
      {anyResult && (
        <div className="flex gap-2">
          {streaming.physcf && (
            <span className="text-xs text-amber-400 flex items-center gap-1 bg-amber-500/10 px-2 py-1 rounded">
              <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              PhysCF running...
            </span>
          )}
          {streaming.comte && (
            <span className="text-xs text-amber-400 flex items-center gap-1 bg-amber-500/10 px-2 py-1 rounded">
              <span className="inline-block w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              CoMTE running...
            </span>
          )}
        </div>
      )}

      {/* 1. Overlay chart */}
      {traces.length > 0 && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-4">
          <h4 className="text-sm font-semibold text-text-primary mb-2">Counterfactual comparison</h4>
          <Plot
            data={traces}
            layout={layout}
            config={plotlyConfig}
            useResizeHandler
            style={{ width: '100%', height: 320 }}
          />
        </div>
      )}

      {/* 2. Comparaison exhaustive */}
      {anyResult && (() => {
        // Compute winners for each comparable metric
        const inBand = comteInfo?.in_band_fraction ?? null
        const physcfTarget = physcfMetrics?.target_loss_final as number | null ?? null
        // For CoMTE, target_loss = 1 - in_band_fraction
        const comteTarget = inBand !== null ? (1 - inBand) : null

        const w = {
          target: pickWinner(physcfTarget, comteTarget, true),
          rmse: pickWinner(commonMetrics.physcf?.rmse, commonMetrics.comte?.rmse, true),
          maxDev: pickWinner(commonMetrics.physcf?.maxDeviation, commonMetrics.comte?.maxDeviation, true),
          time: pickWinner(physcfMetrics?.wall_clock_s as number | null, comteMetrics?.wall_clock_s as number | null, true),
          params: pickWinner(physcfMetrics?.n_params as number | null, comteMetrics?.n_params as number | null, true),
        }
        // Score: count wins per method
        const vals = Object.values(w)
        const pWins = vals.filter(v => v === 'physcf').length
        const cWins = vals.filter(v => v === 'comte').length
        const overall: 'physcf' | 'comte' | null = pWins > cWins ? 'physcf' : cWins > pWins ? 'comte' : null

        return (
        <div className="bg-bg-card rounded-xl border border-white/5 p-4">
          {/* Verdict banner */}
          {overall && physcf && comte && (
            <div className={`rounded-lg px-3 py-2 mb-4 flex items-center justify-between text-sm ${
              overall === 'physcf'
                ? 'bg-orange-500/10 border border-orange-500/20'
                : 'bg-purple-500/10 border border-purple-500/20'
            }`}>
              <span className={overall === 'physcf' ? 'text-orange-400' : 'text-purple-400'}>
                {overall === 'physcf' ? 'PhysCF' : 'CoMTE'} leads on {Math.max(pWins, cWins)}/{vals.filter(v => v !== null).length} comparable criteria
              </span>
              <span className="text-text-secondary text-xs">
                PhysCF {pWins} — {cWins} CoMTE
              </span>
            </div>
          )}

          <h4 className="text-sm font-semibold text-text-primary mb-3">Exhaustive comparison</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left text-text-secondary py-2 pr-4 font-medium"></th>
                  <th className="text-center text-orange-400 py-2 px-3 font-medium">PhysCF</th>
                  <th className="text-center text-purple-400 py-2 px-3 font-medium">CoMTE</th>
                </tr>
              </thead>
              <tbody className="text-text-primary font-mono text-xs">
                {/* --- Convergence & target --- */}
                <SectionHeader label="Target (IPS target)" />
                <MetricRow
                  label="Convergence"
                  physcf={fmtConvergence(physcfMetrics)}
                  comte={comte ? (inBand !== null ? `${Math.round(inBand * 100)}% in-band` : '\u2014') : '\u2014'}
                  highlight
                  winner={w.target}
                />
                <MetricRow
                  label="Target score"
                  physcf={formatNum(physcfTarget)}
                  comte={inBand !== null ? `${(inBand * 100).toFixed(1)}%` : '\u2014'}
                  winner={w.target}
                />
                {/* --- Approche --- */}
                <SectionHeader label="Approach" />
                <MetricRow
                  label="Type"
                  physcf="Continuous gradient"
                  comte="Discrete substitution"
                />
                <MetricRow
                  label="Space"
                  physcf="7 physical params"
                  comte={`${comteInfo?.swapped_features?.length ?? '?'} substituted features`}
                />
                {/* --- Impact sur la sortie --- */}
                <SectionHeader label="Impact on output" />
                <MetricRow
                  label="Mean shift (m)"
                  physcf={commonMetrics.physcf ? fmtSigned(commonMetrics.physcf.meanShift) : '\u2014'}
                  comte={commonMetrics.comte ? fmtSigned(commonMetrics.comte.meanShift) : '\u2014'}
                />
                <MetricRow
                  label="RMSE factual \u2192 CF"
                  physcf={commonMetrics.physcf?.rmse.toFixed(4) ?? '\u2014'}
                  comte={commonMetrics.comte?.rmse.toFixed(4) ?? '\u2014'}
                  winner={w.rmse}
                />
                <MetricRow
                  label="Max deviation (m)"
                  physcf={commonMetrics.physcf?.maxDeviation.toFixed(4) ?? '\u2014'}
                  comte={commonMetrics.comte?.maxDeviation.toFixed(4) ?? '\u2014'}
                  winner={w.maxDev}
                />
                <MetricRow
                  label="Factual mean (m)"
                  physcf={commonMetrics.physcf?.meanOrig.toFixed(3) ?? '\u2014'}
                  comte={commonMetrics.comte?.meanOrig.toFixed(3) ?? '\u2014'}
                />
                <MetricRow
                  label="CF mean (m)"
                  physcf={commonMetrics.physcf?.meanCf.toFixed(3) ?? '\u2014'}
                  comte={commonMetrics.comte?.meanCf.toFixed(3) ?? '\u2014'}
                />
                {/* --- Cout d'optimisation --- */}
                <SectionHeader label="Optimization cost" />
                <MetricRow
                  label="Evaluations"
                  physcf={formatInt(physcfMetrics?.n_iter)}
                  comte={comteInfo ? String(comteInfo.n_candidates_evaluated) : '\u2014'}
                />
                <MetricRow
                  label="Time (s)"
                  physcf={formatNum(physcfMetrics?.wall_clock_s, 1)}
                  comte={formatNum(comteMetrics?.wall_clock_s, 1)}
                  winner={w.time}
                />
                <MetricRow
                  label="Parameters/features"
                  physcf={physcf ? '7 (contraints)' : '\u2014'}
                  comte={comteInfo ? `${comteInfo.swapped_features.length}/${comteInfo.best_mask?.length ?? 3}` : '\u2014'}
                  winner={w.params}
                />
              </tbody>
            </table>
          </div>
        </div>
        )
      })()}

      {/* 3. Method-specific details — side by side */}
      {anyResult && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* PhysCF card */}
          <div className="bg-bg-card rounded-xl border border-white/5 p-4">
            <h4 className="text-sm font-semibold text-orange-400 mb-3 flex items-center gap-2">
              <Thermometer className="w-4 h-4" />
              PhysCF
              <span className="text-[10px] font-normal text-text-secondary">7 physically constrained params</span>
            </h4>
            {physcf ? (
              <div className="space-y-3">
                <div className="space-y-1 text-xs">
                  <MethodMetric label="CC constraint" value="0.07/K" hint="Clausius-Clapeyron" />
                  <MethodMetric label="Space" value="Physics" hint="seasonal P, \u0394T, \u0394ETP, \u0394s" />
                  <MethodMetric label="\u03bb prox" value={physcf.convergence?.length ? 'active' : '\u2014'} hint="distance to identity" />
                </div>

                {/* Theta interpretation */}
                {physcf.theta && Object.keys(physcf.theta).length > 0 && (
                  <div className="border-t border-white/5 pt-2">
                    <button
                      onClick={() => setShowTheta(!showTheta)}
                      className="flex items-center gap-1.5 text-xs text-text-secondary hover:text-text-primary transition-colors mb-2"
                    >
                      {showTheta ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                      <Thermometer className="w-3 h-3" />
                      Physical interpretation
                    </button>
                    {showTheta && (
                      <div className="space-y-2">
                        {(() => {
                          const interp = interpretTheta(physcf.theta)
                          if (interp.length === 0) return null
                          return (
                            <ul className="space-y-0.5">
                              {interp.map((line, i) => (
                                <li key={i} className="text-xs text-text-primary flex items-start gap-1.5">
                                  <span className="text-accent-cyan mt-0.5">&gt;</span>
                                  {line}
                                </li>
                              ))}
                            </ul>
                          )
                        })()}
                        <div className="space-y-0.5">
                          {Object.entries(physcf.theta).map(([key, val]) => (
                            <div key={key} className="flex items-center justify-between text-[11px]">
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
              </div>
            ) : physcfError ? (
              <p className="text-xs text-red-400">{physcfError}</p>
            ) : (
              <p className="text-xs text-text-secondary">Waiting...</p>
            )}
          </div>

          {/* CoMTE card */}
          <div className="bg-bg-card rounded-xl border border-white/5 p-4">
            <h4 className="text-sm font-semibold text-purple-400 mb-3 flex items-center gap-2">
              <Shuffle className="w-4 h-4" />
              CoMTE
              <span className="text-[10px] font-normal text-text-secondary">Ates et al. 2021 — substitution de features</span>
            </h4>
            {comte ? (
              <div className="space-y-3">
                <div className="space-y-1 text-xs">
                  <MethodMetric label="Algorithm" value="Exhaustive search" hint={`2^${comteInfo?.best_mask?.length ?? 3} combinations`} />
                  <MethodMetric label="Distractors" value={`${comteInfo?.n_distractors_used ?? '?'}/${comteInfo?.n_distractors_available ?? '?'}`} hint={`class ${comteInfo?.distractor_class ?? '?'}`} />
                  <MethodMetric label="In-band" value={comteInfo ? `${Math.round(comteInfo.in_band_fraction * 100)}%` : '\u2014'} hint={`threshold ${comteInfo?.tau ? Math.round(comteInfo.tau * 100) : 50}%`} />
                  <MethodMetric label="Features" value={comteInfo?.swapped_features?.join(', ') || 'none'} />
                </div>

                {/* CoMTE explanation */}
                {comteInfo?.explanation && (
                  <div className="border-t border-white/5 pt-2">
                    <p className="text-xs text-text-secondary leading-relaxed">
                      {comteInfo.explanation}
                    </p>
                  </div>
                )}

                {/* Feature mask visualization */}
                {comteInfo?.best_mask && (
                  <div className="border-t border-white/5 pt-2">
                    <p className="text-[10px] uppercase tracking-wider text-text-secondary/60 font-semibold mb-1.5">
                      Feature mask
                    </p>
                    <div className="flex gap-2">
                      {(comte?.theta && Object.keys(comte.theta).length > 0 ? Object.keys(comte.theta) : ['precip', 'temp', 'evap']).slice(0, comteInfo.best_mask.length).map((name, idx) => (
                        <div
                          key={name}
                          className={`px-2 py-1 rounded text-[11px] font-mono ${
                            comteInfo.best_mask[idx] === 1
                              ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                              : 'bg-bg-hover text-text-secondary/40 border border-white/5'
                          }`}
                        >
                          {name} {comteInfo.best_mask[idx] === 1 ? '\u2713' : '\u2717'}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : comteError ? (
              <p className="text-xs text-red-400">{comteError}</p>
            ) : (
              <p className="text-xs text-text-secondary">Waiting...</p>
            )}
          </div>
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
                {anyPastasLoading ? 'Validating...' : 'Validate both methods'}
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
                    <th className="text-center text-purple-400 py-2 px-3 font-medium">CoMTE</th>
                  </tr>
                </thead>
                <tbody className="text-text-primary font-mono text-xs">
                  <tr className="border-b border-white/5">
                    <td className="py-2 pr-4 text-text-secondary">Verdict</td>
                    <PastasCell result={pastasResults.physcf} />
                    <PastasCell result={pastasResults.comte} />
                  </tr>
                  <MetricRow
                    label="RMSE CF"
                    physcf={pastasResults.physcf?.status !== 'error' ? pastasResults.physcf?.rmse_cf.toFixed(4) ?? '\u2014' : '\u2014'}
                    comte={pastasResults.comte?.status !== 'error' ? pastasResults.comte?.rmse_cf.toFixed(4) ?? '\u2014' : '\u2014'}
                    winner={pickWinner(
                      pastasResults.physcf?.status !== 'error' ? pastasResults.physcf?.rmse_cf : null,
                      pastasResults.comte?.status !== 'error' ? pastasResults.comte?.rmse_cf : null,
                      true,
                    )}
                  />
                  <MetricRow
                    label="RMSE baseline"
                    physcf={pastasResults.physcf?.status !== 'error' ? pastasResults.physcf?.rmse_0.toFixed(4) ?? '\u2014' : '\u2014'}
                    comte={pastasResults.comte?.status !== 'error' ? pastasResults.comte?.rmse_0.toFixed(4) ?? '\u2014' : '\u2014'}
                  />
                  <MetricRow
                    label={`Seuil \u03b5 (\u03b3=${pastasResults.physcf?.gamma ?? pastasResults.comte?.gamma ?? 1.5})`}
                    physcf={pastasResults.physcf?.status !== 'error' ? pastasResults.physcf?.epsilon.toFixed(4) ?? '\u2014' : '\u2014'}
                    comte={pastasResults.comte?.status !== 'error' ? pastasResults.comte?.epsilon.toFixed(4) ?? '\u2014' : '\u2014'}
                  />
                </tbody>
              </table>
            </div>
          )}

          {!hasPastas && !anyPastasLoading && (
            <p className="text-[10px] text-text-secondary/50">
              Independent validation by Pastas hydrological model (Transfer Function Noise).
              Compares the TFT's CF predictions with those of an independent physical model.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

/** winner: 'physcf' | 'comte' | null — highlights the better cell in green */
function MetricRow({ label, physcf, comte, highlight, winner }: {
  label: string; physcf: string; comte: string; highlight?: boolean; winner?: 'physcf' | 'comte' | null
}) {
  const pCls = winner === 'physcf' ? 'text-emerald-400 font-semibold' : winner === 'comte' ? 'text-text-secondary/60' : ''
  const cCls = winner === 'comte' ? 'text-emerald-400 font-semibold' : winner === 'physcf' ? 'text-text-secondary/60' : ''
  return (
    <tr className="border-b border-white/5">
      <td className="py-2 pr-4 text-text-secondary">{label}</td>
      <td className={`text-center py-2 px-3 ${highlight ? 'font-semibold' : ''} ${pCls}`}>{physcf}</td>
      <td className={`text-center py-2 px-3 ${highlight ? 'font-semibold' : ''} ${cCls}`}>{comte}</td>
    </tr>
  )
}

/** Compare two numeric metric values. lower=true means lower is better. */
function pickWinner(
  a: unknown, b: unknown, lower: boolean,
): 'physcf' | 'comte' | null {
  const va = typeof a === 'number' ? a : null
  const vb = typeof b === 'number' ? b : null
  if (va === null || vb === null) return null
  if (Math.abs(va - vb) < 1e-10) return null
  if (lower) return va < vb ? 'physcf' : 'comte'
  return va > vb ? 'physcf' : 'comte'
}

function PastasCell({ result }: { result: PastasValidationResult | null | undefined }) {
  if (!result) return <td className="text-center py-2 px-3 text-text-secondary">{'\u2014'}</td>
  if (result.status === 'error') return <td className="text-center py-2 px-3 text-red-400">Error</td>
  return (
    <td className="text-center py-2 px-3">
      <span className={`inline-flex items-center gap-1 ${result.accepted ? 'text-emerald-400' : 'text-amber-400'}`}>
        {result.accepted ? <CheckCircle className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
        {result.accepted ? 'Valid' : 'Rejected'}
      </span>
    </td>
  )
}

function SectionHeader({ label }: { label: string }) {
  return (
    <tr>
      <td colSpan={3} className="pt-3 pb-1 text-[10px] uppercase tracking-wider text-text-secondary/60 font-semibold">
        {label}
      </td>
    </tr>
  )
}

function fmtConvergence(metrics: Record<string, unknown> | undefined): string {
  if (!metrics) return '\u2014'
  const converged = metrics.converged
  const target = metrics.target_loss_final
  if (converged === true) return 'Yes (< 1e-4)'
  if (typeof target === 'number') {
    if (target < 0.01) return `Near (${target.toFixed(4)})`
    return `No (${target.toFixed(4)})`
  }
  return 'No'
}

function fmtSigned(v: number, decimals = 4): string {
  return (v > 0 ? '+' : '') + v.toFixed(decimals)
}

function MethodMetric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-text-secondary">{label}</span>
      <span className="text-text-primary font-mono">
        {value}
        {hint && <span className="text-text-secondary/50 font-sans ml-1">({hint})</span>}
      </span>
    </div>
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
