import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Loader2, Play, Zap, ArrowLeft, Check, AlertTriangle } from 'lucide-react'
import { usePastasFit, usePastasAutoFit, usePastasStationInfo } from '@/hooks/usePastas'
import { usePastasMode } from './PastasLayout'
import { PastasConfigForm } from '@/components/pastas/PastasConfigForm'
import { CalValToggle } from '@/components/pastas/CalValToggle'
import type { AutoFitResult, AutoFitCandidate } from '@/lib/types'

export default function CalibrateStep() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const { mode } = usePastasMode()

  const codeBss = searchParams.get('station') ?? ''
  const recommendedTmin = searchParams.get('tmin') ?? ''
  const recommendedTmax = searchParams.get('tmax') ?? ''

  // Station info for BDLISA preset
  const { data: stationInfo } = usePastasStationInfo(codeBss || null)

  // Config form state (expert mode)
  const [recharge, setRecharge] = useState('Linear')
  const [response, setResponse] = useState('Gamma')
  const [noise, setNoise] = useState('ArNoiseModel')
  const [solver, setSolver] = useState('LeastSquares')
  const [tmin, setTmin] = useState(recommendedTmin)
  const [tmax, setTmax] = useState(recommendedTmax)
  const [modelName, setModelName] = useState('')
  const [valSplit, setValSplit] = useState<number | null>(0.3)
  const [includeTemp, setIncludeTemp] = useState(false)
  const [warmUpYears, setWarmUpYears] = useState(1)
  const [twoPass, setTwoPass] = useState(false)

  // Apply BDLISA preset
  const [presetApplied, setPresetApplied] = useState('')
  useEffect(() => {
    if (stationInfo?.preset && codeBss !== presetApplied) {
      const p = stationInfo.preset as Record<string, string>
      if (p.recharge) setRecharge(p.recharge)
      if (p.response) setResponse(p.response)
      if (p.noise) setNoise(p.noise)
      setPresetApplied(codeBss)
    }
  }, [stationInfo, codeBss, presetApplied])

  // Mutations
  const fitMutation = usePastasFit()
  const autoFitMutation = usePastasAutoFit()

  // Auto-fit state
  const [autoFitResult, setAutoFitResult] = useState<AutoFitResult | null>(null)

  async function handleFit() {
    if (!codeBss) return
    try {
      const result = await fitMutation.mutateAsync({
        code_bss: codeBss,
        tmin: tmin || undefined,
        tmax: tmax || undefined,
        recharge: { type: recharge },
        response: { type: response },
        noise: { type: noise },
        solver: { type: solver },
        name: modelName || undefined,
        val_split: valSplit ?? undefined,
        include_temp: includeTemp || undefined,
      })
      navigate(`/pastas/results?model=${result.run_id}&station=${codeBss}`)
    } catch {
      // Error handled by mutation state
    }
  }

  async function handleAutoFit() {
    if (!codeBss) return
    try {
      const result = await autoFitMutation.mutateAsync({
        code_bss: codeBss,
        warm_up_years: warmUpYears,
        val_split: valSplit ?? undefined,
        include_temp: includeTemp || undefined,
        add_trend: twoPass || null,
      })
      setAutoFitResult(result)
    } catch {
      // Error handled by mutation state
    }
  }

  function viewResults(runId: string) {
    navigate(`/pastas/results?model=${runId}&station=${codeBss}`)
  }

  if (!codeBss) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        <div className="text-center space-y-3">
          <p className="text-sm text-text-secondary">No station selected</p>
          <button
            onClick={() => navigate('/pastas/station')}
            className="flex items-center gap-2 text-accent-cyan text-sm hover:underline mx-auto"
          >
            <ArrowLeft className="w-4 h-4" />
            Go back to Station step
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 flex gap-6 min-h-full">
      {/* Left column — configuration */}
      <div className="w-80 shrink-0 space-y-4">
        {/* Station indicator */}
        <div className="bg-bg-card border border-white/5 rounded-xl p-3 flex items-center justify-between">
          <div>
            <div className="text-[10px] text-text-muted uppercase tracking-wide">Station</div>
            <div className="text-sm font-mono text-accent-cyan">{codeBss}</div>
          </div>
          <button
            onClick={() => navigate(`/pastas/station?station=${encodeURIComponent(codeBss)}`)}
            className="text-xs text-text-muted hover:text-text-secondary"
          >
            Change
          </button>
        </div>

        {mode === 'guided' ? (
          /* Guided mode — auto-fit */
          <>
            <div className="bg-accent-cyan/5 border border-accent-cyan/20 rounded-xl p-4">
              <h2 className="text-sm font-semibold text-accent-cyan mb-2">Auto-fit</h2>
              <p className="text-xs text-text-muted mb-3">
                Automatically tests multiple configurations and selects the best model based on STOWA criteria.
              </p>

              <div className="space-y-3">
                <div className="bg-bg-card border border-white/5 rounded-lg p-3">
                  <CalValToggle valSplit={valSplit} onChange={setValSplit} />
                </div>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeTemp}
                    onChange={e => setIncludeTemp(e.target.checked)}
                    className="accent-accent-cyan w-4 h-4"
                  />
                  <span className="text-xs text-text-secondary">Include temperature</span>
                </label>

                <div>
                  <label className="block text-xs text-text-muted mb-1">
                    Warm-up period ({warmUpYears} year{warmUpYears !== 1 ? 's' : ''})
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={5}
                    step={1}
                    value={warmUpYears}
                    onChange={e => setWarmUpYears(+e.target.value)}
                    className="w-full accent-accent-cyan"
                  />
                </div>
              </div>
            </div>

            <button
              onClick={handleAutoFit}
              disabled={autoFitMutation.isPending}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {autoFitMutation.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running auto-fit...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Auto-fit
                </>
              )}
            </button>
          </>
        ) : (
          /* Expert mode — manual config */
          <>
            <div className="bg-bg-card border border-white/5 rounded-xl p-4">
              <h2 className="text-sm font-semibold text-text-primary mb-2">Model Configuration</h2>
              {stationInfo?.preset != null && (stationInfo.preset as Record<string, string>).label && (
                <div className="mb-3 flex items-center gap-2 bg-accent-cyan/5 border border-accent-cyan/20 rounded-lg px-3 py-1.5">
                  <span className="text-xs text-accent-cyan font-medium">
                    Recommended: {(stationInfo.preset as Record<string, string>).label}
                  </span>
                </div>
              )}
              <PastasConfigForm
                recharge={recharge}
                onRechargeChange={setRecharge}
                response={response}
                onResponseChange={setResponse}
                noise={noise}
                onNoiseChange={setNoise}
                solver={solver}
                onSolverChange={setSolver}
                tmin={tmin}
                onTminChange={setTmin}
                tmax={tmax}
                onTmaxChange={setTmax}
              />
            </div>

            <div className="bg-bg-card border border-white/5 rounded-xl p-4">
              <CalValToggle valSplit={valSplit} onChange={setValSplit} />
            </div>

            <div className="bg-bg-card border border-white/5 rounded-xl p-4">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeTemp}
                  onChange={e => setIncludeTemp(e.target.checked)}
                  className="accent-accent-cyan w-4 h-4"
                />
                <div>
                  <span className="text-sm font-medium text-text-secondary">Include temperature</span>
                  <p className="text-xs text-text-muted">Adds ERA5 temperature as an additional stress.</p>
                </div>
              </label>
            </div>

            <div className="bg-bg-card border border-white/5 rounded-xl p-4">
              <label className="block text-xs text-text-muted mb-1">
                Warm-up period ({warmUpYears} year{warmUpYears !== 1 ? 's' : ''})
              </label>
              <input
                type="range"
                min={0}
                max={5}
                step={1}
                value={warmUpYears}
                onChange={e => setWarmUpYears(+e.target.value)}
                className="w-full accent-accent-cyan"
              />
              <p className="text-xs text-text-muted mt-1">
                Data before this period is used for model initialization only.
              </p>
            </div>

            <div className="bg-bg-card border border-white/5 rounded-xl p-4">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={twoPass}
                  onChange={e => setTwoPass(e.target.checked)}
                  className="accent-accent-cyan w-4 h-4"
                />
                <div>
                  <span className="text-sm font-medium text-text-secondary">Two-pass (add trend)</span>
                  <p className="text-xs text-text-muted">First fit without trend, then add a linear trend stress if residuals show a trend.</p>
                </div>
              </label>
            </div>

            <div className="bg-bg-card border border-white/5 rounded-xl p-4">
              <label className="block text-sm font-medium text-text-secondary mb-1">
                Run name (optional)
              </label>
              <input
                type="text"
                value={modelName}
                onChange={e => setModelName(e.target.value)}
                placeholder="e.g. station_01_linear"
                className="w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm placeholder:text-text-muted focus:outline-none focus:border-accent-cyan/50"
              />
            </div>

            <button
              onClick={handleFit}
              disabled={fitMutation.isPending}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {fitMutation.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Calibrating...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Fit Model
                </>
              )}
            </button>
          </>
        )}

        {(fitMutation.isError || autoFitMutation.isError) && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            <p className="text-xs text-red-400">
              {(fitMutation.error ?? autoFitMutation.error) instanceof Error
                ? ((fitMutation.error ?? autoFitMutation.error) as Error).message
                : 'Calibration failed. Check backend logs.'}
            </p>
          </div>
        )}
      </div>

      {/* Right column — auto-fit results */}
      <div className="flex-1 min-w-0">
        {mode === 'guided' && autoFitResult && (
          <AutoFitResultsTable
            result={autoFitResult}
            onViewResults={viewResults}
          />
        )}

        {mode === 'guided' && !autoFitResult && !autoFitMutation.isPending && (
          <div className="flex items-center justify-center h-full text-text-muted">
            <div className="text-center space-y-2">
              <Zap className="w-10 h-10 mx-auto text-text-muted/30" />
              <p className="text-sm text-text-secondary">Ready to auto-fit</p>
              <p className="text-xs">Click "Auto-fit" to automatically find the best model configuration.</p>
            </div>
          </div>
        )}

        {mode === 'guided' && autoFitMutation.isPending && (
          <div className="flex items-center justify-center h-full text-text-muted">
            <div className="text-center space-y-3">
              <Loader2 className="w-10 h-10 mx-auto text-accent-cyan animate-spin" />
              <p className="text-sm text-text-secondary">Testing configurations...</p>
              <p className="text-xs">This may take a few minutes. Each candidate is fitted and evaluated.</p>
            </div>
          </div>
        )}

        {mode === 'expert' && !fitMutation.isPending && (
          <div className="flex items-center justify-center h-full text-text-muted">
            <div className="text-center space-y-2">
              <Play className="w-10 h-10 mx-auto text-text-muted/30" />
              <p className="text-sm text-text-secondary">Configure and fit</p>
              <p className="text-xs">Set your model parameters and click "Fit Model" to calibrate.</p>
            </div>
          </div>
        )}

        {mode === 'expert' && fitMutation.isPending && (
          <div className="flex items-center justify-center h-full text-text-muted">
            <div className="text-center space-y-3">
              <Loader2 className="w-10 h-10 mx-auto text-accent-cyan animate-spin" />
              <p className="text-sm text-text-secondary">Calibrating model...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// --- Auto-fit results table ---

function AutoFitResultsTable({ result, onViewResults }: {
  result: AutoFitResult
  onViewResults: (runId: string) => void
}) {
  const sorted = [...result.candidates].sort((a, b) => {
    if (a.error && !b.error) return 1
    if (!a.error && b.error) return -1
    return (b.evp ?? -Infinity) - (a.evp ?? -Infinity)
  })

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-text-primary">Auto-fit Results</h2>
          <p className="text-xs text-text-muted mt-0.5">
            {result.candidates.length} configurations tested in {result.total_elapsed_s.toFixed(1)}s
          </p>
        </div>
        {result.best_run_id && (
          <button
            onClick={() => onViewResults(result.best_run_id!)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 transition-colors"
          >
            <Check className="w-4 h-4" />
            View Best Results
          </button>
        )}
      </div>

      {/* Table */}
      <div className="bg-bg-card border border-white/5 rounded-xl overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-text-muted border-b border-white/5">
              <th className="text-left px-3 py-2">Configuration</th>
              <th className="text-right px-3 py-2">EVP</th>
              <th className="text-right px-3 py-2">NSE</th>
              <th className="text-right px-3 py-2">AIC</th>
              <th className="text-center px-3 py-2">STOWA</th>
              <th className="text-right px-3 py-2">Time</th>
              <th className="text-right px-3 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((c, i) => (
              <CandidateRow
                key={i}
                candidate={c}
                isBest={c.run_id === result.best_run_id}
                onViewResults={onViewResults}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function CandidateRow({ candidate, isBest, onViewResults }: {
  candidate: AutoFitCandidate
  isBest: boolean
  onViewResults: (runId: string) => void
}) {
  const c = candidate
  const configLabel = [
    c.config.recharge,
    c.config.response,
    c.config.noise !== 'none' ? c.config.noise : null,
    c.config.include_temp ? '+temp' : null,
  ].filter(Boolean).join(' / ')

  if (c.error) {
    return (
      <tr className="border-b border-white/5 text-red-400/70">
        <td className="px-3 py-2 font-mono">{configLabel}</td>
        <td colSpan={5} className="px-3 py-2">
          <div className="flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" />
            <span className="truncate">{c.error}</span>
          </div>
        </td>
        <td />
      </tr>
    )
  }

  return (
    <tr className={`border-b border-white/5 hover:bg-bg-hover ${isBest ? 'bg-accent-cyan/5' : ''}`}>
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          {isBest && <span className="text-accent-cyan text-[10px] font-semibold">BEST</span>}
          <span className="font-mono text-text-primary">{configLabel}</span>
        </div>
      </td>
      <td className="px-3 py-2 text-right text-text-primary">{c.evp != null ? `${c.evp.toFixed(1)}%` : '--'}</td>
      <td className="px-3 py-2 text-right text-text-primary">{c.nse != null ? c.nse.toFixed(3) : '--'}</td>
      <td className="px-3 py-2 text-right text-text-muted">{c.aic != null ? c.aic.toFixed(1) : '--'}</td>
      <td className="px-3 py-2 text-center">
        {c.stowa ? (
          <span className={`inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full ${
            c.stowa.overall_pass
              ? 'bg-green-500/10 text-green-400 border border-green-500/20'
              : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
          }`}>
            {c.stowa.overall_pass ? 'Pass' : 'Partial'}
          </span>
        ) : '--'}
      </td>
      <td className="px-3 py-2 text-right text-text-muted">{c.elapsed_s.toFixed(1)}s</td>
      <td className="px-3 py-2 text-right">
        {c.run_id && (
          <button
            onClick={() => onViewResults(c.run_id!)}
            className="text-accent-cyan hover:underline text-[10px]"
          >
            View
          </button>
        )}
      </td>
    </tr>
  )
}
