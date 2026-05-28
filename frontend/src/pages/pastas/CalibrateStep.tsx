import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Loader2, Play, Zap, ArrowLeft, Check, AlertTriangle, RotateCcw } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { usePastasFit, usePastasAutoFit, usePastasStationInfo } from '@/hooks/usePastas'
import { InfoTip } from '@/components/pastas/InfoTip'
import { usePastasMode } from './PastasLayout'
import { PastasConfigForm } from '@/components/pastas/PastasConfigForm'
import { CalValToggle } from '@/components/pastas/CalValToggle'
import type { AutoFitResult, AutoFitCandidate } from '@/lib/types'

export default function CalibrateStep() {
  const { t } = useTranslation()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const { mode, setMode, pipeline, setCodeBss, setAutoFitResult: setPipelineAutoFit, selectModel } = usePastasMode()

  const codeBss = searchParams.get('station') ?? pipeline.codeBss ?? ''
  const recommendedTmin = searchParams.get('tmin') ?? ''
  const recommendedTmax = searchParams.get('tmax') ?? ''

  // Refit params from Gallery
  const refitRecharge = searchParams.get('recharge')
  const refitResponse = searchParams.get('response')
  const refitNoise = searchParams.get('noise')
  const refitSolver = searchParams.get('solver')
  const refitIncludeTemp = searchParams.get('include_temp') === '1'
  const isRefit = !!(refitRecharge || refitResponse || refitNoise || refitSolver)

  // Sync station to pipeline context
  useEffect(() => {
    if (codeBss) setCodeBss(codeBss)
  }, [codeBss, setCodeBss])

  // Switch to expert mode when refitting from gallery
  useEffect(() => {
    if (isRefit) setMode('expert')
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Station info for BDLISA preset
  const { data: stationInfo } = usePastasStationInfo(codeBss || null)

  // Config form state (expert mode) — refit params take priority
  const [recharge, setRecharge] = useState(refitRecharge || 'Linear')
  const [response, setResponse] = useState(refitResponse || 'Gamma')
  const [noise, setNoise] = useState(refitNoise || 'ArNoiseModel')
  const [solver, setSolver] = useState(refitSolver || 'LeastSquares')
  const [tmin, setTmin] = useState(recommendedTmin)
  const [tmax, setTmax] = useState(recommendedTmax)
  const [modelName, setModelName] = useState('')
  const [valSplit, setValSplit] = useState<number | null>(0.3)
  const [includeTemp, setIncludeTemp] = useState(refitIncludeTemp)
  const [warmUpYears, setWarmUpYears] = useState(1)
  const [twoPass, setTwoPass] = useState(false)

  // Apply BDLISA preset (only when not refitting)
  const [presetApplied, setPresetApplied] = useState(isRefit ? codeBss : '')
  useEffect(() => {
    if (!isRefit && stationInfo?.preset && codeBss !== presetApplied) {
      const p = stationInfo.preset as Record<string, string>
      if (p.recharge) setRecharge(p.recharge)
      if (p.response) setResponse(p.response)
      if (p.noise) setNoise(p.noise)
      setPresetApplied(codeBss)
    }
  }, [stationInfo, codeBss, presetApplied, isRefit])

  // Mutations
  const fitMutation = usePastasFit()
  const autoFitMutation = usePastasAutoFit()

  // Auto-fit state from pipeline context (persists across tab switches)
  const autoFitResult = pipeline.autoFitResult

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
      selectModel(result.run_id)
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
      setPipelineAutoFit(result)
      // Auto-select the best candidate
      if (result.best_run_id) {
        const bestCandidate = result.candidates.find(c => c.run_id === result.best_run_id)
        selectModel(result.best_run_id, bestCandidate?.stowa ?? null)
      }
    } catch {
      // Error handled by mutation state
    }
  }

  function viewResults(runId: string, stowa?: AutoFitCandidate['stowa']) {
    selectModel(runId, stowa ?? null)
    navigate(`/pastas/results?model=${runId}&station=${codeBss}`)
  }

  if (!codeBss) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        <div className="text-center space-y-3">
          <p className="text-sm text-text-secondary">{t('pastas.station.noSelection')}</p>
          <button
            onClick={() => navigate('/pastas/station')}
            className="flex items-center gap-2 text-accent-cyan text-sm hover:underline mx-auto"
          >
            <ArrowLeft className="w-4 h-4" />
            {t('pastas.ui.backToStationStep')}
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
            <div className="text-[10px] text-text-muted uppercase tracking-wide">{t('pastas.ui.stationSection')}</div>
            <div className="text-sm font-mono text-accent-cyan">{codeBss}</div>
          </div>
          <button
            onClick={() => navigate(`/pastas/station?station=${encodeURIComponent(codeBss)}`)}
            className="text-xs text-text-muted hover:text-text-secondary"
          >
            {t('pastas.ui.change')}
          </button>
        </div>

        {mode === 'guided' ? (
          /* Guided mode — auto-fit */
          <>
            <div className="bg-accent-cyan/5 border border-accent-cyan/20 rounded-xl p-4">
              <h2 className="text-sm font-semibold text-accent-cyan mb-2">{t('pastas.calibrate.autofitButton')}</h2>
              <p className="text-xs text-text-muted mb-3">
                {t('pastas.calibrateExtra.autoFitDescription')}
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
                  <span className="text-xs text-text-secondary flex items-center gap-1">
                    {t('pastas.calibrate.includeTemp')}
                    <InfoTip text={t('pastas.calibrate.includeTempTooltip')} />
                  </span>
                </label>

                <div>
                  <label className="text-xs text-text-muted mb-1 flex items-center gap-1">
                    {t('pastas.calibrate.warmupPeriod', { years: warmUpYears, plural: warmUpYears !== 1 ? 's' : '' })}
                    <InfoTip text={t('pastas.calibrate.warmupTooltip')} />
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
                  {t('pastas.calibrateExtra.autoFitInProgress')}
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  {t('pastas.calibrate.autofitButton')}
                </>
              )}
            </button>
          </>
        ) : (
          /* Expert mode — manual config */
          <>
            {isRefit && (
              <div className="bg-purple-500/10 border border-purple-500/20 rounded-xl p-3 flex items-center gap-2">
                <RotateCcw className="w-4 h-4 text-purple-400 shrink-0" />
                <span className="text-xs text-purple-300">
                  {t('pastas.calibrateExtra.refitFromGallery')}
                </span>
              </div>
            )}

            <div className="bg-bg-card border border-white/5 rounded-xl p-4">
              <h2 className="text-sm font-semibold text-text-primary mb-2">{t('pastas.calibrateExtra.metricConfig')}</h2>
              {stationInfo?.preset != null && (stationInfo.preset as Record<string, string>).label && (
                <div className="mb-3 flex items-center gap-2 bg-accent-cyan/5 border border-accent-cyan/20 rounded-lg px-3 py-1.5">
                  <span className="text-xs text-accent-cyan font-medium">
                    {t('pastas.calibrateExtra.recommended', { label: (stationInfo.preset as Record<string, string>).label })}
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
                  <span className="text-sm font-medium text-text-secondary flex items-center gap-1">
                    {t('pastas.calibrate.includeTemp')}
                    <InfoTip text={t('pastas.fit.includeTempLongTooltip')} />
                  </span>
                  <p className="text-xs text-text-muted">{t('pastas.fit.includeTempDesc')}</p>
                </div>
              </label>
            </div>

            <div className="bg-bg-card border border-white/5 rounded-xl p-4">
              <label className="text-xs text-text-muted mb-1 flex items-center gap-1">
                {t('pastas.calibrate.warmupPeriod', { years: warmUpYears, plural: warmUpYears !== 1 ? 's' : '' })}
                <InfoTip text={t('pastas.calibrate.warmupTooltip')} />
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

            <div className="bg-bg-card border border-white/5 rounded-xl p-4">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={twoPass}
                  onChange={e => setTwoPass(e.target.checked)}
                  className="accent-accent-cyan w-4 h-4"
                />
                <div>
                  <span className="text-sm font-medium text-text-secondary flex items-center gap-1">
                    {t('pastas.calibrate.twoPass')}
                    <InfoTip text={t('pastas.calibrate.twoPassTooltip')} />
                  </span>
                  <p className="text-xs text-text-muted">{t('pastas.calibrate.twoPassSub')}</p>
                </div>
              </label>
            </div>

            <div className="bg-bg-card border border-white/5 rounded-xl p-4">
              <label className="block text-sm font-medium text-text-secondary mb-1">
                {t('pastas.calibrate.modelName')}
              </label>
              <input
                type="text"
                value={modelName}
                onChange={e => setModelName(e.target.value)}
                placeholder={t('pastas.calibrate.modelNamePlaceholder')}
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
                  {t('pastas.calibrate.modelCalibrating')}
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  {t('pastas.calibrate.calibrateButton')}
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
                : t('pastas.fit.calibrationFailed')}
            </p>
          </div>
        )}
      </div>

      {/* Right column — auto-fit results */}
      <div className="flex-1 min-w-0">
        {mode === 'guided' && autoFitResult && (
          <>
            <AutoFitResultsTable
              result={autoFitResult}
              onSelect={(runId, stowa) => selectModel(runId, stowa ?? null)}
              selectedRunId={pipeline.selectedRunId}
            />
            {pipeline.selectedRunId && (
              <div className="flex justify-end mt-4">
                <button
                  onClick={() => viewResults(pipeline.selectedRunId!, autoFitResult.candidates.find(c => c.run_id === pipeline.selectedRunId)?.stowa)}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 transition-colors"
                >
                  <Check className="w-4 h-4" />
                  {t('pastas.calibrate.viewResults')}
                </button>
              </div>
            )}
          </>
        )}

        {mode === 'guided' && !autoFitResult && !autoFitMutation.isPending && (
          <div className="flex items-center justify-center h-full text-text-muted">
            <div className="text-center space-y-2">
              <Zap className="w-10 h-10 mx-auto text-text-muted/30" />
              <p className="text-sm text-text-secondary">{t('pastas.calibrate.readyForAutofit')}</p>
            </div>
          </div>
        )}

        {mode === 'guided' && autoFitMutation.isPending && (
          <div className="flex items-center justify-center h-full text-text-muted">
            <div className="text-center space-y-3">
              <Loader2 className="w-10 h-10 mx-auto text-accent-cyan animate-spin" />
              <p className="text-sm text-text-secondary">{t('pastas.calibrate.autofitInProgress')}</p>
            </div>
          </div>
        )}

        {mode === 'expert' && !fitMutation.isPending && (
          <div className="flex items-center justify-center h-full text-text-muted">
            <div className="text-center space-y-2">
              <Play className="w-10 h-10 mx-auto text-text-muted/30" />
              <p className="text-sm text-text-secondary">{t('pastas.calibrate.configureAndCalibrate')}</p>
            </div>
          </div>
        )}

        {mode === 'expert' && fitMutation.isPending && (
          <div className="flex items-center justify-center h-full text-text-muted">
            <div className="text-center space-y-3">
              <Loader2 className="w-10 h-10 mx-auto text-accent-cyan animate-spin" />
              <p className="text-sm text-text-secondary">{t('pastas.calibrate.modelCalibrating')}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// --- Auto-fit results table ---

function AutoFitResultsTable({ result, onSelect, selectedRunId }: {
  result: AutoFitResult
  onSelect: (runId: string, stowa?: AutoFitCandidate['stowa']) => void
  selectedRunId: string | null
}) {
  const { t } = useTranslation()
  const sorted = [...result.candidates].sort((a, b) => {
    if (a.error && !b.error) return 1
    if (!a.error && b.error) return -1
    return (b.nse ?? -Infinity) - (a.nse ?? -Infinity)
  })

  const errors = sorted.filter(c => c.error)
  const uniqueErrors = [...new Set(errors.map(c => c.error))]
  const allFailed = errors.length === sorted.length && uniqueErrors.length === 1

  if (allFailed) {
    return (
      <div className="space-y-4">
        <div>
          <h2 className="text-sm font-semibold text-text-primary">{t('pastas.calibrate.autofitResults')}</h2>
          <p className="text-xs text-text-muted mt-0.5">
            {t('pastas.calibrate.configsTested', { count: result.candidates.length, seconds: result.total_elapsed_s.toFixed(1) })}
          </p>
        </div>
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-400">{t('pastas.calibrate.allFailed')}</p>
            <p className="text-xs text-red-300/80 mt-1">{uniqueErrors[0]}</p>
            <p className="text-xs text-text-muted mt-2">{t('pastas.calibrate.insufficientOverlap')}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-text-primary">{t('pastas.calibrate.autofitResults')}</h2>
          <p className="text-xs text-text-muted mt-0.5">
            {t('pastas.calibrate.configsTested', { count: result.candidates.length, seconds: result.total_elapsed_s.toFixed(1) })}
          </p>
        </div>
      </div>

      {/* Table */}
      <div className="bg-bg-card border border-white/5 rounded-xl overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-text-muted border-b border-white/5">
              <th className="text-left px-3 py-2">{t('pastas.calibrateExtra.metricConfig')}</th>
              <th className="text-right px-3 py-2">
                <span className="inline-flex items-center gap-1">NSE <InfoTip text={t('pastas.calibrateExtra.nseTip')} /></span>
              </th>
              <th className="text-right px-3 py-2">
                <span className="inline-flex items-center gap-1">RMSE <InfoTip text={t('pastas.calibrateExtra.rmseTip')} /></span>
              </th>
              <th className="text-right px-3 py-2">
                <span className="inline-flex items-center gap-1">KGE <InfoTip text={t('pastas.calibrateExtra.kgeTip')} /></span>
              </th>
              <th className="text-center px-3 py-2">
                <span className="inline-flex items-center gap-1">STOWA <InfoTip text={t('pastas.calibrateExtra.stowaTip')} /></span>
              </th>
              <th className="text-right px-3 py-2">{t('pastas.calibrateExtra.metricTime')}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((c, i) => (
              <CandidateRow
                key={i}
                candidate={c}
                isBest={c.run_id === result.best_run_id}
                isSelected={c.run_id === selectedRunId}
                onSelect={onSelect}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function nseColor(v: number | null): string {
  if (v == null) return 'text-text-muted'
  if (v > 0.7) return 'text-green-400'
  if (v > 0.4) return 'text-accent-cyan'
  return 'text-red-400'
}

function CandidateRow({ candidate, isBest, isSelected, onSelect }: {
  candidate: AutoFitCandidate
  isBest: boolean
  isSelected: boolean
  onSelect: (runId: string, stowa?: AutoFitCandidate['stowa']) => void
}) {
  const { t } = useTranslation()
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
        <td colSpan={4} className="px-3 py-2">
          <div className="flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" />
            <span className="truncate">{c.error}</span>
          </div>
        </td>
        <td />
      </tr>
    )
  }

  const selectable = !!c.run_id

  return (
    <tr
      onClick={() => selectable && onSelect(c.run_id!, c.stowa)}
      className={`border-b border-white/5 transition-colors ${
        selectable ? 'cursor-pointer hover:bg-bg-hover' : 'opacity-50'
      } ${isSelected ? 'bg-accent-cyan/10 border-l-2 border-l-accent-cyan' : isBest ? 'bg-accent-cyan/5' : ''}`}
    >
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          {isSelected && <Check className="w-3 h-3 text-accent-cyan shrink-0" />}
          {!isSelected && isBest && <span className="text-accent-cyan text-[10px] font-semibold shrink-0">{t('pastas.calibrateExtra.best')}</span>}
          <span className="font-mono text-text-primary">{configLabel}</span>
        </div>
      </td>
      <td className={`px-3 py-2 text-right font-mono ${nseColor(c.nse)}`}>{typeof c.nse === 'number' ? c.nse.toFixed(3) : '--'}</td>
      <td className="px-3 py-2 text-right font-mono text-text-primary">{typeof c.rmse === 'number' ? c.rmse.toFixed(4) : '--'}</td>
      <td className={`px-3 py-2 text-right font-mono ${nseColor(c.kge)}`}>{typeof c.kge === 'number' ? c.kge.toFixed(3) : '--'}</td>
      <td className="px-3 py-2 text-center">
        {c.stowa ? (
          <span className={`inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full ${
            c.stowa.overall_pass
              ? 'bg-green-500/10 text-green-400 border border-green-500/20'
              : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
          }`}>
            {c.stowa.overall_pass ? t('pastas.calibrateExtra.stowaPass') : t('pastas.calibrateExtra.stowaPartial')}
          </span>
        ) : '--'}
      </td>
      <td className="px-3 py-2 text-right text-text-muted">{c.elapsed_s.toFixed(1)} s</td>
    </tr>
  )
}
