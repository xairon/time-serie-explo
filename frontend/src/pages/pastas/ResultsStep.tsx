import { useSearchParams, useNavigate, Link } from 'react-router-dom'
import { useEffect } from 'react'
import { ArrowLeft, FlaskConical } from 'lucide-react'
import { usePastasModel } from '@/hooks/usePastas'
import { usePastasMode } from './PastasLayout'
import { FitResultsPanel } from '@/components/pastas/FitResultsPanel'
import { StowaVerdictBanner } from '@/components/pastas/StowaVerdictBanner'
import type { StowaResult } from '@/lib/types'

/**
 * Approximate STOWA verdict from client-side metrics.
 * The real STOWA comes from the backend auto-fit endpoint; this is a fallback
 * for models fitted via the manual path that don't have a full STOWA response.
 */
function approximateStowa(metrics: Record<string, number>): StowaResult | null {
  const evp = metrics.evp
  if (evp == null) return null

  return {
    evp_pass: evp >= 70,
    evp_value: evp,
    autocorrelation_pass: null,
    runs_test_pvalue: null,
    t95_pass: null,
    t95_days: null,
    t95_threshold: null,
    gain_pass: null,
    gain_significance: null,
    overall_pass: null,
    suggestions: evp < 70
      ? ['EVP inférieur à 70%. Essayez d\'autres configurations ou ajoutez des stress.']
      : ['Résultat partiel — lancez l\'auto-fit pour l\'évaluation STOWA complète.'],
  }
}

export default function ResultsStep() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const { mode, pipeline, setCodeBss, selectModel } = usePastasMode()

  const runId = searchParams.get('model') ?? pipeline.selectedRunId
  const codeBss = searchParams.get('station') ?? pipeline.codeBss ?? ''

  // Sync to pipeline context — pass null for STOWA when station changed (stale closure)
  useEffect(() => {
    const sameStation = codeBss === pipeline.codeBss
    if (codeBss) setCodeBss(codeBss)
    if (runId) selectModel(runId, sameStation ? pipeline.selectedStowa : null)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const { data: model, isLoading, isError } = usePastasModel(runId)

  if (!runId) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        <div className="text-center space-y-3">
          <p className="text-sm text-text-secondary">No model selected</p>
          <button
            onClick={() => navigate(codeBss ? `/pastas/calibrate?station=${codeBss}` : '/pastas/calibrate')}
            className="flex items-center gap-2 text-accent-cyan text-sm hover:underline mx-auto"
          >
            <ArrowLeft className="w-4 h-4" />
            Go back to Calibrate
          </button>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        <div className="text-center space-y-2">
          <div className="w-8 h-8 border-2 border-accent-cyan/30 border-t-accent-cyan rounded-full animate-spin mx-auto" />
          <p className="text-sm">Loading model results...</p>
        </div>
      </div>
    )
  }

  if (isError || !model) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        <div className="text-center space-y-3">
          <p className="text-sm text-red-400">Failed to load model</p>
          <button
            onClick={() => navigate(codeBss ? `/pastas/calibrate?station=${codeBss}` : '/pastas/calibrate')}
            className="flex items-center gap-2 text-accent-cyan text-sm hover:underline mx-auto"
          >
            <ArrowLeft className="w-4 h-4" />
            Go back to Calibrate
          </button>
        </div>
      </div>
    )
  }

  // Use real STOWA from auto-fit pipeline if available, otherwise approximate
  const stowa = pipeline.selectedStowa ?? approximateStowa(model.metrics)
  const effectiveCodeBss = codeBss || model.code_bss

  return (
    <div className="p-6 space-y-4">
      {/* Navigation breadcrumb */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs text-text-muted">
          <Link
            to={`/pastas/station?station=${encodeURIComponent(effectiveCodeBss)}`}
            className="text-accent-cyan hover:underline"
          >
            {effectiveCodeBss}
          </Link>
          <span>/</span>
          <span className="text-text-secondary font-mono">{runId?.slice(0, 8)}</span>
        </div>
        <Link
          to={`/pastas/scenarios?model=${runId}&station=${effectiveCodeBss}`}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-500/15 text-purple-400 text-sm font-medium border border-purple-500/25 hover:bg-purple-500/25 transition-colors"
        >
          <FlaskConical className="w-4 h-4" />
          Run Scenarios
        </Link>
      </div>

      {/* STOWA verdict */}
      {stowa && (
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-3">
            STOWA Assessment
          </h3>
          <StowaVerdictBanner stowa={stowa} />
        </div>
      )}

      {/* Full results panel — collapse sections 5-7 in guided mode */}
      <FitResultsPanel
        result={model}
        codeBss={effectiveCodeBss}
        {...(mode === 'guided' ? {} : {})}
      />

      {/* In guided mode, add a note about collapsed sections */}
      {mode === 'guided' && (
        <div className="text-xs text-text-muted text-center py-2">
          Switch to Expert mode for detailed parameters, diagnostics, and signatures.
        </div>
      )}
    </div>
  )
}
