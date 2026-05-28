import { useSearchParams, useNavigate, Link } from 'react-router-dom'
import { useEffect } from 'react'
import { ArrowLeft, FlaskConical } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { usePastasModel } from '@/hooks/usePastas'
import { usePastasMode } from './PastasLayout'
import { FitResultsPanel } from '@/components/pastas/FitResultsPanel'

export default function ResultsStep() {
  const { t } = useTranslation()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const { pipeline, setCodeBss, selectModel } = usePastasMode()

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
          <p className="text-sm text-text-secondary">{t('pastas.ui.noModelSelected')}</p>
          <button
            onClick={() => navigate(codeBss ? `/pastas/calibrate?station=${codeBss}` : '/pastas/calibrate')}
            className="flex items-center gap-2 text-accent-cyan text-sm hover:underline mx-auto"
          >
            <ArrowLeft className="w-4 h-4" />
            {t('pastas.ui.backToCalibration')}
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
          <p className="text-sm">{t('pastas.ui.loadingResults')}</p>
        </div>
      </div>
    )
  }

  if (isError || !model) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        <div className="text-center space-y-3">
          <p className="text-sm text-red-400">{t('pastas.ui.modelLoadFailed')}</p>
          <button
            onClick={() => navigate(codeBss ? `/pastas/calibrate?station=${codeBss}` : '/pastas/calibrate')}
            className="flex items-center gap-2 text-accent-cyan text-sm hover:underline mx-auto"
          >
            <ArrowLeft className="w-4 h-4" />
            {t('pastas.ui.backToCalibration')}
          </button>
        </div>
      </div>
    )
  }

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
          {t('pastas.resultsExtra.runScenarios')}
        </Link>
      </div>

      <FitResultsPanel
        result={model}
        codeBss={effectiveCodeBss}
      />
    </div>
  )
}
