import { useSearchParams, useNavigate, Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { usePastasModel } from '@/hooks/usePastas'
import { usePastasMode } from './PastasLayout'
import { ScenarioWorkflow } from '@/components/pastas/ScenarioWorkflow'

export default function ScenariosStep() {
  const { t } = useTranslation()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const { pipeline } = usePastasMode()

  const runId = searchParams.get('model') ?? pipeline.selectedRunId
  const codeBss = searchParams.get('station') ?? pipeline.codeBss ?? ''

  const { data: model, isLoading, isError } = usePastasModel(runId)

  if (!runId) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        <div className="text-center space-y-3">
          <p className="text-sm text-text-secondary">{t('cleanup.scenarios.noModelSelected')}</p>
          <p className="text-xs">
            {t('cleanup.scenarios.calibrateFirst')}
          </p>
          <button
            onClick={() => navigate(codeBss ? `/pastas/calibrate?station=${codeBss}` : '/pastas/calibrate')}
            className="flex items-center gap-2 text-accent-cyan text-sm hover:underline mx-auto"
          >
            <ArrowLeft className="w-4 h-4" />
            {t('cleanup.scenarios.goToCalibration')}
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
          <p className="text-sm">{t('cleanup.scenarios.loadingModel')}</p>
        </div>
      </div>
    )
  }

  if (isError || !model) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        <div className="text-center space-y-3">
          <p className="text-sm text-red-400">{t('cleanup.scenarios.modelLoadFailed')}</p>
          <button
            onClick={() => navigate(codeBss ? `/pastas/calibrate?station=${codeBss}` : '/pastas/calibrate')}
            className="flex items-center gap-2 text-accent-cyan text-sm hover:underline mx-auto"
          >
            <ArrowLeft className="w-4 h-4" />
            {t('cleanup.scenarios.backToCalibration')}
          </button>
        </div>
      </div>
    )
  }

  const effectiveCodeBss = codeBss || model.code_bss

  return (
    <div className="p-6 space-y-4">
      {/* Breadcrumb */}
      <div className="flex items-center gap-3 text-xs text-text-muted">
        <Link
          to={`/station/piezo/${encodeURIComponent(effectiveCodeBss)}`}
          className="text-accent-cyan hover:underline"
        >
          {effectiveCodeBss}
        </Link>
        <span>/</span>
        <Link
          to={`/pastas/results?model=${runId}&station=${effectiveCodeBss}`}
          className="text-accent-cyan hover:underline"
        >
          {t('cleanup.scenarios.results')}
        </Link>
        <span>/</span>
        <span className="text-text-secondary">{t('cleanup.scenarios.scenarios')}</span>
      </div>

      {/* Scenario workflow */}
      <ScenarioWorkflow
        model={model}
        codeBss={effectiveCodeBss}
        onRefit={(newResult) => {
          navigate(`/pastas/results?model=${newResult.run_id}&station=${effectiveCodeBss}`)
        }}
      />
    </div>
  )
}
