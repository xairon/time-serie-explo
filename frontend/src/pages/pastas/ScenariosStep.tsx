import { useSearchParams, useNavigate, Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { usePastasModel } from '@/hooks/usePastas'
import { ScenarioWorkflow } from '@/components/pastas/ScenarioWorkflow'

export default function ScenariosStep() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  const runId = searchParams.get('model')
  const codeBss = searchParams.get('station') ?? ''

  const { data: model, isLoading, isError } = usePastasModel(runId)

  if (!runId) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted">
        <div className="text-center space-y-3">
          <p className="text-sm text-text-secondary">No model selected</p>
          <p className="text-xs">
            Fit a model first, then come back to run what-if scenarios.
          </p>
          <button
            onClick={() => navigate(codeBss ? `/pastas/calibrate?station=${codeBss}` : '/pastas/calibrate')}
            className="flex items-center gap-2 text-accent-cyan text-sm hover:underline mx-auto"
          >
            <ArrowLeft className="w-4 h-4" />
            Go to Calibrate
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
          <p className="text-sm">Loading model...</p>
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

  const effectiveCodeBss = codeBss || model.code_bss

  return (
    <div className="p-6 space-y-4">
      {/* Breadcrumb */}
      <div className="flex items-center gap-3 text-xs text-text-muted">
        <Link
          to={`/pastas/station?station=${encodeURIComponent(effectiveCodeBss)}`}
          className="text-accent-cyan hover:underline"
        >
          {effectiveCodeBss}
        </Link>
        <span>/</span>
        <Link
          to={`/pastas/results?model=${runId}&station=${effectiveCodeBss}`}
          className="text-accent-cyan hover:underline"
        >
          Results
        </Link>
        <span>/</span>
        <span className="text-text-secondary">Scenarios</span>
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
