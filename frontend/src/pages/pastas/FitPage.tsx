import { useState } from 'react'
import { Loader2, Play } from 'lucide-react'
import { usePastasFit, usePastasPreview } from '@/hooks/usePastas'
import { StationPicker } from '@/components/pastas/StationPicker'
import { PastasConfigForm } from '@/components/pastas/PastasConfigForm'
import { FitResultsPanel } from '@/components/pastas/FitResultsPanel'
import { DataPreviewPanel } from '@/components/pastas/DataPreviewPanel'
import { StationMap } from '@/components/pastas/StationMap'
import type { PastasFitResponse } from '@/lib/types'

export default function FitPage() {
  const fitMutation = usePastasFit()

  // Station picker state
  const [codeBss, setCodeBss] = useState('')

  // Preview
  const { data: preview, isLoading: previewLoading } = usePastasPreview(codeBss || null)

  // Config form state
  const [recharge, setRecharge] = useState('Linear')
  const [response, setResponse] = useState('Gamma')
  const [noise, setNoise] = useState('ArNoiseModel')
  const [solver, setSolver] = useState('LeastSquares')
  const [tmin, setTmin] = useState('')
  const [tmax, setTmax] = useState('')
  const [modelName, setModelName] = useState('')

  // Result
  const [fitResult, setFitResult] = useState<PastasFitResponse | null>(null)

  const canFit = !!codeBss

  async function handleFit() {
    if (!canFit) return
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
      })
      setFitResult(result)
    } catch {
      // Error handled by mutation state
    }
  }

  return (
    <div className="p-6 flex gap-6 min-h-full">
      {/* Left column — configuration */}
      <div className="w-80 shrink-0 space-y-4">
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-4">Station</h2>
          <StationPicker codeBss={codeBss} onChange={setCodeBss} />
        </div>

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-4">Model configuration</h2>
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
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Run name (optional)
          </label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g. station_01_linear"
            className="w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm placeholder:text-text-muted focus:outline-none focus:border-accent-cyan/50"
          />
        </div>

        <button
          onClick={handleFit}
          disabled={!canFit || fitMutation.isPending}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {fitMutation.isPending ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Fitting…
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Fit Model
            </>
          )}
        </button>

        {fitMutation.isError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            <p className="text-xs text-red-400">
              {fitMutation.error instanceof Error
                ? fitMutation.error.message
                : 'Fit failed. Check backend logs.'}
            </p>
          </div>
        )}
      </div>

      {/* Right column — preview + results */}
      <div className="flex-1 min-w-0 space-y-4">
        {previewLoading && (
          <div className="flex items-center justify-center h-24 text-text-muted text-sm gap-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading preview…
          </div>
        )}

        {preview && !previewLoading && (
          <>
            <StationMap
              lat={typeof preview.metadata.latitude === 'number' ? preview.metadata.latitude : null}
              lon={typeof preview.metadata.longitude === 'number' ? preview.metadata.longitude : null}
              label={preview.code_bss}
            />
            <DataPreviewPanel
              preview={preview}
              onRangeChange={(t0, t1) => {
                setTmin(t0)
                setTmax(t1)
              }}
            />
          </>
        )}

        {!preview && !previewLoading && !codeBss && (
          <div className="flex items-center justify-center h-full text-text-muted text-sm">
            <div className="text-center space-y-2">
              <p className="text-text-secondary">No station selected</p>
              <p>Select a station to preview its data.</p>
            </div>
          </div>
        )}

        {fitResult && (
          <>
            {preview && <div className="border-t border-white/5" />}
            <FitResultsPanel result={fitResult} />
          </>
        )}
      </div>
    </div>
  )
}
