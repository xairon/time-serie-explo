import { useState } from 'react'
import { Loader2, Play } from 'lucide-react'
import { useDatasets } from '@/hooks/useDatasets'
import { usePastasFit } from '@/hooks/usePastas'
import { StationPicker } from '@/components/pastas/StationPicker'
import { PastasConfigForm } from '@/components/pastas/PastasConfigForm'
import { FitResultsPanel } from '@/components/pastas/FitResultsPanel'
import type { PastasFitResponse } from '@/lib/types'

export default function FitPage() {
  const { data: datasets = [] } = useDatasets()
  const fitMutation = usePastasFit()

  // Station picker state
  const [datasetId, setDatasetId] = useState('')
  const [stationId, setStationId] = useState('')
  const [precipColumn, setPrecipColumn] = useState('')
  const [evapColumn, setEvapColumn] = useState('')

  // Config form state
  const [recharge, setRecharge] = useState('Linear')
  const [response, setResponse] = useState('Gamma')
  const [noise, setNoise] = useState('ArmaModel')
  const [solver, setSolver] = useState('LeastSquares')
  const [tmin, setTmin] = useState('')
  const [tmax, setTmax] = useState('')
  const [modelName, setModelName] = useState('')

  // Result
  const [fitResult, setFitResult] = useState<PastasFitResponse | null>(null)

  const canFit = !!datasetId && !!precipColumn && !!evapColumn

  async function handleFit() {
    if (!canFit) return
    try {
      const result = await fitMutation.mutateAsync({
        dataset_id: datasetId,
        station_id: stationId || undefined,
        precip_column: precipColumn,
        evap_column: evapColumn,
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
          <h2 className="text-sm font-semibold text-text-primary mb-4">Data selection</h2>
          <StationPicker
            datasets={datasets}
            datasetId={datasetId}
            onDatasetChange={(id) => {
              setDatasetId(id)
              setStationId('')
              setPrecipColumn('')
              setEvapColumn('')
            }}
            stationId={stationId}
            onStationChange={setStationId}
            precipColumn={precipColumn}
            onPrecipChange={setPrecipColumn}
            evapColumn={evapColumn}
            onEvapChange={setEvapColumn}
          />
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

      {/* Right column — results */}
      <div className="flex-1 min-w-0">
        {fitResult ? (
          <FitResultsPanel result={fitResult} />
        ) : (
          <div className="flex items-center justify-center h-full text-text-muted text-sm">
            <div className="text-center space-y-2">
              <p className="text-text-secondary">No results yet</p>
              <p>Select a dataset, configure the model, and click "Fit Model".</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
