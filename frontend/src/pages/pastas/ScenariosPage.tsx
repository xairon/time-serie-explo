import { useState } from 'react'
import { Loader2, Play } from 'lucide-react'
import { usePastasModels, usePastasSimulate } from '@/hooks/usePastas'
import { ScenarioComposer } from '@/components/pastas/ScenarioComposer'
import { ScenarioResultsPanel } from '@/components/pastas/ScenarioResultsPanel'
import type { PastasScenarioResponse } from '@/lib/types'
import type { ModificationData } from '@/components/pastas/ModificationCard'

function modificationToPayload(mod: ModificationData): Record<string, unknown> {
  return mod as unknown as Record<string, unknown>
}

export default function ScenariosPage() {
  const { data: models = [] } = usePastasModels()
  const simulateMutation = usePastasSimulate()

  const [runId, setRunId] = useState('')
  const [tmin, setTmin] = useState('')
  const [tmax, setTmax] = useState('')
  const [modifications, setModifications] = useState<ModificationData[]>([])
  const [simResult, setSimResult] = useState<PastasScenarioResponse | null>(null)

  const canSimulate = !!runId && !!tmin && !!tmax

  async function handleSimulate() {
    if (!canSimulate) return
    try {
      const result = await simulateMutation.mutateAsync({
        run_id: runId,
        tmin,
        tmax,
        modifications: modifications.map(modificationToPayload),
      })
      setSimResult(result)
    } catch {
      // Error handled by mutation state
    }
  }

  const inputClass =
    'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

  return (
    <div className="p-6 flex gap-6 min-h-full">
      {/* Left column — scenario configuration */}
      <div className="w-80 shrink-0 space-y-4">
        {/* Model picker */}
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-3">Model</h2>
          <select
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            className={inputClass}
          >
            <option value="">-- Select a fitted model --</option>
            {models.map((m) => (
              <option key={m.run_id} value={m.run_id}>
                {m.name || m.run_id.slice(0, 8)} — {m.code_bss} ({m.response_type}, EVP {m.evp?.toFixed(1) ?? '?'}%)
              </option>
            ))}
          </select>
        </div>

        {/* Simulation window */}
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-3">Simulation window</h2>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-text-muted mb-1">Start (tmin)</label>
              <input
                type="date"
                value={tmin}
                onChange={(e) => setTmin(e.target.value)}
                className={inputClass}
              />
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">End (tmax)</label>
              <input
                type="date"
                value={tmax}
                onChange={(e) => setTmax(e.target.value)}
                className={inputClass}
              />
            </div>
          </div>
        </div>

        {/* Modifications */}
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-3">Modifications</h2>
          <ScenarioComposer modifications={modifications} onChange={setModifications} />
        </div>

        {/* Simulate button */}
        <button
          onClick={handleSimulate}
          disabled={!canSimulate || simulateMutation.isPending}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {simulateMutation.isPending ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Simulating…
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Simulate Scenario
            </>
          )}
        </button>

        {simulateMutation.isError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            <p className="text-xs text-red-400">
              {simulateMutation.error instanceof Error
                ? simulateMutation.error.message
                : 'Simulation failed. Check backend logs.'}
            </p>
          </div>
        )}
      </div>

      {/* Right column — results */}
      <div className="flex-1 min-w-0">
        {simResult ? (
          <ScenarioResultsPanel result={simResult} />
        ) : (
          <div className="flex items-center justify-center h-full text-text-muted text-sm">
            <div className="text-center space-y-2">
              <p className="text-text-secondary">No simulation yet</p>
              <p>Select a model, set the window, add modifications, and click "Simulate Scenario".</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
