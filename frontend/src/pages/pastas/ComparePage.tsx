import { useState } from 'react'
import { Loader2, GitCompareArrows } from 'lucide-react'
import { usePastasModels, usePastasCompare } from '@/hooks/usePastas'
import { ComparisonView } from '@/components/pastas/ComparisonView'

export default function ComparePage() {
  const { data: models } = usePastasModels()
  const compareMut = usePastasCompare()
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  const toggleModel = (runId: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(runId)) next.delete(runId)
      else if (next.size < 5) next.add(runId)
      return next
    })
  }

  const handleCompare = () => {
    compareMut.mutate(Array.from(selectedIds))
  }

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-semibold text-text-primary">Pastas — Compare Models</h1>

      {/* Model selection */}
      <div className="bg-bg-card rounded-lg border border-white/5 p-4">
        <h2 className="text-sm font-semibold text-text-secondary mb-3">
          Select 2-5 models to compare
        </h2>
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {(models ?? []).map(m => (
            <label key={m.run_id}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                selectedIds.has(m.run_id) ? 'bg-accent-cyan/10' : 'hover:bg-bg-hover'
              }`}>
              <input
                type="checkbox"
                checked={selectedIds.has(m.run_id)}
                onChange={() => toggleModel(m.run_id)}
                className="accent-accent-cyan"
              />
              <span className="text-sm text-text-primary">{m.name || m.run_id.slice(0, 8)}</span>
              <span className="text-xs text-accent-cyan font-mono">{m.code_bss}</span>
              <span className="text-xs text-text-muted">{m.response_type}</span>
              <span className="text-xs text-text-muted ml-auto">EVP {m.evp?.toFixed(1) ?? '?'}%</span>
            </label>
          ))}
          {(!models || models.length === 0) && (
            <p className="text-sm text-text-muted py-4 text-center">No models fitted yet.</p>
          )}
        </div>

        <button
          onClick={handleCompare}
          disabled={selectedIds.size < 2 || compareMut.isPending}
          className="mt-3 flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {compareMut.isPending ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <GitCompareArrows className="w-4 h-4" />
          )}
          Compare ({selectedIds.size})
        </button>

        {compareMut.isError && (
          <div className="mt-2 bg-red-500/10 border border-red-500/30 rounded-lg p-2">
            <p className="text-xs text-red-400">{(compareMut.error as Error).message}</p>
          </div>
        )}
      </div>

      {/* Comparison results */}
      {compareMut.data && <ComparisonView models={compareMut.data.models} />}
    </div>
  )
}
