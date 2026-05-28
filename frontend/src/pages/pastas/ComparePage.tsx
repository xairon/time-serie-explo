import { useState } from 'react'
import { Loader2, GitCompareArrows } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { usePastasModels, usePastasCompare } from '@/hooks/usePastas'
import { ComparisonView } from '@/components/pastas/ComparisonView'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'

export default function ComparePage() {
  const { t } = useTranslation()
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
      <OnboardingBanner
        id="compare"
        title={t('cleanup.pastasCompare.title')}
        description={t('cleanup.pastasCompare.description')}
        steps={[
          t('cleanup.pastasCompare.step1'),
          t('cleanup.pastasCompare.step2'),
          t('cleanup.pastasCompare.step3'),
          t('cleanup.pastasCompare.step4'),
        ]}
      />

      {/* Model selection */}
      <div className="bg-bg-card rounded-lg border border-white/5 p-4">
        <h2 className="text-sm font-semibold text-text-secondary mb-3">
          {t('cleanup.pastasCompare.selectModelsPrompt')}
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
            <p className="text-sm text-text-muted py-4 text-center">{t('cleanup.pastasCompare.noCalibrated')}</p>
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
          {t('cleanup.pastasCompare.compareWithCount', { count: selectedIds.size })}
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
