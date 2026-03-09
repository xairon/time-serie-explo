import { useState } from 'react'
import { Download } from 'lucide-react'
import { CFConfigForm } from '@/components/counterfactual/CFConfigForm'
import { CFResultView } from '@/components/counterfactual/CFResultView'
import { IPSPanel } from '@/components/counterfactual/IPSPanel'
import { PastasPanel } from '@/components/counterfactual/PastasPanel'
import { RadarPlot } from '@/components/charts/RadarPlot'
import { useCounterfactualRun } from '@/hooks/useCounterfactual'
import type { CounterfactualResult } from '@/lib/types'

type RightTab = 'ips' | 'radar' | 'pastas'

export default function CounterfactualPage() {
  const [rightTab, setRightTab] = useState<RightTab>('ips')
  const cfMutation = useCounterfactualRun()

  const result: CounterfactualResult | null = cfMutation.data ?? null

  const handleSubmit = (config: {
    model_id: string
    dataset_id: string
    method: string
    modifications: Record<string, number>
  }) => {
    cfMutation.mutate({
      model_id: config.model_id,
      dataset_id: config.dataset_id,
      modifications: config.modifications,
    })
  }

  const handleExport = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `counterfactual_${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const rightTabs: { key: RightTab; label: string }[] = [
    { key: 'ips', label: 'IPS' },
    { key: 'radar', label: 'Radar' },
    { key: 'pastas', label: 'Pastas' },
  ]

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Analyse contrefactuelle</h1>
          <p className="text-sm text-text-secondary">
            Simulation de scénarios alternatifs pour comprendre l'impact des covariables
          </p>
        </div>
        {result && (
          <button
            onClick={handleExport}
            className="flex items-center gap-2 bg-bg-hover text-text-primary px-4 py-2 rounded-lg border border-white/10 hover:bg-bg-hover/80 transition-colors text-sm"
          >
            <Download className="w-4 h-4" />
            Exporter JSON
          </button>
        )}
      </div>

      {cfMutation.isError && (
        <div className="bg-accent-red/10 border border-accent-red/20 rounded-xl p-4 flex items-center justify-between">
          <p className="text-sm text-accent-red">
            Erreur : {(cfMutation.error as Error).message}
          </p>
          <button
            onClick={() => cfMutation.reset()}
            className="text-xs text-accent-cyan hover:underline"
          >
            Fermer
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Config form */}
        <div className="lg:col-span-3">
          <div className="bg-bg-card rounded-xl border border-white/5 p-5">
            <CFConfigForm onSubmit={handleSubmit} isPending={cfMutation.isPending} />
          </div>
        </div>

        {/* Center: Results */}
        <div className="lg:col-span-5">
          <CFResultView
            result={result}
            isLoading={cfMutation.isPending}
          />
        </div>

        {/* Right: Panels */}
        <div className="lg:col-span-4">
          <div className="bg-bg-card rounded-xl border border-white/5 p-5">
            {/* Right tab bar */}
            <div className="flex border-b border-white/10 mb-4">
              {rightTabs.map((t) => (
                <button
                  key={t.key}
                  onClick={() => setRightTab(t.key)}
                  className={`px-3 py-1.5 text-xs transition-colors ${
                    rightTab === t.key
                      ? 'border-b-2 border-accent-cyan text-accent-cyan'
                      : 'text-text-secondary hover:text-text-primary'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {rightTab === 'ips' && <IPSPanel />}

            {rightTab === 'radar' && (
              result?.theta && Object.keys(result.theta).length > 0 ? (
                <RadarPlot theta={result.theta} className="h-[350px]" />
              ) : (
                <p className="text-xs text-text-secondary italic text-center py-8">
                  Les paramètres theta apparaîtront ici après la génération du contrefactuel.
                </p>
              )
            )}

            {rightTab === 'pastas' && <PastasPanel />}
          </div>
        </div>
      </div>
    </div>
  )
}
