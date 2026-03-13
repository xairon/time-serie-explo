import { useClusterProfiling } from '@/hooks/useLatentSpace'
import { AlertTriangle } from 'lucide-react'
import { MetadataDistributions } from './MetadataDistributions'
import { ConcordanceTable } from './ConcordanceTable'
import { TemporalPrototypes } from './TemporalPrototypes'
import { FeatureFingerprints } from './FeatureFingerprints'
import { ShapExplainability } from './ShapExplainability'

interface ClusterProfilingProps {
  domain: 'piezo' | 'hydro'
  hideUnclassified: boolean
}

function SkeletonBlock({ label }: { label: string }) {
  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-6">
      <div className="flex items-center gap-3">
        <div className="w-5 h-5 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin" />
        <span className="text-text-muted text-sm">{label}</span>
      </div>
    </div>
  )
}

export function ClusterProfiling({ domain, hideUnclassified }: ClusterProfilingProps) {
  const { data, isLoading, isError } = useClusterProfiling(domain, hideUnclassified)

  if (isLoading) {
    return (
      <div className="flex flex-col gap-4 overflow-y-auto pr-2">
        <SkeletonBlock label="Loading metadata distributions..." />
        <SkeletonBlock label="Loading concordance metrics..." />
        <SkeletonBlock label="Loading temporal prototypes..." />
        <SkeletonBlock label="Loading feature fingerprints..." />
        <SkeletonBlock label="Loading SHAP analysis..." />
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="bg-bg-card rounded-xl border border-white/5 p-8 flex flex-col items-center gap-4 max-w-md">
          <AlertTriangle className="w-10 h-10 text-accent-red" />
          <p className="text-text-primary text-center">Failed to load profiling data</p>
        </div>
      </div>
    )
  }

  const profiling = data as Record<string, unknown>
  const warnings = (profiling.warnings as string[]) ?? []

  return (
    <div className="flex flex-col gap-4 overflow-y-auto pr-2">
      {warnings.length > 0 && (
        <div className="flex items-center gap-2 bg-amber-500/10 text-amber-400 px-4 py-2 rounded-lg text-sm">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{warnings.join(' | ')}</span>
        </div>
      )}

      <div className="text-text-muted text-xs">
        {profiling.n_stations as number} stations · {profiling.n_clusters as number} clusters
      </div>

      <MetadataDistributions
        distributions={(profiling.distributions as Record<string, unknown>[]) ?? []}
        domain={domain}
      />
      <ConcordanceTable
        concordance={(profiling.concordance as Record<string, unknown>[]) ?? []}
      />
      <TemporalPrototypes
        prototypes={(profiling.prototypes as Record<string, unknown>[]) ?? []}
      />
      <FeatureFingerprints
        fingerprints={(profiling.fingerprints as Record<string, unknown>[]) ?? []}
      />
      <ShapExplainability
        shap={(profiling.shap as Record<string, unknown>) ?? {}}
      />
    </div>
  )
}
