import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'

interface PipelineMetrics {
  n_points?: number
  umap_prereduction?: {
    input_dim?: number
    output_dim?: number
    n_neighbors?: number
    min_dist?: number
    trustworthiness?: number
  }
  clustering?: {
    method?: string
    n_clusters?: number
    n_noise?: number
    noise_ratio?: number
    silhouette?: number
    davies_bouldin?: number
    calinski_harabasz?: number
    inertia?: number
  }
  umap_visualization?: {
    input_dim?: number
    output_dim?: number
    n_neighbors?: number
    min_dist?: number
    trustworthiness?: number
  }
}

interface QualityMetricsProps {
  metrics: PipelineMetrics | null
}

function QualityBadge({ value, good, warn }: { value: React.ReactNode; good: boolean; warn: boolean }) {
  const color = good ? 'text-green-400' : warn ? 'text-amber-400' : 'text-red-400'
  return <span className={`font-mono ${color}`}>{value}</span>
}

function MetricRow({ label, value, tooltip }: { label: string; value: React.ReactNode; tooltip?: string }) {
  return (
    <div className="flex items-center justify-between gap-3 text-xs" title={tooltip}>
      <span className="text-text-muted">{label}</span>
      <span className="text-text-primary font-mono">{value}</span>
    </div>
  )
}

function StepSection({ title, children, defaultOpen = true }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs font-medium text-text-secondary hover:text-text-primary transition-colors w-full"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        {title}
      </button>
      {open && <div className="flex flex-col gap-1 mt-1 ml-4">{children}</div>}
    </div>
  )
}

export function QualityMetrics({ metrics }: QualityMetricsProps) {
  if (!metrics) return null

  const pre = metrics.umap_prereduction
  const clust = metrics.clustering
  const viz = metrics.umap_visualization

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-3 flex flex-col gap-2 w-full shrink-0 overflow-y-auto text-xs">
      <span className="text-text-primary text-xs font-medium">Pipeline quality</span>

      {metrics.n_points != null && (
        <MetricRow label="Points" value={metrics.n_points.toLocaleString()} />
      )}

      {/* Step 1: UMAP pre-reduction (only for HDBSCAN) */}
      {pre && (
        <StepSection title={`1. UMAP pre-reduction (${pre.input_dim}d → ${pre.output_dim}d)`}>
          <MetricRow label="n_neighbors" value={pre.n_neighbors} />
          <MetricRow label="min_dist" value={pre.min_dist} />
          {pre.trustworthiness != null && (
            <MetricRow
              label="Trustworthiness"
              tooltip="How well local neighborhoods are preserved (0-1, higher=better)"
              value={
                <QualityBadge
                  value={pre.trustworthiness}
                  good={pre.trustworthiness > 0.9}
                  warn={pre.trustworthiness > 0.8}
                />
              }
            />
          )}
        </StepSection>
      )}

      {/* Step 2: Clustering */}
      {clust && (
        <StepSection title={`2. ${clust.method?.toUpperCase() ?? 'Clustering'} (${clust.n_clusters} clusters)`}>
          {clust.noise_ratio != null && (
            <MetricRow
              label="Noise"
              tooltip="Points classified as noise"
              value={
                <QualityBadge
                  value={Number((clust.noise_ratio * 100).toFixed(1))}
                  good={clust.noise_ratio < 0.1}
                  warn={clust.noise_ratio < 0.3}
                />
              }
            />
          )}
          {clust.silhouette != null && (
            <MetricRow
              label="Silhouette"
              tooltip="Cluster cohesion & separation (-1 to 1, higher=better)"
              value={
                <QualityBadge
                  value={clust.silhouette}
                  good={clust.silhouette > 0.3}
                  warn={clust.silhouette > 0.1}
                />
              }
            />
          )}
          {clust.davies_bouldin != null && (
            <MetricRow
              label="Davies-Bouldin"
              tooltip="Cluster separation (lower=better)"
              value={
                <QualityBadge
                  value={clust.davies_bouldin}
                  good={clust.davies_bouldin < 1.5}
                  warn={clust.davies_bouldin < 2.5}
                />
              }
            />
          )}
          {clust.calinski_harabasz != null && (
            <MetricRow
              label="Calinski-Harabasz"
              tooltip="Variance ratio criterion (higher=better). Scale depends on dataset size — compare across runs, not absolute."
              value={
                <QualityBadge
                  value={Math.round(clust.calinski_harabasz).toLocaleString()}
                  good={clust.calinski_harabasz > 1000}
                  warn={clust.calinski_harabasz > 100}
                />
              }
            />
          )}
          {clust.inertia != null && (
            <MetricRow
              label="Inertia"
              tooltip="Within-cluster sum of squares (lower=better for same k)"
              value={clust.inertia.toFixed(0)}
            />
          )}
        </StepSection>
      )}

      {/* Step 3: UMAP visualization */}
      {viz && (
        <StepSection title={`3. UMAP visualization (${viz.input_dim}d → ${viz.output_dim}d)`}>
          <MetricRow label="n_neighbors" value={viz.n_neighbors} />
          <MetricRow label="min_dist" value={viz.min_dist} />
          {viz.trustworthiness != null && (
            <MetricRow
              label="Trustworthiness"
              tooltip="How well local neighborhoods are preserved (0-1, higher=better)"
              value={
                <QualityBadge
                  value={viz.trustworthiness}
                  good={viz.trustworthiness > 0.9}
                  warn={viz.trustworthiness > 0.8}
                />
              }
            />
          )}
        </StepSection>
      )}
    </div>
  )
}
