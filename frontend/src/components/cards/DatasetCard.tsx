import { Database } from 'lucide-react'
import type { DatasetSummary } from '@/lib/types'

interface DatasetCardProps {
  dataset: DatasetSummary
}

export function DatasetCard({ dataset }: DatasetCardProps) {
  const [dateStart, dateEnd] = dataset.date_range

  return (
    <div className="bg-bg-card rounded-xl p-4 border border-white/5 hover:border-accent-cyan/20 transition-colors">
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-lg bg-accent-cyan/10 flex items-center justify-center shrink-0">
          <Database className="w-4 h-4 text-accent-cyan" />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-text-primary truncate">{dataset.name}</h3>
          <p className="text-xs text-text-secondary mt-0.5">
            {dataset.stations.length} station{dataset.stations.length > 1 ? 's' : ''} &middot;{' '}
            {dataset.n_rows.toLocaleString('fr-FR')} lignes
          </p>
        </div>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-bg-hover text-text-secondary uppercase">
          {dataset.source}
        </span>
      </div>
      <div className="mt-3 pt-3 border-t border-white/5 flex items-center justify-between text-xs text-text-secondary">
        <span>Cible : {dataset.target_variable}</span>
        <span>
          {dateStart?.slice(0, 10)} — {dateEnd?.slice(0, 10)}
        </span>
      </div>
    </div>
  )
}
