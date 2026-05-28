import { Database, Trash2 } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { DatasetSummary } from '@/lib/types'

interface DatasetCardProps {
  dataset: DatasetSummary
  onDelete?: (id: string) => void
  isDeleting?: boolean
}

export function DatasetCard({ dataset, onDelete, isDeleting }: DatasetCardProps) {
  const { t } = useTranslation()
  const [dateStart, dateEnd] = dataset.date_range
  const stationsLabel = dataset.stations.length > 1
    ? t('sharedComponents.cards.stationsPlural')
    : t('sharedComponents.cards.stations')

  return (
    <div className="bg-bg-card rounded-xl p-4 border border-white/5 hover:border-accent-cyan/20 transition-colors">
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-lg bg-accent-cyan/10 flex items-center justify-center shrink-0">
          <Database className="w-4 h-4 text-accent-cyan" />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-text-primary truncate">{dataset.name}</h3>
          <p className="text-xs text-text-secondary mt-0.5">
            {dataset.stations.length} {stationsLabel} &middot;{' '}
            {dataset.n_rows.toLocaleString('en-US')} {t('sharedComponents.cards.rows')}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-bg-hover text-text-secondary uppercase">
            {dataset.source_file.startsWith('db://') ? 'DB' : 'CSV'}
          </span>
          {onDelete && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                onDelete(dataset.id)
              }}
              disabled={isDeleting}
              className="p-1 rounded hover:bg-accent-red/10 text-text-secondary hover:text-accent-red transition-colors disabled:opacity-50"
              title={t('sharedComponents.cards.deleteTitle')}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
      <div className="mt-3 pt-3 border-t border-white/5 flex items-center justify-between text-xs text-text-secondary">
        <span>{t('sharedComponents.cards.target')} : {dataset.target_variable}</span>
        <span>
          {dateStart?.slice(0, 10)} — {dateEnd?.slice(0, 10)}
        </span>
      </div>
    </div>
  )
}
