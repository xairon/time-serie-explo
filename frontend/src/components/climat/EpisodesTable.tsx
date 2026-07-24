import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowDown, ArrowUp } from 'lucide-react'
import type { ClimatDroughtEpisode } from '@/lib/observatory-types'
import { sortEpisodes } from '@/lib/climat-episodes'
import type { EpisodeSortKey, SortDirection } from '@/lib/climat-episodes'

interface Props {
  episodes: ClimatDroughtEpisode[]
  isLoading: boolean
  /** The episode still ongoing (see findCurrentEpisode), highlighted in the table. */
  currentEpisode?: ClimatDroughtEpisode
  /** Active drought index (SPI/SPEI toggle in PointPanel), drives the min column header. */
  index?: 'spi' | 'spei'
}

function formatMonth(iso: string, locale: string): string {
  const [y, m] = iso.split('-').map(Number)
  if (!y || !m) return iso
  return new Intl.DateTimeFormat(locale, { month: 'short', year: 'numeric' }).format(new Date(y, m - 1, 1))
}

/** Drought episodes table (Task B2) — début/fin/durée/SPI min/déficit, sortable by
 *  duration or start date, with the ongoing episode (if any) highlighted. Data comes
 *  from GET /observatory/climat/point-episodes (useClimatPointEpisodes). */
export function EpisodesTable({ episodes, isLoading, currentEpisode, index = 'spi' }: Props) {
  const { t, i18n } = useTranslation()
  const [sortKey, setSortKey] = useState<EpisodeSortKey>('duree_mois')
  const [direction, setDirection] = useState<SortDirection>('desc')

  const sorted = useMemo(() => sortEpisodes(episodes, sortKey, direction), [episodes, sortKey, direction])

  const toggleSort = (key: EpisodeSortKey) => {
    if (key === sortKey) setDirection((d) => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setDirection('desc') }
  }

  if (isLoading) {
    return (
      <div className="space-y-1.5">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-7 w-full bg-white/5 rounded animate-pulse" />
        ))}
      </div>
    )
  }

  if (episodes.length === 0) {
    return <p className="text-xs text-text-secondary py-3 text-center">{t('climat.episodes.empty')}</p>
  }

  const SortHeader = ({ label, sortKeyValue, align = 'left' }: { label: string; sortKeyValue: EpisodeSortKey; align?: 'left' | 'right' }) => (
    <th scope="col" className={`font-medium text-text-secondary px-2 py-1.5 ${align === 'right' ? 'text-right' : 'text-left'}`}>
      <button
        type="button"
        onClick={() => toggleSort(sortKeyValue)}
        className={`flex items-center gap-1 hover:text-text-primary transition-colors ${align === 'right' ? 'ml-auto' : ''}`}
      >
        {label}
        {sortKey === sortKeyValue && (direction === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />)}
      </button>
    </th>
  )

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-white/10">
            <SortHeader label={t('climat.episodes.start')} sortKeyValue="debut" />
            <th scope="col" className="text-left font-medium text-text-secondary px-2 py-1.5">{t('climat.episodes.end')}</th>
            <SortHeader label={t('climat.episodes.duration')} sortKeyValue="duree_mois" align="right" />
            <th scope="col" className="text-right font-medium text-text-secondary px-2 py-1.5">{t(`climat.episodes.${index}Min`)}</th>
            <th scope="col" className="text-right font-medium text-text-secondary px-2 py-1.5">{t('climat.episodes.deficit')}</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((ep) => {
            const isCurrent = currentEpisode != null && ep.debut === currentEpisode.debut && ep.fin === currentEpisode.fin
            return (
              <tr key={`${ep.debut}-${ep.fin}`} className={`border-b border-white/5 last:border-0 ${isCurrent ? 'bg-amber-500/10' : ''}`}>
                <td className="px-2 py-1.5 text-text-primary">{formatMonth(ep.debut, i18n.language)}</td>
                <td className="px-2 py-1.5 text-text-primary">
                  {formatMonth(ep.fin, i18n.language)}
                  {isCurrent && (
                    <span className="ml-1.5 text-[9px] px-1 py-0.5 rounded-full bg-amber-500/20 text-amber-400 uppercase tracking-wide align-middle">
                      {t('climat.episodes.ongoing')}
                    </span>
                  )}
                </td>
                <td className="px-2 py-1.5 text-right font-mono text-text-primary">{ep.duree_mois}</td>
                <td className="px-2 py-1.5 text-right font-mono text-text-primary">{ep.index_min.toFixed(2)}</td>
                <td className="px-2 py-1.5 text-right font-mono text-text-primary">{Math.round(ep.deficit_cumule_mm)} mm</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
