import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { X, Download } from 'lucide-react'
import { useClimatPointSeries, useClimatPointEpisodes } from '@/hooks/useClimat'
import { observatoryApi } from '@/lib/observatory-api'
import { findCurrentEpisode } from '@/lib/climat-episodes'
import { PrecipNormalChart } from './PrecipNormalChart'
import { ClimatIndexChart } from './ClimatIndexChart'
import { EpisodesTable } from './EpisodesTable'
import { CompareYearsSection } from './CompareYearsSection'

interface Props {
  lat: number
  lon: number
  onClose: () => void
}

/** Point/Zone side panel (Task B2) — opened when a grid cell is clicked on the
 *  Situation map (or via a deep link carrying ?lat&lon, see useSelectedCellParam).
 *  Composes: précip vs normale (PrecipNormalChart), SPI/STI multi-window
 *  (ClimatIndexChart), drought episodes (EpisodesTable), a direct CSV export link,
 *  and the Comparaison section (CompareYearsSection, Task B3). */
export function PointPanel({ lat, lon, onClose }: Props) {
  const { t } = useTranslation()
  const { data: pointData, isLoading: seriesLoading, isError: seriesError } = useClimatPointSeries(lat, lon)
  const { data: episodes, isLoading: episodesLoading, isError: episodesError } = useClimatPointEpisodes(lat, lon)

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  const series = pointData?.series ?? []
  const lastEntry = series[series.length - 1]
  const currentEpisode = findCurrentEpisode(episodes ?? [], lastEntry?.month, lastEntry?.spi_3)

  return (
    <>
      <div className="md:hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-40" onClick={onClose} />
      <div
        role="dialog"
        aria-label={t('climat.pointPanel.ariaLabel', { lat: lat.toFixed(2), lon: lon.toFixed(2) })}
        className="absolute top-0 right-0 h-full z-40 w-full sm:w-[560px] lg:w-[680px] max-w-full bg-bg-card border-l border-white/10 shadow-2xl transition-transform duration-200 ease-out overflow-y-auto"
      >
        <div className="flex items-start justify-between px-4 py-3 border-b border-white/10 sticky top-0 bg-bg-card z-10">
          <div>
            <h2 className="text-sm font-semibold text-text-primary">{t('climat.pointPanel.title')}</h2>
            <p className="text-xs text-text-secondary mt-0.5">{t('climat.pointPanel.coords', { lat: lat.toFixed(2), lon: lon.toFixed(2) })}</p>
          </div>
          <button onClick={onClose} aria-label={t('observatory.drawer.close')} className="p-1 hover:bg-bg-hover rounded ml-2 flex-shrink-0">
            <X className="w-4 h-4 text-text-secondary" />
          </button>
        </div>

        <div className="p-4 space-y-5">
          {/* Direct link — the browser streams/downloads the CSV, no client-side fetch-into-memory. */}
          <a
            href={observatoryApi.climat.exportPointUrl(lat, lon)}
            download
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-accent-cyan/10 text-accent-cyan hover:bg-accent-cyan/20 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            {t('climat.pointPanel.exportCsv')}
          </a>

          {seriesError && <p className="text-sm text-red-400">{t('climat.pointPanel.loadFailed')}</p>}

          {seriesLoading && !seriesError && (
            <div className="space-y-3">
              <div className="h-48 w-full bg-white/5 rounded-lg animate-pulse" />
              <div className="h-40 w-full bg-white/5 rounded-lg animate-pulse" />
            </div>
          )}

          {!seriesLoading && !seriesError && (
            <>
              <PrecipNormalChart series={series} />
              <ClimatIndexChart series={series} />
              <div>
                <h3 className="text-sm font-semibold text-text-primary mb-2">{t('climat.episodes.title')}</h3>
                <EpisodesTable episodes={episodes ?? []} isLoading={episodesLoading && !episodesError} currentEpisode={currentEpisode} />
                {episodesError && <p className="text-xs text-red-400 mt-1">{t('climat.pointPanel.loadFailed')}</p>}
              </div>
            </>
          )}

          {/* Comparaison (Task B3) — independent of the history/episodes load state
              above, it targets the same cell via its own compare-years fetch. */}
          <CompareYearsSection lat={lat} lon={lon} />
        </div>
      </div>
    </>
  )
}
