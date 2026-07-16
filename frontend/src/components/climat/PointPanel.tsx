import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { X, Download } from 'lucide-react'
import { useClimatPointSeries, useClimatPointEpisodes, EPISODES_WINDOW } from '@/hooks/useClimat'
import { observatoryApi } from '@/lib/observatory-api'
import { findCurrentEpisode, findLastEntryWithSpi } from '@/lib/climat-episodes'
import { classifyBilan } from '@/lib/climat-scale'
import { era5SpiClassColor } from '@/lib/era5-colors'
import type { ClimatPointSeriesEntry } from '@/lib/observatory-types'
import { PrecipNormalChart } from './PrecipNormalChart'
import { ClimatIndexChart } from './ClimatIndexChart'
import { EpisodesTable } from './EpisodesTable'
import { CompareYearsSection } from './CompareYearsSection'

interface Props {
  lat: number
  lon: number
  onClose: () => void
}

/** Formateur local du bloc « bilan du mois ». `climatFormatValue` ne convient pas :
 *  il indexe CLIMAT_VARIABLES, dont temperature/precipitation/etp ont été retirés
 *  (ce ne sont plus des couches). Rend le vrai chiffre, ou — s'il manque. */
function fmtValue(v: number | null | undefined, unit: string, digits = 0): string {
  if (v == null || Number.isNaN(v)) return '—'
  return `${v.toFixed(digits)} ${unit}`
}

/** Idem, mais signé avec un vrai U+2212 (cohérent avec climatFormatValue). */
function fmtSigned(v: number | null | undefined, unit: string): string {
  if (v == null || Number.isNaN(v)) return '—'
  const s = Math.abs(Math.round(v)).toString()
  return `${v < 0 ? `−${s}` : `+${s}`} ${unit}`
}

/** Point/Zone side panel (Task B2) — opened when a grid cell is clicked on the
 *  Situation map (or via a deep link carrying ?lat&lon, see useSelectedCellParam).
 *  Composes: précip vs normale (PrecipNormalChart), SPI/STI multi-window
 *  (ClimatIndexChart), drought episodes (EpisodesTable), a direct CSV export link,
 *  and the Comparaison section (CompareYearsSection, Task B3). */
export function PointPanel({ lat, lon, onClose }: Props) {
  const { t } = useTranslation()
  const { data: pointData, isLoading: seriesLoading, isError: seriesError } = useClimatPointSeries(lat, lon)
  // Window selector lives here (not inside ClimatIndexChart) so the episodes table
  // below follows the same window the SPI/STI chart is showing (Task C3).
  const [indexWindow, setIndexWindow] = useState<number>(EPISODES_WINDOW)
  const { data: episodes, isLoading: episodesLoading, isError: episodesError } = useClimatPointEpisodes(lat, lon, indexWindow)

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  const series = pointData?.series ?? []
  // Dernier mois de la série = le plus récent. Le panneau est une fiche de lieu :
  // il ne suit pas le MonthStepper de la carte (pas de prop `month` — YAGNI).
  const lastEntry = series.length > 0 ? series[series.length - 1] : undefined
  const bilan = lastEntry?.bilan_hydrique
  const bilanClass = bilan != null && !Number.isNaN(bilan) ? classifyBilan(bilan) : undefined
  // The series' last entry is usually the partial current month, which has no
  // SPI/STI yet (null) — scan backward for the last entry that actually has
  // spi_<indexWindow>, so the "ongoing" highlight doesn't silently go dead in
  // production (see findLastEntryWithSpi). Also reads the window-specific field
  // so the highlight stays consistent with whichever window the episodes table
  // was fetched at (see findCurrentEpisode).
  const lastSpiEntry = findLastEntryWithSpi(series, indexWindow)
  const lastEntrySpi = lastSpiEntry?.[`spi_${indexWindow}` as keyof ClimatPointSeriesEntry] as number | null | undefined
  const currentEpisode = findCurrentEpisode(episodes ?? [], lastSpiEntry?.month, lastEntrySpi)

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
              {lastEntry && (
                <div>
                  <h3 className="text-sm font-semibold text-text-primary mb-2">
                    {t('climat.pointPanel.balanceTitle')}
                  </h3>
                  <dl className="rounded-lg border border-white/10 divide-y divide-white/5">
                    {[
                      { k: 'climat.variables.temperature', v: fmtValue(lastEntry.temperature_moyenne, '°C', 1) },
                      { k: 'climat.variables.precipitation', v: fmtValue(lastEntry.precipitation_totale, 'mm') },
                      { k: 'climat.variables.etp', v: fmtValue(lastEntry.etp_totale, 'mm') },
                    ].map(({ k, v }) => (
                      <div key={k} className="flex items-center justify-between px-3 py-1.5">
                        <dt className="text-xs text-text-secondary">{t(k)}</dt>
                        <dd className="text-xs font-medium text-text-primary tabular-nums">{v}</dd>
                      </div>
                    ))}
                    <div className="flex items-center justify-between px-3 py-1.5">
                      <dt className="text-xs text-text-secondary">{t('climat.variables.bilanHydrique')}</dt>
                      <dd className="text-xs font-medium text-text-primary tabular-nums flex items-center gap-1.5">
                        {fmtSigned(bilan, 'mm')}
                        {bilanClass && (
                          <span
                            className="text-[10px] px-1.5 py-0.5 rounded"
                            style={{ backgroundColor: `${era5SpiClassColor(bilanClass)}33`, color: era5SpiClassColor(bilanClass) }}
                          >
                            {t(`climat.bilanClasses.${bilanClass}`, { defaultValue: bilanClass })}
                          </span>
                        )}
                      </dd>
                    </div>
                  </dl>
                  <p className="text-[10px] text-text-secondary mt-1">{t('climat.pointPanel.balanceHint')}</p>
                </div>
              )}
              <PrecipNormalChart series={series} />
              <ClimatIndexChart series={series} window={indexWindow} onWindowChange={setIndexWindow} />
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
