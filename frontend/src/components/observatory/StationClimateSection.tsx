import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { CloudRain } from 'lucide-react'
import { useSPI } from '@/hooks/useObservatory'
import { useClimatPointSeries } from '@/hooks/useClimat'
import { DroughtIndexChart } from './DroughtIndexChart'
import { SpiClassBadge } from './SpiClassBadge'
import { computeRollingCumuls, latestSpiPoint } from '@/lib/climate-cumuls'
import { formatNumber } from '@/lib/observatory-utils'
import { climatDeepLink } from '@/lib/climat-grid'

/** SPI windows supported by the station endpoint (months). */
const SPI_WINDOWS = [1, 3, 6, 12] as const
/** Rolling-cumulative windows shown in the tiles (months). */
const CUMUL_WINDOWS = [3, 6, 12]

interface Props {
  code: string
  type: 'piezo' | 'hydro'
  /** Mapped ERA5 cell coordinates from the station detail payload (null = unmapped station). */
  era5Lat?: number | null
  era5Lon?: number | null
}

/**
 * « Contexte climatique » (Task C2): local SPI of the station's mapped ERA5 cell
 * (window selector 1/3/6/12, WMO class of the latest month as a badge) + rolling
 * precipitation cumuls vs. the 1991-2020 normal, with a link to the Climat page.
 */
export function StationClimateSection({ code, type, era5Lat, era5Lon }: Props) {
  const { t, i18n } = useTranslation()
  const [spiWindow, setSpiWindow] = useState<number>(3)
  const hasCell = era5Lat != null && era5Lon != null

  const { data: spiData, isLoading: spiLoading } = useSPI(code, type, spiWindow)
  const { data: pointSeries } = useClimatPointSeries(era5Lat ?? undefined, era5Lon ?? undefined)

  const latest = useMemo(() => latestSpiPoint(spiData ?? []), [spiData])
  const cumuls = useMemo(() => computeRollingCumuls(pointSeries?.series ?? [], CUMUL_WINDOWS), [pointSeries])
  const hasCumuls = cumuls.some((c) => c != null)

  const localeTag = i18n.language?.startsWith('en') ? 'en-US' : 'fr-FR'
  const latestMonthLabel = latest
    ? new Date(latest.mois).toLocaleDateString(localeTag, { year: 'numeric', month: 'long' })
    : null

  if (!hasCell && !spiLoading && (spiData?.length ?? 0) === 0) {
    return null // Unmapped station with no SPI series: nothing climatic to show.
  }

  return (
    <section className="bg-bg-card border border-white/5 rounded-xl p-5 space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2">
          <CloudRain className="w-4 h-4 text-accent-cyan" />{t('mainPages.station.climate.title')}
        </h2>
        {latest && (
          <span className="flex items-center gap-2 text-xs text-text-secondary">
            {t('mainPages.station.climate.latestMonth')} ({latestMonthLabel}) :
            <span className="font-mono text-text-primary">{latest.spi != null ? latest.spi.toFixed(2) : '--'}</span>
            <SpiClassBadge cls={latest.classification} />
          </span>
        )}
        <div className="flex-1" />
        {hasCell && (
          <Link
            to={climatDeepLink(era5Lat!, era5Lon!)}
            className="text-xs font-semibold text-accent-cyan hover:underline"
          >
            {t('observatory.era5.popupClimatLink')}
          </Link>
        )}
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-text-secondary">{t('mainPages.station.climate.window')}</span>
        <div role="group" aria-label={t('mainPages.station.climate.window')} className="flex gap-1">
          {SPI_WINDOWS.map((w) => (
            <button
              key={w}
              aria-pressed={spiWindow === w}
              onClick={() => setSpiWindow(w)}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${spiWindow === w ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`}
            >
              {t('mainPages.station.climate.windowMonths', { n: w })}
            </button>
          ))}
        </div>
      </div>

      {spiLoading ? (
        <div className="h-64 bg-white/5 rounded animate-pulse" />
      ) : spiData && spiData.length > 0 ? (
        <DroughtIndexChart data={spiData} indexKey="spi" label={t('mainPages.station.spiLabel')} />
      ) : (
        <div className="flex items-center justify-center h-32 text-text-secondary text-sm">
          {t('mainPages.station.climate.noSpiData')}
        </div>
      )}

      {hasCumuls && (
        <div>
          <h3 className="text-xs font-semibold text-text-secondary mb-2">{t('mainPages.station.climate.cumulsTitle')}</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {CUMUL_WINDOWS.map((w, i) => {
              const c = cumuls[i]
              return (
                <div key={w} className="bg-bg-card border border-white/5 rounded-xl p-4">
                  <p className="text-xs text-text-secondary mb-2">{t('mainPages.station.climate.cumulWindow', { n: w })}</p>
                  {c ? (
                    <>
                      <div className="flex items-baseline gap-1.5">
                        <span className="text-xl font-semibold text-text-primary font-mono">{formatNumber(c.cumul, 0)}</span>
                        <span className="text-xs text-text-secondary">mm</span>
                        {c.ecartPct != null && (
                          <span className={`text-xs font-semibold ml-1 ${c.ecartMm < 0 ? 'text-amber-400' : 'text-teal-400'}`}>
                            {c.ecartPct > 0 ? '+' : ''}{formatNumber(c.ecartPct, 0)} %
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-text-secondary mt-1">
                        {t('mainPages.station.climate.normal')} : {formatNumber(c.normale, 0)} mm
                      </p>
                    </>
                  ) : (
                    <span className="text-sm text-text-secondary">{t('mainPages.station.climate.insufficientData')}</span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </section>
  )
}
