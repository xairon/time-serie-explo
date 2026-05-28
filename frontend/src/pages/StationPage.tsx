import { useState, useMemo, useCallback } from 'react'
import { useParams, useLocation, Link } from 'react-router-dom'
import { ArrowLeft, Info, Waves, Brain } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { usePiezoStationDetail, useHydroStationDetail, useHydroSiblings, usePiezoMonthly, useHydroMonthly, usePiezoDaily, useHydroDaily, usePiezoYearly, useHydroYearly, usePiezoSPLI, useHydroSSFI, useSPI } from '@/hooks/useObservatory'
import { StationKPICards } from '@/components/observatory/StationKPICards'
import { TimeseriesChart } from '@/components/observatory/TimeseriesChart'
import { DroughtIndexChart } from '@/components/observatory/DroughtIndexChart'
import { PastasSection } from '@/components/observatory/PastasSection'
import { AddToCompareButton } from '@/components/observatory/AddToCompareButton'
import { CLASSIFICATION_COLORS } from '@/lib/observatory-constants'

type Resolution = 'daily' | 'monthly' | 'yearly'

function formatDateEN(d: string | null | undefined): string { if (!d) return '--'; return new Date(d).toLocaleDateString('fr-FR', { year: 'numeric', month: 'short' }) }
function formatDuration(months: number | null | undefined, monthSuffix: string, yearsSuffix: string): string { if (!months) return '--'; const years = Math.floor(months / 12); const rem = months % 12; if (years === 0) return `${rem} ${monthSuffix}`; if (rem === 0) return `${years} ${yearsSuffix}`; return `${years} ${yearsSuffix} ${rem} ${monthSuffix}` }
function formatPeriod(start: string | null | undefined, end: string | null | undefined, untilWord: string, sinceWord: string): string { if (!start && !end) return '--'; const fmt = (d: string) => new Date(d).toLocaleDateString('fr-FR', { year: 'numeric', month: 'short' }); if (start && end) return `${fmt(start)} -- ${fmt(end)}`; if (end) return `${untilWord} ${fmt(end)}`; return `${sinceWord} ${fmt(start!)}` }

function MetaRow({ label, value, mono = false }: { label: string; value: React.ReactNode; mono?: boolean }) {
  if (value == null || value === '' || value === '--') return null
  return (<div className="flex items-start justify-between gap-2 py-1.5 border-b border-white/5 last:border-0"><span className="text-xs text-gray-500 shrink-0">{label}</span><span className={`text-xs text-gray-200 text-right ${mono ? 'font-mono' : ''}`}>{value}</span></div>)
}

function SkeletonKPI() { return (<div className="grid grid-cols-2 md:grid-cols-4 gap-3">{Array.from({ length: 4 }).map((_, i) => (<div key={i} className="bg-bg-card border border-white/5 rounded-xl p-4 animate-pulse"><div className="h-3 bg-white/10 rounded w-1/2 mb-3" /><div className="h-6 bg-white/5 rounded w-3/4 mb-2" /><div className="h-3 bg-white/5 rounded w-1/3" /></div>))}</div>) }
function SkeletonChart() { return (<div className="bg-bg-card border border-white/5 rounded-xl p-5 animate-pulse"><div className="h-4 bg-white/10 rounded w-1/3 mb-4" /><div className="h-64 bg-white/5 rounded" /></div>) }

export default function StationPage() {
  const { t } = useTranslation()
  const RESOLUTION_OPTIONS: { value: Resolution; label: string }[] = [
    { value: 'daily', label: t('mainPages.station.daily') },
    { value: 'monthly', label: t('mainPages.station.monthly') },
    { value: 'yearly', label: t('mainPages.station.yearly') },
  ]
  const monthSuffix = t('mainPages.station.monthSuffix')
  const yearsSuffix = t('mainPages.station.yearsSuffix')
  const untilWord = t('mainPages.station.until')
  const sinceWord = t('mainPages.station.since')
  const params = useParams(); const location = useLocation()
  const isPiezo = location.pathname.includes('/piezo/')
  const code = (params['*'] || '').replace(/^(piezo|hydro)\//, '')
  const [resolution, setResolution] = useState<Resolution>('daily')

  const defaultEnd = useMemo(() => new Date().toISOString().slice(0, 10), [])
  const defaultStart = useMemo(() => { const d = new Date(); d.setFullYear(d.getFullYear() - 5); return d.toISOString().slice(0, 10) }, [])
  const [dailyStart, setDailyStart] = useState<string | undefined>(defaultStart)
  const [dailyEnd, setDailyEnd] = useState(defaultEnd)
  const dailyLimit = dailyStart === undefined ? 36500 : undefined

  const handleDailyPeriodChange = useCallback((months: number) => {
    if (months === Infinity) {
      setDailyStart(undefined)
    } else {
      const d = new Date()
      d.setMonth(d.getMonth() - months)
      setDailyStart(d.toISOString().slice(0, 10))
    }
    setDailyEnd(new Date().toISOString().slice(0, 10))
  }, [])

  const { data: piezoStation, isLoading: piezoLoading, isError: piezoError } = usePiezoStationDetail(isPiezo ? code : '')
  const { data: hydroStation, isLoading: hydroLoading, isError: hydroError } = useHydroStationDetail(!isPiezo ? code : '')
  const { data: piezoMonthly, isLoading: piezoMonthlyLoading } = usePiezoMonthly(isPiezo ? code : '', { enabled: resolution === 'monthly' })
  const { data: hydroMonthly, isLoading: hydroMonthlyLoading } = useHydroMonthly(!isPiezo ? code : '', { enabled: resolution === 'monthly' })
  const { data: piezoDaily, isLoading: piezoDailyLoading } = usePiezoDaily(isPiezo && resolution === 'daily' ? code : '', dailyStart, dailyEnd, dailyLimit)
  const { data: hydroDaily, isLoading: hydroDailyLoading } = useHydroDaily(!isPiezo && resolution === 'daily' ? code : '', dailyStart, dailyEnd, dailyLimit)
  const { data: piezoYearly, isLoading: piezoYearlyLoading } = usePiezoYearly(isPiezo && resolution === 'yearly' ? code : '')
  const { data: hydroYearly, isLoading: hydroYearlyLoading } = useHydroYearly(!isPiezo && resolution === 'yearly' ? code : '')

  const station: any = isPiezo ? piezoStation : hydroStation
  const monthly = isPiezo ? piezoMonthly : hydroMonthly
  const stationLoading = isPiezo ? piezoLoading : hydroLoading
  const stationError = isPiezo ? piezoError : hydroError
  const type = isPiezo ? 'piezo' as const : 'hydro' as const

  const { data: hydroSiblings } = useHydroSiblings(!isPiezo ? code : '')
  const { data: spliData } = usePiezoSPLI(isPiezo ? code : '')
  const { data: ssfiData } = useHydroSSFI(!isPiezo ? code : '')
  const { data: spiData } = useSPI(code, type)
  const droughtData = isPiezo ? spliData : ssfiData

  const activeData = useMemo(() => { if (resolution === 'daily') return isPiezo ? piezoDaily : hydroDaily; if (resolution === 'yearly') return isPiezo ? piezoYearly : hydroYearly; return monthly }, [resolution, isPiezo, piezoDaily, hydroDaily, piezoYearly, hydroYearly, monthly])
  const activeLoading = resolution === 'daily' ? (isPiezo ? piezoDailyLoading : hydroDailyLoading) : resolution === 'yearly' ? (isPiezo ? piezoYearlyLoading : hydroYearlyLoading) : (isPiezo ? piezoMonthlyLoading : hydroMonthlyLoading)

  if (stationLoading) return (<div className="h-full overflow-y-auto"><div className="max-w-7xl mx-auto px-6 py-6 space-y-6"><div className="flex items-center gap-4 animate-pulse"><div className="w-9 h-9 bg-white/10 rounded-lg" /><div><div className="h-3 bg-white/10 rounded w-20 mb-2" /><div className="h-5 bg-white/10 rounded w-48 mb-1" /><div className="h-3 bg-white/5 rounded w-32" /></div></div><SkeletonKPI /><SkeletonChart /></div></div>)
  if (stationError) return (<div className="flex flex-col items-center justify-center h-full gap-4"><p className="text-red-400">{t('mainPages.station.loadFailed')}</p><button onClick={() => window.location.reload()} className="px-3 py-1.5 text-xs rounded-lg bg-accent-cyan/10 text-accent-cyan hover:bg-accent-cyan/20">{t('mainPages.station.retry')}</button><Link to="/" className="text-accent-cyan hover:underline text-sm">{t('mainPages.station.backToObservatory')}</Link></div>)
  if (!station) return (<div className="flex flex-col items-center justify-center h-full gap-4"><p className="text-text-secondary">{t('mainPages.station.notFound')}</p><Link to="/" className="text-accent-cyan hover:underline text-sm">{t('mainPages.station.backToObservatory')}</Link></div>)

  const name = isPiezo ? (station.nom_commune || station.code_bss) : (station.libelle_station || station.code_station)
  const hydroLabel = !isPiezo && station?.grandeur_hydro_principale === 'H' ? t('mainPages.station.meanHeight') : t('mainPages.station.meanFlow')
  const hydroUnit = !isPiezo && station?.grandeur_hydro_principale === 'H' ? 'm' : 'm\u00b3/s'
  const valueKey = resolution === 'daily' ? (isPiezo ? 'niveau_nappe_eau' : 'resultat_obs_elab') : resolution === 'yearly' ? (isPiezo ? 'niveau_moyen_annuel' : 'resultat_moyen_annuel') : (isPiezo ? 'niveau_moyen' : 'resultat_moyen')
  const valueLabel = isPiezo ? t('mainPages.station.waterLevel') : `${hydroLabel} (${hydroUnit})`
  const unit = isPiezo ? 'm NGF' : hydroUnit

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        <div className="flex items-center gap-4">
          <Link to="/" className="p-2 hover:bg-bg-hover rounded-lg transition-colors" aria-label={t('mainPages.station.backToObservatory')}><ArrowLeft className="w-5 h-5 text-text-secondary" /></Link>
          <div className="flex-1">
            <p className="text-xs text-accent-cyan font-medium uppercase tracking-wide">{isPiezo ? t('mainPages.station.piezometry') : t('mainPages.station.hydrometry')}</p>
            <h1 className="text-xl font-bold text-text-primary">{name}</h1>
            <p className="text-sm text-text-secondary">{station.nom_departement ?? ''} - {code}{!isPiezo && station.nom_cours_eau && ` - ${station.nom_cours_eau}`}</p>
          </div>
          <div className="flex items-center gap-2">
            <AddToCompareButton code={code} type={type} />
            {isPiezo && <Link to={`/pastas/station?station=${encodeURIComponent(code)}`} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-accent-cyan/10 text-accent-cyan hover:bg-accent-cyan/20 transition-colors"><Waves className="w-3.5 h-3.5" />{t('mainPages.station.analyzeInPastas')}</Link>}
            {isPiezo && <Link to={`/ai/data?station=${encodeURIComponent(code)}`} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 transition-colors"><Brain className="w-3.5 h-3.5" />{t('mainPages.station.trainAIModel')}</Link>}
          </div>
        </div>

        <section className="bg-gray-900/50 rounded-xl border border-white/5 p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2"><Info className="w-4 h-4" />{t('mainPages.station.technicalSheet')}</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6">
            {isPiezo ? (<><div><MetaRow label={t('mainPages.station.dataPeriod')} value={formatPeriod(station.premiere_mesure, station.derniere_mesure, untilWord, sinceWord)} /><MetaRow label={t('mainPages.station.duration')} value={formatDuration(station.nb_mois_total, monthSuffix, yearsSuffix)} /><MetaRow label={t('mainPages.station.measures')} value={station.nb_mesures_total?.toLocaleString() ?? null} /><MetaRow label={t('mainPages.station.lastMeasure')} value={formatDateEN(station.derniere_mesure)} /><MetaRow label={t('mainPages.station.stationAltitude')} value={station.altitude_station != null ? `${station.altitude_station.toFixed(0)} m NGF` : null} /></div><div><MetaRow label={t('mainPages.station.bssCode')} value={station.code_bss ?? null} mono /><MetaRow label={t('mainPages.station.coordinates')} value={station.latitude != null && station.longitude != null ? `${station.latitude.toFixed(4)} N, ${station.longitude.toFixed(4)} E` : null} /><MetaRow label={t('mainPages.station.bdlisaCode')} value={station.codes_bdlisa ? (<a href={`https://bdlisa.eaufrance.fr/hydrogeounit/${station.codes_bdlisa.split(',')[0]}`} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">{station.codes_bdlisa}</a>) : null} /><MetaRow label={t('mainPages.station.historicalMin')} value={station.niveau_min_absolu != null ? `${station.niveau_min_absolu.toFixed(2)} m NGF` : null} /><MetaRow label={t('mainPages.station.historicalMax')} value={station.niveau_max_absolu != null ? `${station.niveau_max_absolu.toFixed(2)} m NGF` : null} /></div><div className="col-span-full flex flex-wrap gap-3 pt-2 border-t border-white/5"><a href={`https://ades.eaufrance.fr/Fiche/PtEau?code=${station.code_bss}`} target="_blank" rel="noopener noreferrer" className="text-[11px] text-blue-400 hover:underline">ADES</a><a href="https://hubeau.eaufrance.fr/page/api-piezometrie" target="_blank" rel="noopener noreferrer" className="text-[11px] text-blue-400 hover:underline">Hub'Eau</a></div></>) : (<><div><MetaRow label={t('mainPages.station.dataPeriod')} value={formatPeriod(station.premiere_mesure, station.derniere_mesure, untilWord, sinceWord)} /><MetaRow label={t('mainPages.station.duration')} value={formatDuration(station.nb_mois_total, monthSuffix, yearsSuffix)} /><MetaRow label={t('mainPages.station.lastMeasure')} value={formatDateEN(station.derniere_mesure)} /></div><div><MetaRow label={t('mainPages.station.stationCode')} value={station.code_station ?? null} mono /><MetaRow label={t('mainPages.station.coordinates')} value={station.latitude_station != null && station.longitude_station != null ? `${station.latitude_station.toFixed(4)} N, ${station.longitude_station.toFixed(4)} E` : null} /><MetaRow label={t('mainPages.station.riverCode')} value={station.code_cours_eau ? (<a href={`https://services.sandre.eaufrance.fr/Courdo/Fiche/client/fiche_courdo.php?CdSandre=${station.code_cours_eau}`} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline font-mono">{station.code_cours_eau}</a>) : null} /><MetaRow label={t('mainPages.station.historicalMin')} value={station.resultat_min_global != null ? `${station.resultat_min_global.toFixed(2)} ${hydroUnit}` : null} /><MetaRow label={t('mainPages.station.historicalMax')} value={station.resultat_max_global != null ? `${station.resultat_max_global.toFixed(2)} ${hydroUnit}` : null} /></div><div className="col-span-full flex flex-wrap gap-3 pt-2 border-t border-white/5"><a href={`https://www.vigicrues.gouv.fr/niv3-station.php?CdStationHydro=${station.code_station}`} target="_blank" rel="noopener noreferrer" className="text-[11px] text-blue-400 hover:underline">VigiCrues</a><a href="https://hubeau.eaufrance.fr/page/api-hydrometrie" target="_blank" rel="noopener noreferrer" className="text-[11px] text-blue-400 hover:underline">Hub'Eau</a></div></>)}
          </div>
        </section>

        {!isPiezo && hydroSiblings && hydroSiblings.siblings.length > 0 && (
          <section className="bg-gray-900/50 rounded-xl border border-white/5 p-4">
            <h2 className="text-sm font-semibold text-gray-300 mb-1 flex items-center gap-2"><Waves className="w-4 h-4" />{t('mainPages.station.hydroSite')} - {hydroSiblings.nb_stations} {t('observatory.stations').toLowerCase()}</h2>
            <p className="text-xs text-gray-500 mb-3">{hydroSiblings.libelle_site || hydroSiblings.code_site}{hydroSiblings.nom_cours_eau ? ` - ${hydroSiblings.nom_cours_eau}` : ''}</p>
            <div className="space-y-1">{hydroSiblings.siblings.map(sib => (<Link key={sib.code_station} to={`/station/hydro/${sib.code_station}`} className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-bg-hover transition-colors"><div><span className="text-xs text-gray-200">{sib.libelle_station || sib.code_station}</span><span className="text-[10px] text-gray-500 ml-2">{sib.code_station}</span></div><div className="flex items-center gap-2">{sib.classification && <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CLASSIFICATION_COLORS[sib.classification] ?? '#6b7280' }} title={sib.classification} />}</div></Link>))}</div>
          </section>
        )}

        <StationKPICards station={station} type={type} />

        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs text-text-secondary font-medium">{t('mainPages.station.resolution')}</span>
          <div role="group" aria-label={t('mainPages.station.temporalResolution')} className="flex gap-1">{RESOLUTION_OPTIONS.filter(opt => { if (opt.value === 'monthly') { const loading = isPiezo ? piezoMonthlyLoading : hydroMonthlyLoading; if (loading) return true; return (monthly?.length ?? 0) > 0 }; return true }).map(opt => (<button key={opt.value} aria-pressed={resolution === opt.value} onClick={() => setResolution(opt.value)} className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${resolution === opt.value ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`}>{opt.label}</button>))}</div>
          {resolution === 'daily' && (<div className="flex items-center gap-2 ml-2"><input type="date" aria-label={t('mainPages.station.startDate')} value={dailyStart ?? ''} onChange={(e) => setDailyStart(e.target.value || undefined)} className="bg-bg-card border border-white/10 rounded-lg px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-cyan/50" /><span className="text-xs text-text-secondary">-</span><input type="date" aria-label={t('mainPages.station.endDate')} value={dailyEnd} onChange={(e) => setDailyEnd(e.target.value)} className="bg-bg-card border border-white/10 rounded-lg px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-cyan/50" /></div>)}
        </div>

        {activeLoading ? <SkeletonChart /> : activeData && activeData.length > 0 ? (<div className="bg-bg-card border border-white/5 rounded-xl p-5"><TimeseriesChart data={activeData} valueKey={valueKey} valueLabel={valueLabel} unit={unit} precipKey={resolution === 'yearly' ? 'precipitation_totale_annuelle' : 'precipitation_totale'} percentiles={undefined} resolution={resolution} defaultPeriod={resolution === 'daily' ? 60 : Infinity} onPeriodChange={resolution === 'daily' ? handleDailyPeriodChange : undefined} /></div>) : (<div className="bg-bg-card border border-white/5 rounded-xl p-5 flex items-center justify-center h-64 text-text-secondary text-sm">{t('mainPages.station.noDataForResolution')}</div>)}

        {(droughtData && droughtData.length > 0) || (spiData && spiData.length > 0) ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {droughtData && droughtData.length > 0 && (<div className="bg-bg-card border border-white/5 rounded-xl p-5"><DroughtIndexChart data={droughtData} indexKey={isPiezo ? 'spli' : 'ssfi'} label={isPiezo ? t('mainPages.station.spliLabel') : t('mainPages.station.ssfiLabel')} /></div>)}
            {spiData && spiData.length > 0 && (<div className="bg-bg-card border border-white/5 rounded-xl p-5"><DroughtIndexChart data={spiData} indexKey="spi" label={t('mainPages.station.spiLabel')} /></div>)}
          </div>
        ) : null}

        {isPiezo && <PastasSection codeBss={code} />}
      </div>
    </div>
  )
}
