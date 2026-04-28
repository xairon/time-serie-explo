import { useState, useMemo, useCallback } from 'react'
import { useParams, useLocation, Link } from 'react-router-dom'
import { ArrowLeft, Info, Droplets, Waves, Brain } from 'lucide-react'
import { usePiezoStationDetail, useHydroStationDetail, usePiezoSiblings, useHydroSiblings, usePiezoMonthly, useHydroMonthly, usePiezoDaily, useHydroDaily, usePiezoYearly, useHydroYearly, usePiezoSPLI, useHydroSSFI, useSPI } from '@/hooks/useObservatory'
import { StationKPICards } from '@/components/observatory/StationKPICards'
import { TimeseriesChart } from '@/components/observatory/TimeseriesChart'
import { DroughtIndexChart } from '@/components/observatory/DroughtIndexChart'
import { PastasSection } from '@/components/observatory/PastasSection'
import { CLASSIFICATION_COLORS } from '@/lib/observatory-constants'
import { formatDate } from '@/lib/observatory-utils'

type Resolution = 'daily' | 'monthly' | 'yearly'
const RESOLUTION_OPTIONS: { value: Resolution; label: string }[] = [{ value: 'daily', label: 'Daily' }, { value: 'monthly', label: 'Monthly' }, { value: 'yearly', label: 'Yearly' }]

function formatDateEN(d: string | null | undefined): string { if (!d) return '--'; return new Date(d).toLocaleDateString('en-GB', { year: 'numeric', month: 'short' }) }
function formatDuration(months: number | null | undefined): string { if (!months) return '--'; const years = Math.floor(months / 12); const rem = months % 12; if (years === 0) return `${rem} months`; if (rem === 0) return `${years} years`; return `${years} years ${rem} months` }
function formatPeriod(start: string | null | undefined, end: string | null | undefined): string { if (!start && !end) return '--'; const fmt = (d: string) => new Date(d).toLocaleDateString('en-GB', { year: 'numeric', month: 'short' }); if (start && end) return `${fmt(start)} -- ${fmt(end)}`; if (end) return `until ${fmt(end)}`; return `since ${fmt(start!)}` }

function MetaRow({ label, value, mono = false }: { label: string; value: React.ReactNode; mono?: boolean }) {
  if (value == null || value === '' || value === '--') return null
  return (<div className="flex items-start justify-between gap-2 py-1.5 border-b border-white/5 last:border-0"><span className="text-xs text-gray-500 shrink-0">{label}</span><span className={`text-xs text-gray-200 text-right ${mono ? 'font-mono' : ''}`}>{value}</span></div>)
}

function SkeletonKPI() { return (<div className="grid grid-cols-2 md:grid-cols-4 gap-3">{Array.from({ length: 4 }).map((_, i) => (<div key={i} className="bg-bg-card border border-white/5 rounded-xl p-4 animate-pulse"><div className="h-3 bg-white/10 rounded w-1/2 mb-3" /><div className="h-6 bg-white/5 rounded w-3/4 mb-2" /><div className="h-3 bg-white/5 rounded w-1/3" /></div>))}</div>) }
function SkeletonChart() { return (<div className="bg-bg-card border border-white/5 rounded-xl p-5 animate-pulse"><div className="h-4 bg-white/10 rounded w-1/3 mb-4" /><div className="h-64 bg-white/5 rounded" /></div>) }

export default function StationPage() {
  const params = useParams(); const location = useLocation()
  const isPiezo = location.pathname.includes('/piezo/')
  const code = (params['*'] || '').replace(/^(piezo|hydro)\//, '')
  const [resolution, setResolution] = useState<Resolution>('daily')

  const defaultEnd = useMemo(() => new Date().toISOString().slice(0, 10), [])
  const [dailyStart, setDailyStart] = useState<string | undefined>(undefined)
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

  const { data: piezoStation, isLoading: piezoLoading } = usePiezoStationDetail(isPiezo ? code : '')
  const { data: hydroStation, isLoading: hydroLoading } = useHydroStationDetail(!isPiezo ? code : '')
  const { data: piezoMonthly, isLoading: piezoMonthlyLoading } = usePiezoMonthly(isPiezo ? code : '', { enabled: resolution === 'monthly' })
  const { data: hydroMonthly, isLoading: hydroMonthlyLoading } = useHydroMonthly(!isPiezo ? code : '', { enabled: resolution === 'monthly' })
  const { data: piezoDaily, isLoading: piezoDailyLoading } = usePiezoDaily(isPiezo && resolution === 'daily' ? code : '', dailyStart, dailyEnd, dailyLimit)
  const { data: hydroDaily, isLoading: hydroDailyLoading } = useHydroDaily(!isPiezo && resolution === 'daily' ? code : '', dailyStart, dailyEnd, dailyLimit)
  const { data: piezoYearly, isLoading: piezoYearlyLoading } = usePiezoYearly(isPiezo && resolution === 'yearly' ? code : '')
  const { data: hydroYearly, isLoading: hydroYearlyLoading } = useHydroYearly(!isPiezo && resolution === 'yearly' ? code : '')

  const station: any = isPiezo ? piezoStation : hydroStation
  const monthly = isPiezo ? piezoMonthly : hydroMonthly
  const stationLoading = isPiezo ? piezoLoading : hydroLoading
  const type = isPiezo ? 'piezo' as const : 'hydro' as const

  const { data: piezoSiblings } = usePiezoSiblings(isPiezo ? code : '')
  const { data: hydroSiblings } = useHydroSiblings(!isPiezo ? code : '')
  const { data: spliData } = usePiezoSPLI(isPiezo ? code : '')
  const { data: ssfiData } = useHydroSSFI(!isPiezo ? code : '')
  const { data: spiData } = useSPI(code, type)
  const droughtData = isPiezo ? spliData : ssfiData

  const activeData = useMemo(() => { if (resolution === 'daily') return isPiezo ? piezoDaily : hydroDaily; if (resolution === 'yearly') return isPiezo ? piezoYearly : hydroYearly; return monthly }, [resolution, isPiezo, piezoDaily, hydroDaily, piezoYearly, hydroYearly, monthly])
  const activeLoading = resolution === 'daily' ? (isPiezo ? piezoDailyLoading : hydroDailyLoading) : resolution === 'yearly' ? (isPiezo ? piezoYearlyLoading : hydroYearlyLoading) : (isPiezo ? piezoMonthlyLoading : hydroMonthlyLoading)

  if (stationLoading) return (<div className="h-full overflow-y-auto"><div className="max-w-7xl mx-auto px-6 py-6 space-y-6"><div className="flex items-center gap-4 animate-pulse"><div className="w-9 h-9 bg-white/10 rounded-lg" /><div><div className="h-3 bg-white/10 rounded w-20 mb-2" /><div className="h-5 bg-white/10 rounded w-48 mb-1" /><div className="h-3 bg-white/5 rounded w-32" /></div></div><SkeletonKPI /><SkeletonChart /></div></div>)
  if (!station) return (<div className="flex flex-col items-center justify-center h-full gap-4"><p className="text-text-secondary">Station not found</p><Link to="/" className="text-accent-cyan hover:underline text-sm">Back to observatory</Link></div>)

  const name = isPiezo ? (station.nom_commune || station.code_bss) : (station.libelle_station || station.code_station)
  const hydroLabel = !isPiezo && station?.grandeur_hydro_principale === 'H' ? 'Mean height' : 'Mean discharge'
  const hydroUnit = !isPiezo && station?.grandeur_hydro_principale === 'H' ? 'm' : 'm\u00b3/s'
  const valueKey = resolution === 'daily' ? (isPiezo ? 'niveau_nappe_eau' : 'resultat_obs_elab') : resolution === 'yearly' ? (isPiezo ? 'niveau_moyen_annuel' : 'resultat_moyen_annuel') : (isPiezo ? 'niveau_moyen' : 'resultat_moyen')
  const valueLabel = isPiezo ? 'Water level (m NGF)' : `${hydroLabel} (${hydroUnit})`
  const unit = isPiezo ? 'm NGF' : hydroUnit

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        <div className="flex items-center gap-4">
          <Link to="/" className="p-2 hover:bg-bg-hover rounded-lg transition-colors" aria-label="Back to observatory"><ArrowLeft className="w-5 h-5 text-text-secondary" /></Link>
          <div className="flex-1">
            <p className="text-xs text-accent-cyan font-medium uppercase tracking-wide">{isPiezo ? 'Piezometry' : 'Hydrometry'}</p>
            <h1 className="text-xl font-bold text-text-primary">{name}</h1>
            <p className="text-sm text-text-secondary">{station.nom_departement ?? ''} - {code}{!isPiezo && station.nom_cours_eau && ` - ${station.nom_cours_eau}`}</p>
          </div>
          <div className="flex items-center gap-2">
            {isPiezo && <Link to={`/pastas/station?station=${encodeURIComponent(code)}`} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-accent-cyan/10 text-accent-cyan hover:bg-accent-cyan/20 transition-colors"><Waves className="w-3.5 h-3.5" />Analyze in Pastas</Link>}
            <Link to={`/data?station=${encodeURIComponent(code)}`} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 transition-colors"><Brain className="w-3.5 h-3.5" />Train AI model</Link>
          </div>
        </div>

        <section className="bg-gray-900/50 rounded-xl border border-white/5 p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2"><Info className="w-4 h-4" />Technical Sheet</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6">
            {isPiezo ? (<><div><MetaRow label="Data period" value={formatPeriod(station.premiere_mesure, station.derniere_mesure)} /><MetaRow label="Duration" value={formatDuration(station.nb_mois_total)} /><MetaRow label="Measurements" value={station.nb_mesures_total?.toLocaleString() ?? null} /><MetaRow label="Last measurement" value={formatDateEN(station.derniere_mesure)} /><MetaRow label="Station altitude" value={station.altitude_station != null ? `${station.altitude_station.toFixed(0)} m NGF` : null} /></div><div><MetaRow label="BSS Code" value={station.code_bss ?? null} mono /><MetaRow label="Coordinates" value={station.latitude != null && station.longitude != null ? `${station.latitude.toFixed(4)} N, ${station.longitude.toFixed(4)} E` : null} /><MetaRow label="BDLISA Code" value={station.codes_bdlisa ? (<a href={`https://bdlisa.eaufrance.fr/hydrogeounit/${station.codes_bdlisa.split(',')[0]}`} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">{station.codes_bdlisa}</a>) : null} /><MetaRow label="Historic min" value={station.niveau_min_absolu != null ? `${station.niveau_min_absolu.toFixed(2)} m NGF` : null} /><MetaRow label="Historic max" value={station.niveau_max_absolu != null ? `${station.niveau_max_absolu.toFixed(2)} m NGF` : null} /></div><div className="col-span-full flex flex-wrap gap-3 pt-2 border-t border-white/5"><a href={`https://ades.eaufrance.fr/Fiche/PtEau?code=${station.code_bss}`} target="_blank" rel="noopener noreferrer" className="text-[11px] text-blue-400 hover:underline">ADES</a><a href="https://hubeau.eaufrance.fr/page/api-piezometrie" target="_blank" rel="noopener noreferrer" className="text-[11px] text-blue-400 hover:underline">Hub'Eau</a></div></>) : (<><div><MetaRow label="Data period" value={formatPeriod(station.premiere_mesure, station.derniere_mesure)} /><MetaRow label="Duration" value={formatDuration(station.nb_mois_total)} /><MetaRow label="Last measurement" value={formatDateEN(station.derniere_mesure)} /></div><div><MetaRow label="Station code" value={station.code_station ?? null} mono /><MetaRow label="Coordinates" value={station.latitude_station != null && station.longitude_station != null ? `${station.latitude_station.toFixed(4)} N, ${station.longitude_station.toFixed(4)} E` : null} /><MetaRow label="River code" value={station.code_cours_eau ? (<a href={`https://services.sandre.eaufrance.fr/Courdo/Fiche/client/fiche_courdo.php?CdSandre=${station.code_cours_eau}`} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline font-mono">{station.code_cours_eau}</a>) : null} /><MetaRow label="Historic min" value={station.resultat_min_global != null ? `${station.resultat_min_global.toFixed(2)} ${hydroUnit}` : null} /><MetaRow label="Historic max" value={station.resultat_max_global != null ? `${station.resultat_max_global.toFixed(2)} ${hydroUnit}` : null} /></div><div className="col-span-full flex flex-wrap gap-3 pt-2 border-t border-white/5"><a href={`https://www.vigicrues.gouv.fr/niv3-station.php?CdStationHydro=${station.code_station}`} target="_blank" rel="noopener noreferrer" className="text-[11px] text-blue-400 hover:underline">VigiCrues</a><a href="https://hubeau.eaufrance.fr/page/api-hydrometrie" target="_blank" rel="noopener noreferrer" className="text-[11px] text-blue-400 hover:underline">Hub'Eau</a></div></>)}
          </div>
        </section>

        {isPiezo && piezoSiblings && piezoSiblings.siblings.length > 0 && (
          <section className="bg-gray-900/50 rounded-xl border border-white/5 p-4">
            <h2 className="text-sm font-semibold text-gray-300 mb-1 flex items-center gap-2"><Droplets className="w-4 h-4" />Aquifer<span className="text-xs font-mono text-accent-cyan bg-accent-cyan/10 px-1.5 py-0.5 rounded-full">{piezoSiblings.nb_stations}</span></h2>
            <p className="text-xs text-gray-500 mb-3">Other piezometers in the same BDLISA basin ({piezoSiblings.code_bdlisa}), sorted by proximity</p>
            <div className="overflow-x-auto"><table className="w-full text-xs"><thead><tr className="border-b border-white/5"><th className="text-left py-2 px-2 text-gray-500 font-medium">Station</th><th className="text-left py-2 px-2 text-gray-500 font-medium">Dept.</th><th className="text-left py-2 px-2 text-gray-500 font-medium">Status</th><th className="text-left py-2 px-2 text-gray-500 font-medium">Last measurement</th><th className="text-right py-2 px-2 text-gray-500 font-medium">Distance</th></tr></thead><tbody>{piezoSiblings.siblings.map(sib => (<tr key={sib.code_bss} className="border-b border-white/5 hover:bg-bg-hover transition-colors"><td className="py-1.5 px-2"><Link to={`/station/piezo/${sib.code_bss}`} className="text-accent-cyan hover:underline font-mono">{sib.code_bss}</Link>{sib.nom_commune && <span className="text-gray-500 ml-1.5">{sib.nom_commune}</span>}</td><td className="py-1.5 px-2 text-gray-400">{sib.code_departement}</td><td className="py-1.5 px-2">{sib.classification && <span className="inline-flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ backgroundColor: CLASSIFICATION_COLORS[sib.classification] ?? '#6b7280' }} /><span className="text-gray-300">{sib.classification.replace(/_/g, ' ').toLowerCase()}</span></span>}</td><td className="py-1.5 px-2 text-gray-400">{formatDate(sib.derniere_mesure)}</td><td className="py-1.5 px-2 text-right text-gray-400 font-mono">{sib.distance_km != null ? `${sib.distance_km} km` : '--'}</td></tr>))}</tbody></table></div>
          </section>
        )}

        {!isPiezo && hydroSiblings && hydroSiblings.siblings.length > 0 && (
          <section className="bg-gray-900/50 rounded-xl border border-white/5 p-4">
            <h2 className="text-sm font-semibold text-gray-300 mb-1 flex items-center gap-2"><Waves className="w-4 h-4" />Hydrometric site - {hydroSiblings.nb_stations} stations</h2>
            <p className="text-xs text-gray-500 mb-3">{hydroSiblings.libelle_site || hydroSiblings.code_site}{hydroSiblings.nom_cours_eau ? ` - ${hydroSiblings.nom_cours_eau}` : ''}</p>
            <div className="space-y-1">{hydroSiblings.siblings.map(sib => (<Link key={sib.code_station} to={`/station/hydro/${sib.code_station}`} className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-bg-hover transition-colors"><div><span className="text-xs text-gray-200">{sib.libelle_station || sib.code_station}</span><span className="text-[10px] text-gray-500 ml-2">{sib.code_station}</span></div><div className="flex items-center gap-2">{sib.classification && <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: CLASSIFICATION_COLORS[sib.classification] ?? '#6b7280' }} title={sib.classification} />}</div></Link>))}</div>
          </section>
        )}

        <StationKPICards station={station} type={type} />

        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs text-text-secondary font-medium">Resolution:</span>
          <div role="group" aria-label="Temporal resolution" className="flex gap-1">{RESOLUTION_OPTIONS.filter(opt => { if (opt.value === 'monthly') { const loading = isPiezo ? piezoMonthlyLoading : hydroMonthlyLoading; if (loading) return true; return (monthly?.length ?? 0) > 0 }; return true }).map(opt => (<button key={opt.value} aria-pressed={resolution === opt.value} onClick={() => setResolution(opt.value)} className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${resolution === opt.value ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary hover:text-text-primary'}`}>{opt.label}</button>))}</div>
          {resolution === 'daily' && (<div className="flex items-center gap-2 ml-2"><input type="date" aria-label="Start date" value={dailyStart ?? ''} onChange={(e) => setDailyStart(e.target.value || undefined)} className="bg-bg-card border border-white/10 rounded-lg px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-cyan/50" /><span className="text-xs text-text-secondary">-</span><input type="date" aria-label="End date" value={dailyEnd} onChange={(e) => setDailyEnd(e.target.value)} className="bg-bg-card border border-white/10 rounded-lg px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent-cyan/50" /></div>)}
        </div>

        {activeLoading ? <SkeletonChart /> : activeData && activeData.length > 0 ? (<div className="bg-bg-card border border-white/5 rounded-xl p-5"><TimeseriesChart data={activeData} valueKey={valueKey} valueLabel={valueLabel} unit={unit} precipKey={resolution === 'yearly' ? 'precipitation_totale_annuelle' : 'precipitation_totale'} percentiles={undefined} resolution={resolution} onPeriodChange={resolution === 'daily' ? handleDailyPeriodChange : undefined} /></div>) : (<div className="bg-bg-card border border-white/5 rounded-xl p-5 flex items-center justify-center h-64 text-text-secondary text-sm">No data for this resolution</div>)}

        {(droughtData && droughtData.length > 0) || (spiData && spiData.length > 0) ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {droughtData && droughtData.length > 0 && (<div className="bg-bg-card border border-white/5 rounded-xl p-5"><DroughtIndexChart data={droughtData} indexKey={isPiezo ? 'spli' : 'ssfi'} label={isPiezo ? 'Standardized Piezometric Index (SPLI/IPS)' : 'Standardized Streamflow Index (SSFI)'} /></div>)}
            {spiData && spiData.length > 0 && (<div className="bg-bg-card border border-white/5 rounded-xl p-5"><DroughtIndexChart data={spiData} indexKey="spi" label="Standardized Precipitation Index (SPI)" /></div>)}
          </div>
        ) : null}

        {isPiezo && <PastasSection codeBss={code} />}
      </div>
    </div>
  )
}
