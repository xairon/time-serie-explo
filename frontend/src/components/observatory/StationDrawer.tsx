import { Link } from 'react-router-dom'
import { X, ExternalLink, TrendingUp, TrendingDown, Minus, Droplets, Waves } from 'lucide-react'
import { ClassificationBadge } from './ClassificationBadge'
import { AddToCompareButton } from './AddToCompareButton'
import { formatNumber, formatDate } from '@/lib/observatory-utils'
import { CLASSIFICATION_COLORS } from '@/lib/observatory-constants'
import { usePiezoStationDetail, useHydroStationDetail, useHydroSiblings, useObsPastasSummary } from '@/hooks/useObservatory'

interface Props {
  code: string
  type: 'piezo' | 'hydro'
  onClose: () => void
}

const TREND_CONFIG: Record<string, { label: string; icon: typeof TrendingUp; color: string }> = {
  HAUSSE_FORTE:         { label: 'Hausse forte',          icon: TrendingUp,   color: '#3b82f6' },
  HAUSSE_SIGNIFICATIVE: { label: 'Hausse significative',  icon: TrendingUp,   color: '#60a5fa' },
  STABLE:               { label: 'Stable',                icon: Minus,        color: '#10b981' },
  BAISSE_SIGNIFICATIVE: { label: 'Baisse significative',  icon: TrendingDown, color: '#f97316' },
  BAISSE_FORTE:         { label: 'Baisse forte',          icon: TrendingDown, color: '#ef4444' },
}

function isRecent(dateStr: string | null | undefined): boolean {
  if (!dateStr) return false
  const d = new Date(dateStr); const ago = new Date(); ago.setMonth(ago.getMonth() - 3)
  return d >= ago
}

function DrawerSkeleton({ onClose }: { onClose: () => void }) {
  return (
    <div className="p-4">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 space-y-2"><div className="h-3 w-16 bg-white/10 rounded animate-pulse" /><div className="h-5 w-40 bg-white/10 rounded animate-pulse" /><div className="h-3 w-24 bg-white/10 rounded animate-pulse" /></div>
        <button onClick={onClose} aria-label="Fermer" className="p-1 hover:bg-bg-hover rounded"><X className="w-4 h-4 text-text-secondary" /></button>
      </div>
      <div className="space-y-3"><div className="h-16 w-full bg-white/5 rounded-lg animate-pulse" /><div className="h-16 w-full bg-white/5 rounded-lg animate-pulse" /><div className="h-12 w-full bg-white/5 rounded-lg animate-pulse" /></div>
    </div>
  )
}

function InfoRow({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between py-1.5">
      <span className="text-xs text-text-secondary">{label}</span>
      <div className="text-right"><span className="text-xs text-text-primary font-medium">{value}</span>{sub && <div className="text-[10px] text-text-secondary">{sub}</div>}</div>
    </div>
  )
}

export function StationDrawer({ code, type, onClose }: Props) {
  const isPiezo = type === 'piezo'
  const piezoQuery = usePiezoStationDetail(isPiezo ? code : '')
  const hydroQuery = useHydroStationDetail(!isPiezo ? code : '')
  const { data: station, isLoading, isError } = isPiezo ? piezoQuery : hydroQuery

  const stationCode = isPiezo ? (station as any)?.code_bss ?? code : (station as any)?.code_station ?? code
  const hydroSiblings = useHydroSiblings(!isPiezo ? stationCode : '')
  const { data: pastasSummary } = useObsPastasSummary(isPiezo ? stationCode : '')

  const content = (() => {
    if (isError) return (<div className="p-4"><div className="flex items-start justify-between mb-4"><p className="text-sm text-red-400">Impossible de charger cette station.</p><button onClick={onClose} aria-label="Fermer" className="p-1 hover:bg-bg-hover rounded"><X className="w-4 h-4 text-text-secondary" /></button></div></div>)
    if (isLoading || !station) return <DrawerSkeleton onClose={onClose} />

    const s = station as any
    const name = isPiezo ? (s.nom_commune || s.code_bss) : (s.libelle_station || s.code_station)
    const sCode = isPiezo ? s.code_bss : s.code_station
    const dept = s.nom_departement ?? s.code_departement ?? ''
    const classification = isPiezo ? s.classification_derniere_annee : s.classification_resultat_dern_annee
    const classColor = CLASSIFICATION_COLORS[classification] ?? '#6b7280'
    const currentValue = isPiezo ? s.niveau_derniere_annee : s.resultat_moyen_dern_annee
    const historicMean = isPiezo ? s.niveau_moyen_global : s.resultat_moyen_global
    const isHauteur = s.grandeur_hydro_principale === 'H'
    const unit = isPiezo ? 'm NGF' : (isHauteur ? 'm' : 'm\u00b3/s')
    const trendKey = s.tendance_classification
    const trendConf = trendKey ? TREND_CONFIG[trendKey] : null
    const slope = isPiezo ? s.slope_niveau : null
    const histMin = isPiezo ? s.niveau_min_absolu : s.resultat_min_global
    const histMax = isPiezo ? s.niveau_max_absolu : s.resultat_max_global
    const lastMeasure = s.derniere_mesure
    const recent = isRecent(lastMeasure)
    const dataCount = isPiezo ? s.nb_mesures_total : s.nb_jours_total
    const dataPeriodYears = s.nb_mois_total ? Math.round(s.nb_mois_total / 12) : null

    return (
      <div className="p-4 space-y-3">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium uppercase tracking-wide ${isPiezo ? 'bg-accent-cyan/20 text-accent-cyan' : 'bg-accent-indigo/20 text-accent-indigo'}`}>{isPiezo ? 'Piézomètre' : 'Hydrométrie'}</span>
              <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${recent ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>{recent ? 'Active' : 'Inactive'}</span>
              {pastasSummary && (<span className="text-[10px] px-1.5 py-0.5 rounded font-medium" style={{ backgroundColor: (pastasSummary.evp ?? 0) >= 70 ? 'rgba(16,185,129,0.2)' : 'rgba(245,158,11,0.2)', color: (pastasSummary.evp ?? 0) >= 70 ? '#10b981' : '#f59e0b' }}>PASTAS {pastasSummary.evp != null ? `${pastasSummary.evp.toFixed(0)}%` : ''}</span>)}
            </div>
            <h3 className="text-base font-semibold text-text-primary mt-1.5 break-words leading-tight">{name}</h3>
            <p className="text-xs text-text-secondary mt-0.5">{dept} - {sCode}</p>
            {!isPiezo && s.nom_cours_eau && (<p className="text-xs text-accent-indigo/80 mt-0.5 flex items-center gap-1"><Waves className="w-3 h-3" />{s.nom_cours_eau}</p>)}
          </div>
          <button onClick={onClose} aria-label="Fermer" className="p-1 hover:bg-bg-hover rounded ml-2 flex-shrink-0"><X className="w-4 h-4 text-text-secondary" /></button>
        </div>

        {recent ? (
          <div className="bg-white/[0.03] rounded-lg p-3 border border-white/5">
            <div className="text-[10px] uppercase tracking-wider text-text-secondary mb-2">État actuel</div>
            <div className="flex items-center justify-between">
              <ClassificationBadge classification={classification} />
              {currentValue != null && (<span className="text-lg font-semibold font-mono" style={{ color: classColor }}>{formatNumber(currentValue)} <span className="text-xs text-text-secondary font-normal">{unit}</span></span>)}
            </div>
            {historicMean != null && currentValue != null && (<div className="mt-2 text-[11px] text-text-secondary">Moyenne historique : <span className="text-text-primary font-mono">{formatNumber(historicMean)}</span> {unit}<span className="ml-1.5">({currentValue > historicMean ? '+' : ''}{formatNumber(currentValue - historicMean, 2)} {unit})</span></div>)}
          </div>
        ) : (
          <div className="bg-amber-500/10 rounded-lg p-3 border border-amber-500/20"><div className="text-xs text-amber-400">Station inactive — dernière mesure le {formatDate(lastMeasure)}</div></div>
        )}

        {recent && trendConf && (
          <div className="bg-white/[0.03] rounded-lg p-3 border border-white/5">
            <div className="text-[10px] uppercase tracking-wider text-text-secondary mb-2">Tendance</div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2"><trendConf.icon className="w-4 h-4" style={{ color: trendConf.color }} /><span className="text-sm font-medium" style={{ color: trendConf.color }}>{trendConf.label}</span></div>
              {slope != null && (<span className="text-xs font-mono text-text-primary">{slope > 0 ? '+' : ''}{formatNumber(slope, 3)} m/an</span>)}
            </div>
          </div>
        )}

        <div className="bg-white/[0.03] rounded-lg p-3 border border-white/5">
          <div className="text-[10px] uppercase tracking-wider text-text-secondary mb-2">Historique</div>
          <div className="divide-y divide-white/5">
            {histMin != null && histMax != null && <InfoRow label="Plage" value={<><span className="font-mono">{formatNumber(histMin)}</span> -- <span className="font-mono">{formatNumber(histMax)}</span> {unit}</>} />}
            <InfoRow label="Dernière mesure" value={formatDate(lastMeasure)} />
            <InfoRow label="Données" value={<>{dataCount?.toLocaleString() ?? '--'} {isPiezo ? 'mesures' : 'jours'}</>} sub={dataPeriodYears ? `${dataPeriodYears} années d'enregistrement` : undefined} />
          </div>
        </div>

        {!isPiezo && hydroSiblings.data && hydroSiblings.data.siblings.length > 0 && (
          <div className="bg-white/[0.03] rounded-lg p-3 border border-white/5">
            <div className="text-[10px] uppercase tracking-wider text-text-secondary mb-2">Site hydrométrique - {hydroSiblings.data.nb_stations} stations</div>
            <p className="text-xs text-text-primary mb-2 font-medium">{hydroSiblings.data.libelle_site || hydroSiblings.data.code_site}{hydroSiblings.data.nom_cours_eau ? ` - ${hydroSiblings.data.nom_cours_eau}` : ''}</p>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {hydroSiblings.data.siblings.slice(0, 5).map(sib => (
                <Link key={sib.code_station} to={`/station/hydro/${sib.code_station}`} className="flex items-center justify-between py-1 px-1.5 rounded hover:bg-bg-hover text-xs transition-colors">
                  <span className="text-text-primary truncate">{sib.libelle_station || sib.code_station}</span>
                  <span className="flex items-center gap-1.5 shrink-0 ml-2">
                    {sib.classification && <span className="w-2 h-2 rounded-full" style={{ backgroundColor: CLASSIFICATION_COLORS[sib.classification] ?? '#6b7280' }} title={sib.classification} />}
                  </span>
                </Link>
              ))}
            </div>
          </div>
        )}

        {recent && s.niveau_alerte && (<div className="bg-red-500/10 rounded-lg p-3 border border-red-500/20"><div className="flex items-center gap-2"><Droplets className="w-4 h-4 text-red-400" /><span className="text-xs font-medium text-red-400">Alerte : {s.niveau_alerte}</span></div></div>)}

        <div className="flex items-center gap-2">
          <Link to={`/station/${type}/${sCode}`} className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium bg-accent-cyan/10 text-accent-cyan hover:bg-accent-cyan/20 transition-colors">Voir les détails <ExternalLink className="w-4 h-4" /></Link>
          <AddToCompareButton code={sCode} type={type} variant="compact" />
        </div>
      </div>
    )
  })()

  return (
    <>
      <div className="md:hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-40" onClick={onClose} />
      <div role="dialog" aria-label={`Station ${code}`} className="absolute top-0 left-0 h-full z-40 w-full sm:w-80 bg-bg-card border-r border-white/10 shadow-2xl transition-transform duration-200 ease-out overflow-y-auto translate-x-0">{content}</div>
    </>
  )
}
