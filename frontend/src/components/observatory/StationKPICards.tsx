import { useTranslation } from 'react-i18next'
import { formatNumber } from '@/lib/observatory-utils'

interface Props { station: any; type: 'piezo' | 'hydro' }

export function StationKPICards({ station, type }: Props) {
  const { t } = useTranslation()
  const isHauteur = station?.grandeur_hydro_principale === 'H'
  const hydroUnit = isHauteur ? 'm' : 'm³/s'
  const cards = type === 'piezo'
    ? [
        { title: t('observatory.kpi.currentState'), value: formatNumber(station.niveau_derniere_annee ?? station.niveau_moyen_global), unit: 'm NGF', sub: (station.niveau_moyen_global != null ? <p className="text-xs text-text-secondary mt-1">{t('observatory.kpi.historicalMean')} : {formatNumber(station.niveau_moyen_global)} m NGF</p> : null) },
        { title: t('observatory.kpi.precipMean'), value: formatNumber(station.precipitation_moyenne_mensuelle), unit: 'mm/mois', sub: null },
        { title: t('observatory.kpi.tempMean'), value: formatNumber(station.temperature_moyenne_globale), unit: '°C', sub: null },
      ]
    : [
        { title: t('observatory.kpi.currentState'), value: formatNumber(station.resultat_moyen_dern_annee ?? station.resultat_moyen_global, 2), unit: hydroUnit, sub: (station.resultat_moyen_global != null ? <p className="text-xs text-text-secondary mt-1">{t('observatory.kpi.historicalMean')} : {formatNumber(station.resultat_moyen_global, 2)} {hydroUnit}</p> : null) },
        { title: t('observatory.kpi.minMax'), value: `${formatNumber(station.resultat_min_global)} / ${formatNumber(station.resultat_max_global)}`, unit: hydroUnit, sub: null },
        { title: t('observatory.kpi.stddev'), value: formatNumber(station.resultat_stddev_global, 2), unit: hydroUnit, sub: null },
        { title: t('observatory.kpi.measurementDays'), value: station.nb_jours_total?.toLocaleString() ?? 'N/A', unit: '', sub: null },
      ]
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {cards.map((card) => (<div key={card.title} className="bg-bg-card border border-white/5 rounded-xl p-4"><p className="text-xs text-text-secondary mb-2">{card.title}</p><div className="flex items-baseline gap-1.5"><span className="text-xl font-semibold text-text-primary font-mono">{card.value}</span><span className="text-xs text-text-secondary">{card.unit}</span></div>{card.sub && <div className="mt-2">{card.sub}</div>}</div>))}
    </div>
  )
}
