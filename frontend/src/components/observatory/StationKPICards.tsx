import { ClassificationBadge } from './ClassificationBadge'
import { formatNumber } from '@/lib/observatory-utils'

interface Props { station: any; type: 'piezo' | 'hydro' }

export function StationKPICards({ station, type }: Props) {
  const isHauteur = station?.grandeur_hydro_principale === 'H'
  const hydroUnit = isHauteur ? 'm' : 'm\u00b3/s'
  const cards = type === 'piezo'
    ? [
        { title: 'État actuel', value: formatNumber(station.niveau_derniere_annee ?? station.niveau_moyen_global), unit: 'm NGF', sub: (<><ClassificationBadge classification={station.classification_derniere_annee} />{station.niveau_moyen_global != null && <p className="text-xs text-text-secondary mt-1">Moyenne historique : {formatNumber(station.niveau_moyen_global)} m NGF</p>}</>) },
        { title: 'Tendance', value: station.slope_niveau != null ? `${station.slope_niveau > 0 ? '+' : ''}${formatNumber(station.slope_niveau, 3)}` : 'N/A', unit: 'm/an', sub: (<>{station.r2_niveau != null && <span className="text-xs text-text-secondary">R2 = {formatNumber(station.r2_niveau, 2)}</span>}{station.nb_mois_tendance != null && <p className="text-xs text-text-secondary mt-0.5">sur {station.nb_mois_tendance} mois</p>}</>) },
        { title: 'Précipitation moy.', value: formatNumber(station.precipitation_moyenne_mensuelle), unit: 'mm/mois', sub: null },
        { title: 'Température moy.', value: formatNumber(station.temperature_moyenne_globale), unit: '°C', sub: null },
      ]
    : [
        { title: 'État actuel', value: formatNumber(station.resultat_moyen_dern_annee ?? station.resultat_moyen_global, 2), unit: hydroUnit, sub: (<><ClassificationBadge classification={station.classification_resultat_dern_annee} />{station.resultat_moyen_global != null && <p className="text-xs text-text-secondary mt-1">Moyenne historique : {formatNumber(station.resultat_moyen_global, 2)} {hydroUnit}</p>}</>) },
        { title: 'Min / Max', value: `${formatNumber(station.resultat_min_global)} / ${formatNumber(station.resultat_max_global)}`, unit: hydroUnit, sub: null },
        { title: 'Écart-type', value: formatNumber(station.resultat_stddev_global, 2), unit: hydroUnit, sub: null },
        { title: 'Jours de mesure', value: station.nb_jours_total?.toLocaleString() ?? 'N/A', unit: '', sub: null },
      ]
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {cards.map((card) => (<div key={card.title} className="bg-bg-card border border-white/5 rounded-xl p-4"><p className="text-xs text-text-secondary mb-2">{card.title}</p><div className="flex items-baseline gap-1.5"><span className="text-xl font-semibold text-text-primary font-mono">{card.value}</span><span className="text-xs text-text-secondary">{card.unit}</span></div>{card.sub && <div className="mt-2">{card.sub}</div>}</div>))}
    </div>
  )
}
