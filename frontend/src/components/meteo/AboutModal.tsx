// frontend/src/components/meteo/AboutModal.tsx
import { useEffect } from 'react'

interface Props {
  onClose: () => void
}

export function AboutModal({ onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div
        className="bg-white rounded-xl shadow-xl max-w-lg w-[min(92vw,520px)] p-6"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="À propos"
      >
        <div className="flex items-start justify-between mb-3">
          <h2 className="text-lg font-bold text-slate-800">À propos</h2>
          <button onClick={onClose} aria-label="Fermer" className="p-1 rounded hover:bg-slate-100 text-slate-400 hover:text-slate-600">
            <svg width="16" height="16" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        <div className="space-y-3 text-sm text-slate-600 leading-relaxed">
          <p>
            Cette carte présente la <strong>situation des nappes phréatiques</strong> par
            secteur hydrogéologique, dans l'esprit du bulletin MétéEAU Nappes du BRGM,
            calculée à partir des données de la plateforme Junon.
          </p>
          <p>
            Le niveau de chaque secteur est déterminé par l'<strong>Indicateur Piézométrique
            Standardisé (IPS)</strong> des stations qui le composent, calculé sur une période
            de référence fixe <strong>1991-2020</strong>. Les flèches indiquent l'évolution
            des niveaux par rapport au mois précédent.
          </p>
          <p>
            Les secteurs affichés sont les secteurs hydrogéologiques du Bulletin de
            Situation Hydrologique (BSH), © BRGM / Eaufrance. Les mesures piézométriques
            proviennent du réseau ADES (Hub'Eau).
          </p>
          <div>
            <p className="font-semibold text-slate-700 mb-1">
              Pourquoi la carte peut différer du bulletin MétéEAU Nappes officiel ?
            </p>
            <ul className="list-disc pl-5 space-y-1 text-[13px]">
              <li>
                <strong>Période de référence</strong> : nos indicateurs sont calculés sur la
                normale fixe 1991-2020 ; le BRGM situe chaque mois par rapport à
                l'ensemble de la chronique disponible de chaque piézomètre.
              </li>
              <li>
                <strong>Jeu de stations</strong> : le bulletin officiel repose sur un réseau
                restreint de piézomètres sélectionnés et qualifiés par des hydrogéologues ;
                nous utilisons l'ensemble des stations ADES disposant d'un historique suffisant.
              </li>
              <li>
                <strong>Agrégation</strong> : la classe d'un secteur est ici la médiane
                automatique des stations qui le composent ; le bulletin BRGM est validé et
                ajusté par des experts régionaux avant publication.
              </li>
              <li>
                <strong>Date de calcul</strong> : le bulletin décrit la situation au 1er du
                mois ; notre carte agrège les mesures de l'ensemble du mois.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
