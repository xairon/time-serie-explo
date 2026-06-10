// frontend/src/components/meteo/AboutModal.tsx
interface Props {
  onClose: () => void
}

export function AboutModal({ onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose} role="dialog" aria-modal="true" aria-label="À propos">
      <div className="bg-white rounded-xl shadow-xl max-w-lg w-[min(92vw,520px)] p-6" onClick={(e) => e.stopPropagation()}>
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
        </div>
      </div>
    </div>
  )
}
