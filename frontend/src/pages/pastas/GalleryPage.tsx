import { ModelTable } from '@/components/pastas/ModelTable'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'
import { usePastasMode } from './PastasLayout'

export default function GalleryPage() {
  const { pipeline } = usePastasMode()
  return (
    <div className="p-6">
      <OnboardingBanner
        id="gallery"
        title="Bibliothèque de modèles"
        description="Parcourez vos modèles calibrés. Par défaut, seuls les modèles de la station chargée dans le lab sont affichés ; basculez pour voir toutes les stations. Triez par station, performance (EVP, NSE) ou date."
        steps={[
          'Par défaut, filtré sur la station en cours dans le lab',
          'Triez en cliquant sur les en-têtes de colonne',
          'Filtrez par nom ou code BSS via la barre de recherche',
          'Téléchargez un modèle (.pas) ou ses métriques (CSV) via l\'icône de téléchargement',
        ]}
      />
      <ModelTable defaultCodeBss={pipeline.codeBss || undefined} />
    </div>
  )
}
