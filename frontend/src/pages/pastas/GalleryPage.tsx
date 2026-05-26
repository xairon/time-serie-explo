import { ModelTable } from '@/components/pastas/ModelTable'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'

export default function GalleryPage() {
  return (
    <div className="p-6">
      <OnboardingBanner
        id="gallery"
        title="Bibliothèque de modèles"
        description="Parcourez tous vos modèles calibrés. Triez par station, performance (EVP, NSE) ou date. Exportez un modèle en .pas ou CSV pour le partager."
        steps={[
          'Triez en cliquant sur les en-têtes de colonne',
          'Filtrez par nom ou code BSS via la barre de recherche',
          'Téléchargez un modèle (.pas) ou ses métriques (CSV) via l\'icône de téléchargement',
          'Supprimez un modèle avec l\'icône de corbeille',
        ]}
      />
      <ModelTable />
    </div>
  )
}
