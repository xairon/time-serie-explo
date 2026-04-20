import { ModelTable } from '@/components/pastas/ModelTable'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'

export default function GalleryPage() {
  return (
    <div className="p-6">
      <OnboardingBanner
        id="gallery"
        title="Bibliothèque de modèles"
        description="Retrouvez tous vos modèles calibrés. Triez par station, performance (EVP, NSE), ou date. Exportez un modèle au format .pas ou en CSV pour le partager."
        steps={[
          'Triez en cliquant sur les en-têtes de colonnes',
          'Filtrez par nom ou code BSS avec la barre de recherche',
          'Téléchargez un modèle (.pas) ou ses métriques (CSV) via l\'icône ↓',
          'Supprimez un modèle avec l\'icône poubelle',
        ]}
      />
      <ModelTable />
    </div>
  )
}
