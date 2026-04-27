import { ModelTable } from '@/components/pastas/ModelTable'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'

export default function GalleryPage() {
  return (
    <div className="p-6">
      <OnboardingBanner
        id="gallery"
        title="Model Library"
        description="Browse all your calibrated models. Sort by station, performance (EVP, NSE), or date. Export a model as .pas or CSV to share it."
        steps={[
          'Sort by clicking on column headers',
          'Filter by name or BSS code with the search bar',
          'Download a model (.pas) or its metrics (CSV) via the download icon',
          'Delete a model with the trash icon',
        ]}
      />
      <ModelTable />
    </div>
  )
}
