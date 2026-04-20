import { ModelTable } from '@/components/pastas/ModelTable'

export default function GalleryPage() {
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold text-text-primary mb-4">Pastas — Model Gallery</h1>
      <ModelTable />
    </div>
  )
}
