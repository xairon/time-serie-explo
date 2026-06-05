import { AIModelTable } from '@/components/forecasting/AIModelTable'

export default function AIModelsPage() {
  return (
    <div className="p-6 max-w-7xl mx-auto space-y-4">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Modèles entraînés</h1>
        <p className="text-sm text-text-secondary mt-1">
          Tous vos modèles IA calibrés. Triez par métrique, lancez une prévision ou supprimez.
        </p>
      </div>
      <AIModelTable />
    </div>
  )
}
