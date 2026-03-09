import { Link } from 'react-router-dom'
import { Cpu, Database, GraduationCap, Server } from 'lucide-react'
import { useHealth } from '@/hooks/useHealth'
import { useDatasets } from '@/hooks/useDatasets'
import { useModels } from '@/hooks/useModels'
import { StatusCard } from '@/components/cards/StatusCard'
import { DatasetCard } from '@/components/cards/DatasetCard'
import { ModelCard } from '@/components/cards/ModelCard'

export default function DashboardPage() {
  const { data: health, isLoading: healthLoading } = useHealth()
  const { data: datasets, isLoading: datasetsLoading } = useDatasets()
  const { data: models, isLoading: modelsLoading } = useModels()

  return (
    <div className="p-6 space-y-8 max-w-7xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-1">Dashboard</h1>
        <p className="text-sm text-text-secondary">Vue d'ensemble de la plateforme Junon</p>
      </div>

      {/* Status cards */}
      {healthLoading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-24 bg-bg-card rounded-xl animate-pulse border border-white/5" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatusCard
            label="GPU"
            value={health?.gpu?.available ? health.gpu.device ?? 'Disponible' : 'Indisponible'}
            icon={Cpu}
            status={health?.gpu?.available ? 'ok' : 'error'}
          />
          <StatusCard
            label="Datasets"
            value={datasets?.length ?? 0}
            icon={Database}
            status="neutral"
          />
          <StatusCard
            label="Modèles"
            value={models?.length ?? 0}
            icon={GraduationCap}
            status="neutral"
          />
          <StatusCard
            label="Redis"
            value={health?.redis === 'ok' ? 'Connecté' : 'Hors ligne'}
            icon={Server}
            status={health?.redis === 'ok' ? 'ok' : 'error'}
          />
        </div>
      )}

      {/* Datasets section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-text-primary">Datasets</h2>
          <Link
            to="/data"
            className="text-xs text-accent-cyan hover:underline"
          >
            Gérer les données
          </Link>
        </div>

        {datasetsLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <div
                key={i}
                className="h-28 bg-bg-card rounded-xl animate-pulse border border-white/5"
              />
            ))}
          </div>
        ) : datasets && datasets.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map((ds) => (
              <DatasetCard key={ds.id} dataset={ds} />
            ))}
          </div>
        ) : (
          <div className="bg-bg-card rounded-xl border border-white/5 p-8 text-center">
            <Database className="w-8 h-8 text-text-secondary mx-auto mb-2" />
            <p className="text-sm text-text-secondary mb-3">Aucun dataset importé</p>
            <Link
              to="/data"
              className="text-sm text-accent-cyan hover:underline"
            >
              Importer des données
            </Link>
          </div>
        )}
      </section>

      {/* Models section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-text-primary">Modèles entraînés</h2>
          <Link
            to="/training"
            className="text-xs text-accent-cyan hover:underline"
          >
            Entraîner un modèle
          </Link>
        </div>

        {modelsLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <div
                key={i}
                className="h-28 bg-bg-card rounded-xl animate-pulse border border-white/5"
              />
            ))}
          </div>
        ) : models && models.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((m) => (
              <ModelCard key={m.id} model={m} />
            ))}
          </div>
        ) : (
          <div className="bg-bg-card rounded-xl border border-white/5 p-8 text-center">
            <GraduationCap className="w-8 h-8 text-text-secondary mx-auto mb-2" />
            <p className="text-sm text-text-secondary mb-3">Aucun modèle entraîné</p>
            <Link
              to="/training"
              className="text-sm text-accent-cyan hover:underline"
            >
              Démarrer un entraînement
            </Link>
          </div>
        )}
      </section>
    </div>
  )
}
