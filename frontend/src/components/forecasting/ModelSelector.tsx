import { useModels } from '@/hooks/useModels'
import { MODEL_COLORS } from '@/lib/constants'

interface ModelSelectorProps {
  value: string
  onChange: (id: string) => void
}

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  const { data: models, isLoading } = useModels()

  if (isLoading) {
    return <div className="h-9 bg-bg-hover rounded-lg animate-pulse" />
  }

  if (!models?.length) {
    return (
      <p className="text-xs text-text-secondary italic">
        Aucun modèle entraîné. Rendez-vous sur la page Entraînement.
      </p>
    )
  }

  const selected = models.find((m) => m.id === value)

  return (
    <div className="space-y-2">
      <label className="block text-xs text-text-secondary">Modèle</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
      >
        <option value="">Sélectionner un modèle</option>
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name} ({m.model_type})
          </option>
        ))}
      </select>
      {selected && (
        <div className="flex flex-wrap gap-1.5 mt-1">
          {Object.entries(selected.metrics)
            .slice(0, 3)
            .map(([key, val]) => (
              <span
                key={key}
                className="text-[10px] px-2 py-0.5 rounded-full border border-white/10"
                style={{ color: MODEL_COLORS[selected.model_type] ?? '#06b6d4' }}
              >
                {key}: {val.toFixed(4)}
              </span>
            ))}
        </div>
      )}
    </div>
  )
}
