import { useState, useMemo } from 'react'
import { useModels, useModelDetail } from '@/hooks/useModels'
import { MODEL_COLORS } from '@/lib/constants'

interface ModelSelectorProps {
  value: string
  onChange: (id: string) => void
}

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  const { data: models, isLoading } = useModels()
  const [stationFilter, setStationFilter] = useState<string>('')
  const { data: selectedDetail } = useModelDetail(value || null)

  // Extract unique stations from all models
  const stations = useMemo(() => {
    if (!models?.length) return []
    const stationSet = new Set<string>()
    for (const m of models) {
      if (m.primary_station) stationSet.add(m.primary_station)
      for (const s of m.stations) stationSet.add(s)
    }
    return Array.from(stationSet).sort()
  }, [models])

  // Filter models by selected station
  const filteredModels = useMemo(() => {
    if (!models?.length) return []
    if (!stationFilter) return models
    return models.filter(
      (m) =>
        m.primary_station === stationFilter ||
        m.stations.includes(stationFilter),
    )
  }, [models, stationFilter])

  if (isLoading) {
    return <div className="h-9 bg-bg-hover rounded-lg animate-pulse" />
  }

  if (!models?.length) {
    return (
      <p className="text-xs text-text-secondary italic">
        Aucun modele entraine. Rendez-vous sur la page Entrainement.
      </p>
    )
  }

  const formatLabel = (m: typeof models[0]): string => {
    const mae = m.metrics['MAE'] ?? m.metrics['mae']
    const rmse = m.metrics['RMSE'] ?? m.metrics['rmse']
    const metricStr = mae != null
      ? `MAE=${mae.toFixed(4)}`
      : rmse != null
        ? `RMSE=${rmse.toFixed(4)}`
        : ''
    const dateStr = m.created_at
      ? new Date(m.created_at).toLocaleDateString('fr-FR')
      : ''
    const parts = [m.model_name]
    if (metricStr) parts[0] += ` (${metricStr})`
    if (dateStr) parts.push(dateStr)
    return parts.join(' — ')
  }

  return (
    <div className="space-y-2">
      {/* Station filter */}
      {stations.length > 1 && (
        <div>
          <label className="block text-xs text-text-secondary mb-1">Station</label>
          <select
            value={stationFilter}
            onChange={(e) => {
              setStationFilter(e.target.value)
              // Reset model selection when station changes
              onChange('')
            }}
            className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">Toutes les stations</option>
            {stations.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Model selector */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Modele</label>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
        >
          <option value="">Selectionner un modele</option>
          {filteredModels.map((m) => (
            <option key={m.model_id} value={m.model_id}>
              {formatLabel(m)}
            </option>
          ))}
        </select>
      </div>

      {/* Info card for selected model */}
      {value && selectedDetail && (
        <div className="bg-bg-card rounded-lg border border-white/5 p-3 space-y-2">
          <div className="flex items-center gap-2">
            <span
              className="text-xs font-semibold px-2 py-0.5 rounded-full border border-white/10"
              style={{ color: MODEL_COLORS[selectedDetail.model_type] ?? '#06b6d4' }}
            >
              {selectedDetail.model_type}
            </span>
            <span className="text-xs text-text-secondary">
              {new Date(selectedDetail.created_at).toLocaleDateString('fr-FR', {
                day: '2-digit',
                month: 'short',
                year: 'numeric',
              })}
            </span>
          </div>
          <div className="text-xs text-text-secondary">
            Station(s) : {selectedDetail.stations.join(', ') || selectedDetail.primary_station || '—'}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(selectedDetail.metrics)
              .filter(
                ([key, val]) =>
                  val != null &&
                  typeof val === 'number' &&
                  !key.startsWith('system/') &&
                  !key.startsWith('sliding_'),
              )
              .slice(0, 4)
              .map(([key, val]) => (
                <span
                  key={key}
                  className="text-[10px] px-2 py-0.5 rounded-full border border-white/10"
                  style={{ color: MODEL_COLORS[selectedDetail.model_type] ?? '#06b6d4' }}
                >
                  {key}: {(val as number).toFixed(4)}
                </span>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
