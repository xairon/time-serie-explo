import { GraduationCap } from 'lucide-react'
import type { ModelSummary } from '@/lib/types'
import { MODEL_COLORS, METRIC_LABELS } from '@/lib/constants'

interface ModelCardProps {
  model: ModelSummary
}

export function ModelCard({ model }: ModelCardProps) {
  const color = MODEL_COLORS[model.model_type] ?? '#06b6d4'
  const topMetrics = Object.entries(model.metrics)
    .filter(([key, val]) => val != null && typeof val === 'number' && !key.startsWith('system/') && !key.startsWith('sliding_'))
    .slice(0, 3)

  return (
    <div className="bg-bg-card rounded-xl p-4 border border-white/5 hover:border-accent-cyan/20 transition-colors">
      <div className="flex items-start gap-3">
        <div
          className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
          style={{ backgroundColor: `${color}15` }}
        >
          <GraduationCap className="w-4 h-4" style={{ color }} />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-text-primary truncate">{model.model_name}</h3>
          <p className="text-xs text-text-secondary mt-0.5">
            {model.model_type}
            {model.primary_station ? ` — ${model.primary_station}` : ''}
          </p>
        </div>
      </div>
      {topMetrics.length > 0 && (
        <div className="mt-3 pt-3 border-t border-white/5 flex flex-wrap gap-2">
          {topMetrics.map(([key, val]) => (
            <span
              key={key}
              className="text-[10px] px-2 py-0.5 rounded-full border border-white/10 text-text-secondary"
            >
              {METRIC_LABELS[key] ?? key}: {(val as number).toFixed(4)}
            </span>
          ))}
        </div>
      )}
      <p className="text-[10px] text-text-secondary mt-2">
        {new Date(model.created_at).toLocaleDateString('fr-FR')}
      </p>
    </div>
  )
}
