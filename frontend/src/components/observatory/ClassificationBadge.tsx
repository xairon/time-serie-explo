import { CLASSIFICATION_COLORS, CLASSIFICATION_LABELS } from '@/lib/observatory-constants'

export function ClassificationBadge({ classification }: { classification: string | null | undefined }) {
  if (!classification) return <span className="text-text-secondary text-xs">N/A</span>
  const color = CLASSIFICATION_COLORS[classification] ?? '#6b7280'
  const label = CLASSIFICATION_LABELS[classification] ?? classification
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
      style={{ backgroundColor: `${color}20`, color }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} aria-hidden="true" />
      {label}
    </span>
  )
}
