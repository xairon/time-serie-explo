import { Trash2, Droplets, TrendingUp, ArrowUpDown } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { PumpingProfileData, PumpingRange, AdaptiveBoundsData } from '@/lib/types'
import {
  PumpingSyntheticEditor,
  type PumpingSyntheticData,
} from './modifications/PumpingSyntheticEditor'
import {
  PumpingUploadEditor,
  type PumpingUploadData,
} from './modifications/PumpingUploadEditor'
import { LinearTrendEditor, type LinearTrendData } from './modifications/LinearTrendEditor'
import { ScaleStressEditor, type ScaleStressData } from './modifications/ScaleStressEditor'

export type ModificationData =
  | PumpingSyntheticData
  | PumpingUploadData
  | LinearTrendData
  | ScaleStressData

interface ModificationCardProps {
  index: number
  data: ModificationData
  onChange: (data: ModificationData) => void
  onDelete: () => void
  pumpingProfiles?: Record<string, PumpingProfileData> | null
  scaleStressLimits?: PumpingRange | null
  linearTrendLimits?: PumpingRange | null
  adaptiveBounds?: AdaptiveBoundsData | null
}

function getTypeMeta(t: (k: string) => string): Record<ModificationData['type'], { label: string; icon: React.ElementType; color: string }> {
  return {
    pumping_synthetic: { label: t('pastas.modCard.well'), icon: Droplets, color: 'text-blue-400' },
    pumping_upload: { label: t('pastas.modCard.pumpingData'), icon: Droplets, color: 'text-blue-400' },
    linear_trend: { label: t('pastas.modCard.trend'), icon: TrendingUp, color: 'text-yellow-400' },
    scale_stress: { label: t('pastas.modCard.climateAdjustment'), icon: ArrowUpDown, color: 'text-purple-400' },
  }
}

export function ModificationCard({ index, data, onChange, onDelete, pumpingProfiles, scaleStressLimits, linearTrendLimits, adaptiveBounds }: ModificationCardProps) {
  const { t } = useTranslation()
  const meta = getTypeMeta(t)[data.type]
  const Icon = meta.icon

  return (
    <div className="bg-bg-primary border border-white/10 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/5">
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-muted">#{index + 1}</span>
          <Icon className={`w-3.5 h-3.5 ${meta.color}`} />
          <span className="text-xs font-medium text-text-secondary">{meta.label}</span>
        </div>
        <button
          onClick={onDelete}
          className="p-1 text-text-muted hover:text-red-400 transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Editor */}
      <div className="p-3">
        {data.type === 'pumping_synthetic' && (
          <PumpingSyntheticEditor data={data} onChange={(d) => onChange(d)} profiles={pumpingProfiles} adaptiveBounds={adaptiveBounds} />
        )}
        {data.type === 'pumping_upload' && (
          <PumpingUploadEditor
            data={data}
            onChange={(d) => onChange(d)}
          />
        )}
        {data.type === 'linear_trend' && (
          <LinearTrendEditor data={data} onChange={(d) => onChange(d)} limits={linearTrendLimits} />
        )}
        {data.type === 'scale_stress' && (
          <ScaleStressEditor data={data} onChange={(d) => onChange(d)} limits={scaleStressLimits} />
        )}
      </div>
    </div>
  )
}
