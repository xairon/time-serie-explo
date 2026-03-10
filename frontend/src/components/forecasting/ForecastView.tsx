import { ForecastPlot } from '@/components/charts/ForecastPlot'
import type { ForecastResult } from '@/lib/types'

interface ForecastViewProps {
  result: ForecastResult | null
  isLoading: boolean
  inputChunkLength?: number
  outputChunkLength?: number
  className?: string
}

export function ForecastView({ result, isLoading, inputChunkLength, outputChunkLength, className = '' }: ForecastViewProps) {
  if (isLoading) {
    return (
      <div className={`bg-bg-card rounded-xl border border-white/5 p-6 ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-bg-hover rounded w-1/3" />
          <div className="h-[300px] bg-bg-hover rounded-lg" />
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div
        className={`bg-bg-card rounded-xl border border-white/5 p-6 flex items-center justify-center ${className}`}
      >
        <p className="text-text-secondary text-sm">
          Selectionnez un modele et lancez une prevision pour afficher les resultats.
        </p>
      </div>
    )
  }

  return (
    <div className={`bg-bg-card rounded-xl border border-white/5 p-4 ${className}`}>
      <ForecastPlot
        result={result}
        inputChunkLength={inputChunkLength}
        outputChunkLength={outputChunkLength}
        className="h-[400px]"
      />
    </div>
  )
}
