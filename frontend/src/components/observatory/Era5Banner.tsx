import { useTranslation } from 'react-i18next'
import { ERA5_VARIABLES, era5GradientCss, era5RawDomain, STI_CLASS_ORDER, STI_CLASS_COLORS, SPI_CLASS_ORDER, SPI_CLASS_COLORS } from '@/lib/era5-colors'
import type { Era5Variable, Era5Granularity } from '@/lib/era5-colors'

interface Props {
  era5Active: boolean
  era5Variable: Era5Variable
  era5Window: number
  /** Active period: a date string like '2024-03-01' or a month like '2024-03-01'. May be empty string. */
  era5Period: string
  /** Granularity of the active ERA5 data; drives the raw legend domain. Defaults to 'daily'. */
  era5Granularity?: Era5Granularity
}

const ANOMALY_VARIABLES: Era5Variable[] = ['anomaly']

export function Era5Banner({ era5Active, era5Variable, era5Window, era5Period, era5Granularity = 'daily' }: Props) {
  const { t, i18n } = useTranslation()

  if (!era5Active) return null

  const cfg = ERA5_VARIABLES[era5Variable]
  const isAnomaly = ANOMALY_VARIABLES.includes(era5Variable)
  // Water balance is 0-centred (like an anomaly) but uses the granularity-aware raw domain.
  const isWaterBalance = era5Variable === 'waterBalance'
  // For anomaly variables keep using the fixed divergent stop bounds.
  // For raw variables use the granularity-aware domain so the legend agrees with the map.
  let minVal: number
  let maxVal: number
  if (isAnomaly) {
    const stops = cfg.stops
    minVal = stops[0][0]
    maxVal = stops[stops.length - 1][0]
  } else {
    ;[minVal, maxVal] = era5RawDomain(era5Variable as 'temperature' | 'precipitation' | 'evaporation' | 'waterBalance', era5Granularity)
  }

  let periodLabel = ''
  if (era5Period) {
    const m = era5Period.match(/^(\d{4})-(\d{2})/)
    if (m) {
      const date = new Date(Number(m[1]), Number(m[2]) - 1, 1)
      periodLabel = new Intl.DateTimeFormat(i18n.language, { month: 'short', year: 'numeric' }).format(date)
    } else {
      periodLabel = era5Period
    }
  }

  // STI and SPI share a discrete 7-class McKee legend (only the colours + labels differ).
  if (era5Variable === 'tempStdIndex' || era5Variable === 'precipStdIndex') {
    const isSpi = era5Variable === 'precipStdIndex'
    const classOrder = isSpi ? SPI_CLASS_ORDER : STI_CLASS_ORDER
    const classColors = isSpi ? SPI_CLASS_COLORS : STI_CLASS_COLORS
    const ns = isSpi ? 'observatory.spi' : 'observatory.sti'
    return (
      <div
        className="absolute bottom-16 left-3 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-3 py-2 shadow-lg pointer-events-none"
        style={{ maxWidth: '180px' }}
      >
        <div className="text-xs font-semibold text-text-primary leading-tight">
          {t(cfg.labelKey)}
        </div>
        <div className="text-[10px] text-text-secondary mt-0.5">
          <span>{t(`observatory.drawer.era5Window${era5Window}`)} · </span>
          {periodLabel && <span>{periodLabel}</span>}
        </div>
        <div className="mt-1.5 space-y-0.5">
          {[...classOrder].reverse().map((cls) => (
            <div key={cls} className="flex items-center gap-1.5">
              <span className="w-3 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: classColors[cls] }} />
              <span className="text-[9px] text-text-secondary">{t(`${ns}.${cls}`, { defaultValue: cls })}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div
      className="absolute bottom-16 left-3 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-3 py-2 shadow-lg pointer-events-none"
      style={{ maxWidth: '180px' }}
    >
      {/* Variable label */}
      <div className="text-xs font-semibold text-text-primary leading-tight">
        {t(cfg.labelKey)}
      </div>
      {/* Window + period */}
      <div className="text-[10px] text-text-secondary mt-0.5">
        {isAnomaly && (
          <span>{t(`observatory.drawer.era5Window${era5Window}`)} · </span>
        )}
        {periodLabel && <span>{periodLabel}</span>}
      </div>
      {/* Gradient legend */}
      <div className="mt-1.5">
        <div className="h-2.5 rounded" style={{ background: era5GradientCss(era5Variable) }} />
        <div className="relative flex justify-between text-[9px] text-text-secondary mt-0.5">
          <span>{String(minVal).replace('-', '−')} {cfg.unit}</span>
          {(isAnomaly || isWaterBalance) && (
            <span className="absolute left-1/2 -translate-x-1/2">0</span>
          )}
          <span>+{maxVal} {cfg.unit}</span>
        </div>
      </div>
    </div>
  )
}
