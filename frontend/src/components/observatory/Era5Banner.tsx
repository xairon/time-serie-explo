import { useTranslation } from 'react-i18next'
import { ERA5_VARIABLES, era5GradientCss } from '@/lib/era5-colors'
import type { Era5Variable } from '@/lib/era5-colors'

interface Props {
  era5Active: boolean
  era5Variable: Era5Variable
  era5Window: number
  /** Active period: a date string like '2024-03-01' or a month like '2024-03-01'. May be empty string. */
  era5Period: string
}

function formatPeriodLabel(period: string): string {
  if (!period) return ''
  // If it looks like YYYY-MM (month-only from timeline), display as-is
  const m = period.match(/^(\d{4})-(\d{2})/)
  if (!m) return period
  const months: Record<string, string> = {
    '01': 'janv.', '02': 'févr.', '03': 'mars', '04': 'avr.',
    '05': 'mai', '06': 'juin', '07': 'juil.', '08': 'août',
    '09': 'sept.', '10': 'oct.', '11': 'nov.', '12': 'déc.',
  }
  return `${months[m[2]] ?? m[2]} ${m[1]}`
}

const ANOMALY_VARIABLES: Era5Variable[] = ['anomaly', 'precipAnomaly']

export function Era5Banner({ era5Active, era5Variable, era5Window, era5Period }: Props) {
  const { t } = useTranslation()

  if (!era5Active) return null

  const cfg = ERA5_VARIABLES[era5Variable]
  const isAnomaly = ANOMALY_VARIABLES.includes(era5Variable)
  const stops = cfg.stops
  const minVal = stops[0][0]
  const maxVal = stops[stops.length - 1][0]
  const periodLabel = formatPeriodLabel(era5Period)

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
          {isAnomaly && (
            <span className="absolute left-1/2 -translate-x-1/2">0</span>
          )}
          <span>+{maxVal} {cfg.unit}</span>
        </div>
      </div>
    </div>
  )
}
