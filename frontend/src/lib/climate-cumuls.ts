// Pure helpers for the "Contexte climatique" station section (Task C2).
// No statistics are recomputed here beyond simple sums: the monthly values and
// the 1991-2020 calendar-month normals both come from the warehouse marts via
// GET /observatory/climat/point-series.
import type { ClimatPointSeriesEntry, SPIDataPoint } from './observatory-types'
import { SPI_CLASS_COLORS } from './era5-colors'

/** Rolling cumulative precipitation over the last N months vs. the climatological normal. */
export interface RollingCumul {
  /** Window length in months (3, 6, 12…). */
  window: number
  /** Sum of observed monthly precipitation over the window (mm). */
  cumul: number
  /** Sum of the 1991-2020 calendar-month normals over the same months (mm). */
  normale: number
  /** cumul − normale (mm): negative = deficit. */
  ecartMm: number
  /** Relative deviation in % ((cumul − normale) / normale × 100); null when the normal is 0. */
  ecartPct: number | null
  /** First month of the window (YYYY-MM-DD). */
  from: string
  /** Last month of the window (YYYY-MM-DD). */
  to: string
}

/** 0-based absolute month index (year × 12 + month) used to detect calendar gaps. */
function monthIndex(iso: string): number {
  const d = new Date(iso)
  return d.getUTCFullYear() * 12 + d.getUTCMonth()
}

/**
 * Computes the rolling precipitation cumuls vs. normal for each requested window.
 *
 * The windows are anchored on the last COMPLETE month with an observed value
 * (`mois_complet !== false` and non-null `precipitation_totale`) — the current
 * partial month would otherwise bias every cumul low. A window is returned as
 * `null` when it cannot be computed honestly: not enough months, a calendar gap
 * inside the window, or a missing observed/normal value for any month.
 *
 * Returns one entry per requested window, in the same order.
 */
export function computeRollingCumuls(
  series: ClimatPointSeriesEntry[],
  windows: number[],
): (RollingCumul | null)[] {
  const sorted = series
    .slice()
    .sort((a, b) => (a.month < b.month ? -1 : a.month > b.month ? 1 : 0))

  // Anchor: last complete month with an observed precipitation value.
  let anchor = -1
  for (let i = sorted.length - 1; i >= 0; i--) {
    const e = sorted[i]
    if (e.precipitation_totale != null && e.mois_complet !== false) { anchor = i; break }
  }
  if (anchor < 0) return windows.map(() => null)

  return windows.map((w) => {
    if (w <= 0 || anchor + 1 < w) return null
    const slice = sorted.slice(anchor + 1 - w, anchor + 1)
    // Reject calendar gaps: months must be consecutive.
    for (let i = 1; i < slice.length; i++) {
      if (monthIndex(slice[i].month) !== monthIndex(slice[i - 1].month) + 1) return null
    }
    let cumul = 0
    let normale = 0
    for (const e of slice) {
      if (e.precipitation_totale == null || e.precipitation_normale == null) return null
      cumul += e.precipitation_totale
      normale += e.precipitation_normale
    }
    const ecartMm = cumul - normale
    return {
      window: w,
      cumul,
      normale,
      ecartMm,
      ecartPct: normale !== 0 ? (ecartMm / normale) * 100 : null,
      from: slice[0].month,
      to: slice[slice.length - 1].month,
    }
  })
}

/** Latest data point with a non-null SPI value (the series may end on null months). */
export function latestSpiPoint(data: SPIDataPoint[]): SPIDataPoint | null {
  for (let i = data.length - 1; i >= 0; i--) {
    if (data[i].spi != null) return data[i]
  }
  return null
}

/** Visual mapping for a WMO/McKee SPI class badge. */
export interface SpiBadge {
  /** Background colour (SPI_CLASS_COLORS, precipitation-oriented BrBG). */
  color: string
  /** Readable text colour for that background (dark on light classes, white on dark ones). */
  textColor: string
  /** i18n key of the class label (observatory.spi.*). */
  labelKey: string
}

/** Relative luminance (0-1) of a #rrggbb colour — enough to pick a readable text colour. */
function luminance(hex: string): number {
  const n = parseInt(hex.slice(1), 16)
  const r = (n >> 16) & 0xff
  const g = (n >> 8) & 0xff
  const b = n & 0xff
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
}

/** Maps an index_class string to its badge colour, text colour and i18n label key.
 *  Unknown classes fall back to the grey UNKNOWN badge. */
export function spiClassBadge(cls: string | null | undefined): SpiBadge {
  const key = cls != null && cls in SPI_CLASS_COLORS ? cls : 'UNKNOWN'
  const color = SPI_CLASS_COLORS[key]
  return {
    color,
    textColor: luminance(color) > 0.55 ? '#1f2937' : '#ffffff',
    labelKey: `observatory.spi.${key}`,
  }
}
