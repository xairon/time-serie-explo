export type Era5Variable = 'temperature' | 'precipitation' | 'evaporation' | 'waterBalance' | 'anomaly' | 'tempStdIndex' | 'precipStdIndex'
export type Era5Granularity = 'daily' | 'monthly'

export interface Era5VarConfig {
  key: Era5Variable
  prop: 'temperature_2m' | 'total_precipitation' | 'potential_evaporation' | 'water_balance' | 'anomaly' | 'sti' | 'spi'
  unit: string
  labelKey: string
  stops: Array<[number, string]>
}

export const ERA5_VARIABLES: Record<Era5Variable, Era5VarConfig> = {
  temperature: {
    key: 'temperature', prop: 'temperature_2m', unit: '°C',
    labelKey: 'observatory.drawer.era5VarTemperature',
    stops: [[-10, '#3b2d8c'], [-5, '#3d6fd0'], [0, '#4aa3e0'], [5, '#7fd0e8'], [10, '#9fdfa8'], [15, '#e6e36a'], [20, '#f4b942'], [25, '#ef7d2f'], [30, '#df3b2c'], [35, '#c01f8a']],
  },
  precipitation: {
    key: 'precipitation', prop: 'total_precipitation', unit: 'mm',
    labelKey: 'observatory.drawer.era5VarPrecipitation',
    stops: [[0, '#f7fbff'], [5, '#c6dbef'], [15, '#6baed6'], [30, '#2171b5'], [50, '#08306b']],
  },
  evaporation: {
    key: 'evaporation', prop: 'potential_evaporation', unit: 'mm',
    labelKey: 'observatory.drawer.era5VarEvaporation',
    // stored negative; more negative = more evapotranspiration
    stops: [[-10, '#54278f'], [-6, '#756bb1'], [-3, '#9e9ac8'], [-1, '#cbc9e2'], [0, '#f2f0f7']],
  },
  waterBalance: {
    key: 'waterBalance', prop: 'water_balance', unit: 'mm',
    labelKey: 'observatory.drawer.era5VarWaterBalance',
    // Climatic water balance P − ETP (mm). Divergent around 0: deficit = red/brown,
    // surplus = blue. Symmetric stops; the display domain is rescaled per granularity.
    stops: [[-100, '#b2182b'], [-40, '#ef8a62'], [-10, '#fddbc7'], [0, '#f7f7f7'], [10, '#d1e5f0'], [40, '#67a9cf'], [100, '#2166ac']],
  },
  anomaly: {
    key: 'anomaly', prop: 'anomaly', unit: '°C',
    labelKey: 'observatory.drawer.era5VarAnomaly',
    stops: [[-5, '#2166ac'], [-2.5, '#67a9cf'], [-0.5, '#d1e5f0'], [0, '#f7f7f7'], [0.5, '#fddbc7'], [2.5, '#ef8a62'], [5, '#b2182b']],
  },
  precipStdIndex: {
    key: 'precipStdIndex', prop: 'spi', unit: 'σ',
    labelKey: 'observatory.drawer.era5VarPrecipStdIndex',
    // Continuous z-score fallback scale (dry→wet, BrBG); discrete class colours used by the SPI layer
    stops: [[-2, '#8c510a'], [-1, '#dfc27d'], [0, '#f5f5f5'], [1, '#80cdc1'], [2, '#01665e']],
  },
  tempStdIndex: {
    key: 'tempStdIndex', prop: 'sti', unit: 'σ',
    labelKey: 'observatory.drawer.era5VarStdIndex',
    // Continuous z-score fallback scale (cold→hot); discrete class colours used by the STI layer
    stops: [[-2, '#313695'], [-1, '#74add1'], [0, '#10b981'], [1, '#f46d43'], [2, '#7f0000']],
  },
}

export function era5ColorExpression(v: Era5Variable): unknown[] {
  const cfg = ERA5_VARIABLES[v]
  const expr: unknown[] = ['interpolate', ['linear'], ['to-number', ['get', cfg.prop]]]
  for (const [value, color] of cfg.stops) {
    expr.push(value, color)
  }
  return expr
}

/** Returns a CSS linear-gradient string built from the variable's colour stops,
 *  with each stop positioned proportionally across [min, max]. */
export function era5GradientCss(variable: Era5Variable): string {
  const stops = ERA5_VARIABLES[variable].stops
  const values = stops.map(([v]) => v)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min
  const parts = stops.map(([v, c]) => {
    const pct = range === 0 ? 0 : ((v - min) / range) * 100
    return `${c} ${pct.toFixed(1)}%`
  })
  return `linear-gradient(to right, ${parts.join(', ')})`
}

export function era5FormatValue(v: Era5Variable, value: number | null): string {
  if (value == null || Number.isNaN(value)) return '—'
  const cfg = ERA5_VARIABLES[v]
  if (v === 'anomaly') {
    const s = value.toFixed(1)
    return `${value < 0 ? s.replace('-', '−') : `+${s}`} ${cfg.unit}`
  }
  if (v === 'tempStdIndex' || v === 'precipStdIndex') {
    const s = Math.abs(value).toFixed(1)
    return `${value < 0 ? `−${s}` : `+${s}`} ${cfg.unit}`
  }
  if (v === 'waterBalance') {
    const s = Math.abs(Math.round(value)).toString()
    return `${value < 0 ? `−${s}` : `+${s}`} ${cfg.unit}`
  }
  const shown = v === 'evaporation' ? Math.abs(value) : value
  return `${shown.toFixed(1)} ${cfg.unit}`
}

// ---------------------------------------------------------------------------
// STI (Standardized Temperature Index) — discrete 7-class colour map
// Temperature-oriented: cold (indigo) → normal (green) → hot (dark red).
// Inverse of the piezo CLASSIFICATION_COLORS (which use red=low, blue=high).
// ---------------------------------------------------------------------------

/** McKee 7-class colour map keyed by index_class string (temperature-oriented). */
export const STI_CLASS_COLORS: Record<string, string> = {
  EXTREMEMENT_BAS: '#313695',   // indigo — very cold
  TRES_BAS:        '#4575b4',   // medium blue — cold
  BAS:             '#74add1',   // light blue — slightly cold
  NORMAL:          '#10b981',   // green — normal
  HAUT:            '#f46d43',   // orange — slightly hot
  TRES_HAUT:       '#d73027',   // red — hot
  EXTREMEMENT_HAUT:'#7f0000',   // dark red — very hot
  UNKNOWN:         '#6b7280',   // grey — unknown / insufficient data
}

/** Ordered from coldest to hottest (for legends). */
export const STI_CLASS_ORDER = [
  'EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT',
] as const

/** Returns the CSS colour for a given STI index_class string; falls back to grey. */
export function era5StiClassColor(cls: string): string {
  return STI_CLASS_COLORS[cls] ?? STI_CLASS_COLORS['UNKNOWN']
}

// NOTE: The canonical French STI class labels live in i18n under
// `observatory.sti.*` (rendered by Era5Banner / ObservatoryMap). A second,
// divergent label map + formatter used to live here; it was dead code and its
// wording disagreed with the i18n keys, so it was removed to keep one source of
// truth. If a formatted "z σ · <label>" string is ever needed, read the label
// from t('observatory.sti.<class>') at the call site.

/**
 * Classifies a z-score into one of the 7 McKee STI classes.
 * Mirrors the backend thresholds in api/era5_anomaly.py.
 * null/NaN → 'UNKNOWN'.
 */
export function classifyIndex(z: number | null | undefined): string {
  if (z == null || Number.isNaN(z)) return 'UNKNOWN'
  // Backend uses the half-open convention lo <= z < hi (api/era5_anomaly.py), so
  // the warm-side comparators are strict '<' — an exact boundary (e.g. z=0.84)
  // must land in the warmer class to match the per-cell backend classification.
  if (z < -1.75) return 'EXTREMEMENT_BAS'
  if (z < -1.28) return 'TRES_BAS'
  if (z < -0.84) return 'BAS'
  if (z < 0.84) return 'NORMAL'
  if (z < 1.28) return 'HAUT'
  if (z < 1.75) return 'TRES_HAUT'
  return 'EXTREMEMENT_HAUT'
}

/**
 * Build a MapLibre 'match' fill-color expression that maps the 'index_class'
 * feature property (a McKee class string) to its STI colour.
 * Unknown/missing classes fall back to grey.
 */
export function era5StiClassMatchExpr(): unknown[] {
  const expr: unknown[] = ['match', ['get', 'index_class']]
  for (const cls of STI_CLASS_ORDER) {
    expr.push(cls, STI_CLASS_COLORS[cls])
  }
  expr.push(STI_CLASS_COLORS['UNKNOWN']) // fallback
  return expr
}

// ---------------------------------------------------------------------------
// SPI (Standardized Precipitation Index, McKee 1993) — discrete 7-class colour map
// Precipitation-oriented (BrBG diverging): dry (brown) → normal (neutral) → wet (teal).
// Opposite orientation to the STI (which is cold→hot); shares the same McKee classes.
// ---------------------------------------------------------------------------

/** McKee 7-class colour map keyed by index_class string (precipitation-oriented). */
export const SPI_CLASS_COLORS: Record<string, string> = {
  EXTREMEMENT_BAS: '#8c510a',   // dark brown — extreme drought
  TRES_BAS:        '#bf812d',   // brown — severe dry
  BAS:             '#dfc27d',   // tan — moderately dry
  NORMAL:          '#f5f5f5',   // neutral — near normal
  HAUT:            '#80cdc1',   // light teal — moderately wet
  TRES_HAUT:       '#35978f',   // teal — very wet
  EXTREMEMENT_HAUT:'#01665e',   // dark teal — extremely wet
  UNKNOWN:         '#6b7280',   // grey — unknown / insufficient data
}

/** Ordered from driest to wettest (for legends). */
export const SPI_CLASS_ORDER = [
  'EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT',
] as const

/** Returns the CSS colour for a given SPI index_class string; falls back to grey. */
export function era5SpiClassColor(cls: string): string {
  return SPI_CLASS_COLORS[cls] ?? SPI_CLASS_COLORS['UNKNOWN']
}

/** MapLibre 'match' fill-color expression mapping the 'index_class' property to its SPI colour. */
export function era5SpiClassMatchExpr(): unknown[] {
  const expr: unknown[] = ['match', ['get', 'index_class']]
  for (const cls of SPI_CLASS_ORDER) {
    expr.push(cls, SPI_CLASS_COLORS[cls])
  }
  expr.push(SPI_CLASS_COLORS['UNKNOWN']) // fallback
  return expr
}

/**
 * Returns a suitable [min, max] display domain for a raw ERA5 variable,
 * adapted to the temporal granularity (daily vs monthly aggregates).
 *
 * Use this for legend bounds and colour expressions where granularity is known.
 * The domain is wired into the MapLibre interpolate expression (era5-grid-fill
 * layer) via buildRawGridColorExpression, and into the by-zone choropleth via
 * era5ZoneColorExpression, so monthly raw maps don't saturate.
 */
export function era5RawDomain(
  variable: 'temperature' | 'precipitation' | 'evaporation' | 'waterBalance',
  granularity: Era5Granularity,
): [number, number] {
  if (variable === 'precipitation') {
    return granularity === 'monthly' ? [0, 200] : [0, 50]
  }
  if (variable === 'evaporation') {
    return granularity === 'monthly' ? [-100, 0] : [-10, 0]
  }
  if (variable === 'waterBalance') {
    // Symmetric around 0 so the divergent scale stays centred; wider for monthly sums.
    return granularity === 'monthly' ? [-150, 150] : [-15, 15]
  }
  // temperature: same scale covers both daily and monthly means
  return [-10, 35]
}
