export type Era5Variable = 'temperature' | 'precipitation' | 'evaporation' | 'anomaly' | 'precipAnomaly' | 'tempStdIndex'
export type Era5Granularity = 'daily' | 'monthly'

export interface Era5VarConfig {
  key: Era5Variable
  prop: 'temperature_2m' | 'total_precipitation' | 'potential_evaporation' | 'anomaly' | 'sti'
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
  anomaly: {
    key: 'anomaly', prop: 'anomaly', unit: '°C',
    labelKey: 'observatory.drawer.era5VarAnomaly',
    stops: [[-5, '#2166ac'], [-2.5, '#67a9cf'], [-0.5, '#d1e5f0'], [0, '#f7f7f7'], [0.5, '#fddbc7'], [2.5, '#ef8a62'], [5, '#b2182b']],
  },
  precipAnomaly: {
    key: 'precipAnomaly', prop: 'anomaly', unit: '%',
    labelKey: 'observatory.drawer.era5VarPrecipAnomaly',
    // Divergent scale centred on 0: dry = brown/red, wet = teal/blue
    stops: [[-80, '#8c510a'], [-40, '#d8b365'], [-10, '#f6e8c3'], [0, '#f5f5f5'], [10, '#c7eae5'], [40, '#5ab4ac'], [80, '#01665e']],
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
  if (v === 'precipAnomaly') {
    const s = Math.round(value).toString()
    return `${value < 0 ? s.replace('-', '−') : `+${s}`} ${cfg.unit}`
  }
  if (v === 'tempStdIndex') {
    const s = Math.abs(value).toFixed(1)
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

/** French labels for STI classes (temperature-oriented). */
const STI_CLASS_LABELS_FR: Record<string, string> = {
  EXTREMEMENT_BAS:  'Extrêmement froid',
  TRES_BAS:         'Très froid',
  BAS:              'Froid',
  NORMAL:           'Normal',
  HAUT:             'Chaud',
  TRES_HAUT:        'Très chaud',
  EXTREMEMENT_HAUT: 'Extrêmement chaud',
  UNKNOWN:          'Non classé',
}

/** Returns the CSS colour for a given STI index_class string; falls back to grey. */
export function era5StiClassColor(cls: string): string {
  return STI_CLASS_COLORS[cls] ?? STI_CLASS_COLORS['UNKNOWN']
}

/**
 * Formats an STI z-score with its class label.
 * Example: era5FormatSti(2.1, 'TRES_HAUT') → '+2.1 σ · Très chaud'
 * Uses Unicode minus (U+2212) for negative values.
 */
export function era5FormatSti(z: number | null, cls: string | null): string {
  if (z == null || Number.isNaN(z)) return '—'
  const abs = Math.abs(z).toFixed(1)
  const zStr = z < 0 ? `−${abs} σ` : `+${abs} σ`
  const label = STI_CLASS_LABELS_FR[cls ?? 'UNKNOWN'] ?? STI_CLASS_LABELS_FR['UNKNOWN']
  return `${zStr} · ${label}`
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
  variable: 'temperature' | 'precipitation' | 'evaporation',
  granularity: Era5Granularity,
): [number, number] {
  if (variable === 'precipitation') {
    return granularity === 'monthly' ? [0, 200] : [0, 50]
  }
  if (variable === 'evaporation') {
    return granularity === 'monthly' ? [-100, 0] : [-10, 0]
  }
  // temperature: same scale covers both daily and monthly means
  return [-10, 35]
}
