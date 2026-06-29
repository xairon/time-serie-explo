export type Era5Variable = 'temperature' | 'precipitation' | 'evaporation' | 'anomaly'

export interface Era5VarConfig {
  key: Era5Variable
  prop: 'temperature_2m' | 'total_precipitation' | 'potential_evaporation' | 'anomaly_c'
  unit: string
  labelKey: string
  stops: Array<[number, string]>
}

export const ERA5_VARIABLES: Record<Era5Variable, Era5VarConfig> = {
  temperature: {
    key: 'temperature', prop: 'temperature_2m', unit: '°C',
    labelKey: 'observatory.drawer.era5VarTemperature',
    stops: [[-10, '#2166ac'], [0, '#67a9cf'], [10, '#d1e5f0'], [20, '#fddbc7'], [27, '#ef8a62'], [35, '#b2182b']],
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
    key: 'anomaly', prop: 'anomaly_c', unit: '°C',
    labelKey: 'observatory.drawer.era5VarAnomaly',
    stops: [[-5, '#2166ac'], [-2.5, '#67a9cf'], [-0.5, '#d1e5f0'], [0, '#f7f7f7'], [0.5, '#fddbc7'], [2.5, '#ef8a62'], [5, '#b2182b']],
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
  const shown = v === 'evaporation' ? Math.abs(value) : value
  return `${shown.toFixed(1)} ${cfg.unit}`
}
