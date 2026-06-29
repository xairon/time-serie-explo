export type Era5Variable = 'temperature' | 'precipitation' | 'evaporation'

export interface Era5VarConfig {
  key: Era5Variable
  prop: 'temperature_2m' | 'total_precipitation' | 'potential_evaporation'
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
}

export function era5ColorExpression(v: Era5Variable): unknown[] {
  const cfg = ERA5_VARIABLES[v]
  const expr: unknown[] = ['interpolate', ['linear'], ['to-number', ['get', cfg.prop]]]
  for (const [value, color] of cfg.stops) {
    expr.push(value, color)
  }
  return expr
}

export function era5FormatValue(v: Era5Variable, value: number | null): string {
  if (value == null || Number.isNaN(value)) return '—'
  const cfg = ERA5_VARIABLES[v]
  const shown = v === 'evaporation' ? Math.abs(value) : value
  return `${shown.toFixed(1)} ${cfg.unit}`
}
