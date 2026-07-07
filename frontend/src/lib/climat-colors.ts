// Colour scales for the Climat module (Lot 2) — variable picker on the Situation view.
// SPI/STI reuse the existing 7-class McKee palette from era5-colors.ts (same thresholds,
// same warehouse classification via api/era5_anomaly.py::classify_index — no duplication
// of the class→colour mapping). The raw variables (temperature/precipitation/etp/bilan
// hydrique) get their own monthly-scaled gradients here because the Climat endpoints
// (api/routers/observatory_climat.py) return a generic `value` property, not the
// per-field shape used by the Observatory overlay's ERA5GridPoint.
import { SPI_CLASS_COLORS, SPI_CLASS_ORDER, STI_CLASS_COLORS, STI_CLASS_ORDER } from './era5-colors'

export type ClimatVariable = 'spi' | 'sti' | 'bilan_hydrique' | 'precipitation' | 'temperature' | 'etp'

export type ClimatVariableKind = 'index' | 'raw'

export interface ClimatVarConfig {
  key: ClimatVariable
  kind: ClimatVariableKind
  /** `variable` query param for GET /observatory/climat/grid-monthly (raw vars only). */
  monthlyParam?: 'temperature' | 'precipitation' | 'etp' | 'bilan_hydrique'
  unit: string
  labelKey: string
  /** Gradient colour stops, raw vars only (index vars use the discrete McKee palette). */
  stops: Array<[number, string]>
}

export const CLIMAT_VARIABLES: Record<ClimatVariable, ClimatVarConfig> = {
  spi: {
    key: 'spi', kind: 'index',
    unit: 'σ', labelKey: 'climat.variables.spi',
    stops: [],
  },
  sti: {
    key: 'sti', kind: 'index',
    unit: 'σ', labelKey: 'climat.variables.sti',
    stops: [],
  },
  bilan_hydrique: {
    key: 'bilan_hydrique', kind: 'raw', monthlyParam: 'bilan_hydrique',
    unit: 'mm', labelKey: 'climat.variables.bilanHydrique',
    // Climatic water balance P − ETP (mm/month). Divergent: deficit = red/brown, surplus = blue.
    stops: [[-150, '#b2182b'], [-75, '#ef8a62'], [-20, '#fddbc7'], [0, '#f7f7f7'], [20, '#d1e5f0'], [75, '#67a9cf'], [150, '#2166ac']],
  },
  precipitation: {
    key: 'precipitation', kind: 'raw', monthlyParam: 'precipitation',
    unit: 'mm', labelKey: 'climat.variables.precipitation',
    stops: [[0, '#f7fbff'], [20, '#c6dbef'], [60, '#6baed6'], [120, '#2171b5'], [200, '#08306b']],
  },
  temperature: {
    key: 'temperature', kind: 'raw', monthlyParam: 'temperature',
    unit: '°C', labelKey: 'climat.variables.temperature',
    stops: [[-10, '#3b2d8c'], [-5, '#3d6fd0'], [0, '#4aa3e0'], [5, '#7fd0e8'], [10, '#9fdfa8'], [15, '#e6e36a'], [20, '#f4b942'], [25, '#ef7d2f'], [30, '#df3b2c'], [35, '#c01f8a']],
  },
  etp: {
    key: 'etp', kind: 'raw', monthlyParam: 'etp',
    unit: 'mm', labelKey: 'climat.variables.etp',
    stops: [[0, '#fff5eb'], [30, '#fdbe85'], [60, '#fd8d3c'], [100, '#e6550d'], [150, '#a63603']],
  },
}

/** Ordered for the picker UI: SPI first (default, per plan), then STI, then the raw variables. */
export const CLIMAT_VARIABLE_ORDER: ClimatVariable[] = [
  'spi', 'sti', 'bilan_hydrique', 'precipitation', 'temperature', 'etp',
]

export const CLIMAT_WINDOWS = [1, 3, 6, 12] as const
export type ClimatWindow = (typeof CLIMAT_WINDOWS)[number]

export function isClimatIndexVariable(v: ClimatVariable): boolean {
  return CLIMAT_VARIABLES[v].kind === 'index'
}

/** MapLibre 'interpolate' fill-color expression reading the generic `value` property. */
export function climatRawColorExpression(variable: ClimatVariable): unknown[] {
  const cfg = CLIMAT_VARIABLES[variable]
  const expr: unknown[] = ['interpolate', ['linear'], ['to-number', ['get', 'value']]]
  for (const [value, color] of cfg.stops) expr.push(value, color)
  return expr
}

/** MapLibre 'match' fill-color expression mapping `index_class` to the SPI/STI palette. */
export function climatIndexColorExpression(variable: 'spi' | 'sti'): unknown[] {
  const order = variable === 'spi' ? SPI_CLASS_ORDER : STI_CLASS_ORDER
  const colors = variable === 'spi' ? SPI_CLASS_COLORS : STI_CLASS_COLORS
  const expr: unknown[] = ['match', ['get', 'index_class']]
  for (const cls of order) expr.push(cls, colors[cls])
  expr.push(colors.UNKNOWN)
  return expr
}

/** CSS linear-gradient for the raw-variable legend (stops positioned proportionally over [min, max]). */
export function climatGradientCss(variable: ClimatVariable): string {
  const stops = CLIMAT_VARIABLES[variable].stops
  const values = stops.map(([v]) => v)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min
  const parts = stops.map(([v, c]) => `${c} ${(range === 0 ? 0 : ((v - min) / range) * 100).toFixed(1)}%`)
  return `linear-gradient(to right, ${parts.join(', ')})`
}

/** [min, max] domain of the raw-variable gradient (for legend bounds). */
export function climatRawDomain(variable: ClimatVariable): [number, number] {
  const stops = CLIMAT_VARIABLES[variable].stops
  return [stops[0][0], stops[stops.length - 1][0]]
}

export function climatFormatValue(variable: ClimatVariable, value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  const cfg = CLIMAT_VARIABLES[variable]
  if (variable === 'spi' || variable === 'sti') {
    const s = Math.abs(value).toFixed(1)
    return `${value < 0 ? `−${s}` : `+${s}`} ${cfg.unit}`
  }
  if (variable === 'bilan_hydrique') {
    const s = Math.abs(Math.round(value)).toString()
    return `${value < 0 ? `−${s}` : `+${s}`} ${cfg.unit}`
  }
  return `${value.toFixed(1)} ${cfg.unit}`
}
