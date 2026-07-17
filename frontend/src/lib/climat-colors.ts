// Colour scales for the Climat module (Lot 2) — variable picker on the Situation view.
// SPI/STI reuse the existing 7-class McKee palette from era5-colors.ts (same thresholds,
// same warehouse classification via api/era5_anomaly.py::classify_index — no duplication
// of the class→colour mapping). Le bilan hydrique et les températures journalières
// (Tx/Tn/Tmoy) gardent leurs échelles ici car les endpoints Climat
// (api/routers/observatory_climat.py) renvoient une propriété générique `value`.
import { SPI_CLASS_COLORS, SPI_CLASS_ORDER, STI_CLASS_COLORS, STI_CLASS_ORDER } from './era5-colors'

export type ClimatVariable =
  | 'spi' | 'sti' | 'bilan_hydrique'
  | 'tmax' | 'tmin' | 'tmean' | 'precip_daily'

export type ClimatVariableKind = 'index' | 'raw' | 'daily'

export interface ClimatVarConfig {
  key: ClimatVariable
  kind: ClimatVariableKind
  /** `variable` query param for GET /observatory/climat/grid-monthly (raw vars only). */
  monthlyParam?: 'bilan_hydrique'
  /** `variable` query param for GET /observatory/climat/daily-temp (daily vars only). */
  dailyParam?: 'tmax' | 'tmin' | 'tmean'
  unit: string
  labelKey: string
  /** Gradient colour stops, raw/daily vars only (index vars use the discrete McKee palette). */
  stops: Array<[number, string]>
}

/** Vivid continuous weather-map ramp for absolute daily °C (Tx/Tn/Tmoy) — a
 *  classic heatwave-map look (Météo-France/ECMWF style): deep blue for hard
 *  frost through blue/cyan-green/yellow/orange/red for the ordinary range, dark
 *  red then purple to flag heatwave-record territory. Tn uses the identical ramp
 *  (no separate cold-only scale): night lows self-adapt to the cool half of it. */
export const DAILY_TEMP_STOPS: Array<[number, string]> = [
  [-10, '#1b2c6b'], [0, '#2e6fba'], [10, '#33b6a6'], [20, '#f7e24c'],
  [28, '#f2933d'], [34, '#e23b32'], [38, '#a31e22'], [42, '#7a2d8e'],
]

/** Bornes de classes de la pluie journalière (mm), convention des cartes météo
 *  (Météo-France/ECMWF). FIXES et ABSOLUES : 20 mm c'est 20 mm, en janvier comme
 *  en juillet — deux jours restent comparables. Ne JAMAIS les ré-ancrer sur le
 *  jour affiché : la couleur deviendrait un encodage relatif, c'est-à-dire un
 *  indice maison (cf. spec 2026-07-16, l'erreur commise puis rejetée sur la
 *  température mensuelle).
 *
 *  Pourquoi non linéaires : mesuré sur la grille France, une rampe linéaire 0-50
 *  place 71 % du territoire dans ses 5 premiers % un jour ordinaire (la moitié
 *  des mailles est sous 1 mm). Les classes rendent la carte lisible sans toucher
 *  au domaine. Couvre le 0 -> 98,8 mm réellement observé ; au-delà de 50 mm, la
 *  saturation dans la classe haute EST l'information. */
export const PRECIP_DAILY_BOUNDS: number[] = [0.1, 1, 2, 5, 10, 20, 50]

/** ColorBrewer Blues 8 classes — séquentielle mono-teinte, monotone en luminance,
 *  sûre en déficience de vision des couleurs. La première est quasi blanche : elle
 *  porte le « sec » (< 0,1 mm). */
export const PRECIP_DAILY_COLORS: string[] = [
  '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594',
]

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
  tmax: {
    key: 'tmax', kind: 'daily', dailyParam: 'tmax',
    unit: '°C', labelKey: 'climat.variables.tmax',
    stops: DAILY_TEMP_STOPS,
  },
  tmin: {
    key: 'tmin', kind: 'daily', dailyParam: 'tmin',
    unit: '°C', labelKey: 'climat.variables.tmin',
    stops: DAILY_TEMP_STOPS,
  },
  tmean: {
    key: 'tmean', kind: 'daily', dailyParam: 'tmean',
    unit: '°C', labelKey: 'climat.variables.tmean',
    stops: DAILY_TEMP_STOPS,
  },
  precip_daily: {
    key: 'precip_daily', kind: 'daily',
    unit: 'mm', labelKey: 'climat.variables.precipDaily',
    stops: [],   // classes discrètes, pas de dégradé — cf. climatPrecipDailyColorExpression
  },
}

/** Ordered for the picker's "Données journalières" section (Tx/Tn/Tmoy + pluie) —
 *  kept apart from CLIMAT_VARIABLE_ORDER so the monthly picker stays uncluttered. */
export const DAILY_VARIABLE_ORDER: ClimatVariable[] = ['tmax', 'tmin', 'tmean', 'precip_daily']

export const CLIMAT_WINDOWS = [1, 3, 6, 12] as const
export type ClimatWindow = (typeof CLIMAT_WINDOWS)[number]

export function isClimatIndexVariable(v: ClimatVariable): boolean {
  return CLIMAT_VARIABLES[v].kind === 'index'
}

export function isClimatDailyVariable(v: ClimatVariable): boolean {
  return CLIMAT_VARIABLES[v].kind === 'daily'
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

/** MapLibre 'step' fill-color expression mapping `value` (mm) to the 7 discrete
 *  presentation classes — thresholds aligned EXACTLY with classifyBilan's bands
 *  (climat-scale.ts: <-150 / [-150,-75) / [-75,-20) / [-20,20] / (20,75] / (75,150] / >150)
 *  and colours from SPI_CLASS_COLORS (era5-colors.ts), so the bilan hydrique map matches
 *  the Task 5 legend instead of a continuous gradient. */
export function climatBilanColorExpression(): unknown[] {
  const C = SPI_CLASS_COLORS
  return [
    'step', ['get', 'value'],
    C.EXTREMEMENT_BAS,          // value < -150
    -150, C.TRES_BAS,
    -75, C.BAS,
    -20, C.NORMAL,
    20, C.HAUT,
    75, C.TRES_HAUT,
    150, C.EXTREMEMENT_HAUT,
  ]
}

/** MapLibre 'step' : classe discrète depuis `value` (mm), bornes alignées EXACTEMENT
 *  sur PRECIP_DAILY_BOUNDS pour que la carte et la légende ne puissent pas diverger.
 *  Même mécanisme que climatBilanColorExpression. */
export function climatPrecipDailyColorExpression(): unknown[] {
  const expr: unknown[] = ['step', ['get', 'value'], PRECIP_DAILY_COLORS[0]]
  PRECIP_DAILY_BOUNDS.forEach((b, i) => expr.push(b, PRECIP_DAILY_COLORS[i + 1]))
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
