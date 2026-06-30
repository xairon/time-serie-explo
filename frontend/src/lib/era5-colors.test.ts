import { describe, it, expect } from 'vitest'
import {
  ERA5_VARIABLES, era5ColorExpression, era5FormatValue, era5GradientCss, era5RawDomain,
  STI_CLASS_COLORS, STI_CLASS_ORDER, era5StiClassColor, era5FormatSti,
} from './era5-colors'

describe('era5-colors', () => {
  it('maps each variable to its data property', () => {
    expect(ERA5_VARIABLES.temperature.prop).toBe('temperature_2m')
    expect(ERA5_VARIABLES.precipitation.prop).toBe('total_precipitation')
    expect(ERA5_VARIABLES.evaporation.prop).toBe('potential_evaporation')
  })

  it('builds an interpolate expression reading the right property', () => {
    const expr = era5ColorExpression('temperature') as any[]
    expect(expr[0]).toBe('interpolate')
    expect(expr[2]).toEqual(['to-number', ['get', 'temperature_2m']])
    // remaining entries are alternating stop/colour pairs
    expect(expr.length).toBeGreaterThan(4)
  })

  it('formats ETP as a positive magnitude and null as a dash', () => {
    expect(era5FormatValue('evaporation', -3.1)).toBe('3.1 mm')
    expect(era5FormatValue('temperature', 12.34)).toBe('12.3 °C')
    expect(era5FormatValue('precipitation', null)).toBe('—')
  })

  it('includes the anomaly variable with a divergent scale', () => {
    expect(ERA5_VARIABLES.anomaly.prop).toBe('anomaly')
    const expr = era5ColorExpression('anomaly') as any[]
    expect(expr[2]).toEqual(['to-number', ['get', 'anomaly']])
    // divergent scale includes a 0 midpoint stop
    const stopValues = expr.slice(3).filter((_, i) => i % 2 === 0)
    expect(stopValues).toContain(0)
  })

  it('formats anomaly with an explicit sign', () => {
    expect(era5FormatValue('anomaly', 2.3)).toBe('+2.3 °C')
    expect(era5FormatValue('anomaly', -1.1)).toBe('−1.1 °C')
    expect(era5FormatValue('anomaly', 0)).toBe('+0.0 °C')
    expect(era5FormatValue('anomaly', null)).toBe('—')
  })

  it('era5GradientCss returns a linear-gradient string containing the first and last stop colours', () => {
    const css = era5GradientCss('temperature')
    expect(css).toMatch(/^linear-gradient\(to right,/)
    const stops = ERA5_VARIABLES.temperature.stops
    expect(css).toContain(stops[0][1])                     // first colour
    expect(css).toContain(stops[stops.length - 1][1])      // last colour
    // first stop should be positioned at 0.0%, last at 100.0%
    expect(css).toContain('0.0%')
    expect(css).toContain('100.0%')
  })

  it('era5GradientCss positions anomaly 0-stop at exactly 50%', () => {
    const css = era5GradientCss('anomaly')
    // anomaly stops: -5 … 0 … +5 → 0 is at (0 - (-5)) / (5 - (-5)) * 100 = 50%
    expect(css).toContain('#f7f7f7 50.0%')
  })

  it('precipAnomaly is in the model with prop=anomaly and a 0-centred divergent scale', () => {
    expect(ERA5_VARIABLES.precipAnomaly.prop).toBe('anomaly')
    expect(ERA5_VARIABLES.precipAnomaly.unit).toBe('%')
    const expr = era5ColorExpression('precipAnomaly') as any[]
    expect(expr[2]).toEqual(['to-number', ['get', 'anomaly']])
    // divergent scale must include a 0 midpoint stop
    const stopValues = expr.slice(3).filter((_: unknown, i: number) => i % 2 === 0)
    expect(stopValues).toContain(0)
  })

  it('formats precipAnomaly as a signed integer percent with Unicode minus', () => {
    expect(era5FormatValue('precipAnomaly', 50)).toBe('+50 %')
    expect(era5FormatValue('precipAnomaly', -40)).toBe('−40 %')
    expect(era5FormatValue('precipAnomaly', 0)).toBe('+0 %')
    expect(era5FormatValue('precipAnomaly', null)).toBe('—')
  })

  it('era5GradientCss works for precipAnomaly and places 0-stop at 50%', () => {
    const css = era5GradientCss('precipAnomaly')
    expect(css).toMatch(/^linear-gradient\(to right,/)
    // precipAnomaly stops: -80..0..+80 → 0 is at (0-(-80))/(80-(-80))*100 = 50%
    expect(css).toContain('#f5f5f5 50.0%')
    // extreme colours present
    expect(css).toContain('#8c510a')
    expect(css).toContain('#01665e')
  })

  it('era5RawDomain returns granularity-aware domains for raw variables', () => {
    expect(era5RawDomain('precipitation', 'daily')).toEqual([0, 50])
    expect(era5RawDomain('precipitation', 'monthly')).toEqual([0, 200])
    // temperature same for both granularities
    expect(era5RawDomain('temperature', 'daily')).toEqual(era5RawDomain('temperature', 'monthly'))
    // evaporation monthly has wider range than daily
    const [evapDailyMin] = era5RawDomain('evaporation', 'daily')
    const [evapMonthlyMin] = era5RawDomain('evaporation', 'monthly')
    expect(evapMonthlyMin).toBeLessThan(evapDailyMin)
  })

  // --- STI (Standardized Temperature Index) ---

  it('STI_CLASS_COLORS has all 7 McKee classes plus UNKNOWN', () => {
    const expected = ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT', 'UNKNOWN']
    for (const cls of expected) {
      expect(STI_CLASS_COLORS).toHaveProperty(cls)
      expect(STI_CLASS_COLORS[cls]).toMatch(/^#[0-9a-fA-F]{6}$/)
    }
    expect(Object.keys(STI_CLASS_COLORS)).toHaveLength(8)
  })

  it('STI_CLASS_COLORS cold→hot orientation: EXTREMEMENT_BAS is indigo, EXTREMEMENT_HAUT is dark red', () => {
    expect(STI_CLASS_COLORS.EXTREMEMENT_BAS).toBe('#313695')
    expect(STI_CLASS_COLORS.TRES_BAS).toBe('#4575b4')
    expect(STI_CLASS_COLORS.BAS).toBe('#74add1')
    expect(STI_CLASS_COLORS.NORMAL).toBe('#10b981')
    expect(STI_CLASS_COLORS.HAUT).toBe('#f46d43')
    expect(STI_CLASS_COLORS.TRES_HAUT).toBe('#d73027')
    expect(STI_CLASS_COLORS.EXTREMEMENT_HAUT).toBe('#7f0000')
    expect(STI_CLASS_COLORS.UNKNOWN).toBe('#6b7280')
  })

  it('STI_CLASS_ORDER has 7 entries in cold→hot order', () => {
    expect(STI_CLASS_ORDER).toHaveLength(7)
    expect(STI_CLASS_ORDER[0]).toBe('EXTREMEMENT_BAS')
    expect(STI_CLASS_ORDER[6]).toBe('EXTREMEMENT_HAUT')
  })

  it('era5StiClassColor maps known classes and falls back to grey for unknown', () => {
    expect(era5StiClassColor('EXTREMEMENT_BAS')).toBe('#313695')
    expect(era5StiClassColor('NORMAL')).toBe('#10b981')
    expect(era5StiClassColor('EXTREMEMENT_HAUT')).toBe('#7f0000')
    expect(era5StiClassColor('UNKNOWN')).toBe('#6b7280')
    // fallback for unrecognised string
    expect(era5StiClassColor('NOT_A_CLASS')).toBe('#6b7280')
    expect(era5StiClassColor('')).toBe('#6b7280')
  })

  it('era5FormatSti formats z-score with sign and class label', () => {
    expect(era5FormatSti(2.1, 'TRES_HAUT')).toBe('+2.1 σ · Très chaud')
    expect(era5FormatSti(-1.5, 'TRES_BAS')).toBe('−1.5 σ · Très froid')
    expect(era5FormatSti(0.0, 'NORMAL')).toBe('+0.0 σ · Normal')
    expect(era5FormatSti(null, null)).toBe('—')
  })

  it('tempStdIndex is present in ERA5_VARIABLES with correct prop and unit', () => {
    expect(ERA5_VARIABLES).toHaveProperty('tempStdIndex')
    expect(ERA5_VARIABLES.tempStdIndex.prop).toBe('sti')
    expect(ERA5_VARIABLES.tempStdIndex.unit).toBe('σ')
    expect(ERA5_VARIABLES.tempStdIndex.labelKey).toBe('observatory.drawer.era5VarStdIndex')
  })

  it('era5FormatValue formats tempStdIndex as signed z-score', () => {
    expect(era5FormatValue('tempStdIndex', 2.1)).toBe('+2.1 σ')
    expect(era5FormatValue('tempStdIndex', -1.5)).toBe('−1.5 σ')
    expect(era5FormatValue('tempStdIndex', 0.0)).toBe('+0.0 σ')
    expect(era5FormatValue('tempStdIndex', null)).toBe('—')
  })

  it('has strictly increasing stop values for every ERA5 variable (MapLibre requires monotonic stops)', () => {
    for (const key of Object.keys(ERA5_VARIABLES) as (keyof typeof ERA5_VARIABLES)[]) {
      const expr = era5ColorExpression(key) as unknown[]
      // expr = ['interpolate', ['linear'], ['to-number', ['get', prop]], stop0, color0, stop1, color1, ...]
      // numeric stops start at index 3 (every other entry)
      const stops: number[] = []
      for (let i = 3; i < expr.length; i += 2) {
        stops.push(expr[i] as number)
      }
      for (let i = 1; i < stops.length; i++) {
        expect(stops[i]).toBeGreaterThan(stops[i - 1])
      }
    }
  })
})
