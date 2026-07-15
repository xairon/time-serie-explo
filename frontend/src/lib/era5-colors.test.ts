import { describe, it, expect } from 'vitest'
import {
  ERA5_VARIABLES, era5ColorExpression, era5FormatValue, era5GradientCss, era5RawDomain,
  STI_CLASS_COLORS, STI_CLASS_ORDER, era5StiClassColor, classifyIndex, era5StiClassMatchExpr,
  SPI_CLASS_COLORS, SPI_CLASS_ORDER, era5SpiClassColor, era5SpiClassMatchExpr,
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

  it('precipStdIndex (SPI) is in the model with prop=spi, σ unit and a 0-centred divergent scale', () => {
    expect(ERA5_VARIABLES.precipStdIndex.prop).toBe('spi')
    expect(ERA5_VARIABLES.precipStdIndex.unit).toBe('σ')
    const expr = era5ColorExpression('precipStdIndex') as any[]
    expect(expr[2]).toEqual(['to-number', ['get', 'spi']])
    // divergent scale must include a 0 midpoint stop
    const stopValues = expr.slice(3).filter((_: unknown, i: number) => i % 2 === 0)
    expect(stopValues).toContain(0)
  })

  it('formats precipStdIndex (SPI) as a signed z-score with Unicode minus', () => {
    expect(era5FormatValue('precipStdIndex', 2.1)).toBe('+2.1 σ')
    expect(era5FormatValue('precipStdIndex', -1.5)).toBe('−1.5 σ')
    expect(era5FormatValue('precipStdIndex', 0.0)).toBe('+0.0 σ')
    expect(era5FormatValue('precipStdIndex', null)).toBe('—')
  })

  it('era5GradientCss works for precipStdIndex and places the 0-stop at 50% (BrBG dry→wet)', () => {
    const css = era5GradientCss('precipStdIndex')
    expect(css).toMatch(/^linear-gradient\(to right,/)
    // precipStdIndex stops: -2..0..+2 → 0 is at (0-(-2))/(2-(-2))*100 = 50%
    expect(css).toContain('#f5f5f5 50.0%')
    // extreme colours present: brown (dry) and dark teal (wet)
    expect(css).toContain('#8c510a')
    expect(css).toContain('#01665e')
  })

  it('SPI_CLASS_COLORS/ORDER mirror STI (dry→wet, RdBu red→blue) and match expr builds', () => {
    expect(SPI_CLASS_ORDER).toHaveLength(7)
    expect(SPI_CLASS_COLORS.EXTREMEMENT_BAS).toBe('#b2182b')  // extreme drought
    expect(SPI_CLASS_COLORS.EXTREMEMENT_HAUT).toBe('#2166ac') // extreme wet
    expect(era5SpiClassColor('NOT_A_CLASS')).toBe(SPI_CLASS_COLORS.UNKNOWN)
    const expr = era5SpiClassMatchExpr() as any[]
    expect(expr[0]).toBe('match')
    expect(expr[expr.length - 1]).toBe(SPI_CLASS_COLORS.UNKNOWN)
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

  it('STI_CLASS_COLORS cold→hot orientation (RdBu): EXTREMEMENT_BAS is blue, EXTREMEMENT_HAUT is red', () => {
    expect(STI_CLASS_COLORS.EXTREMEMENT_BAS).toBe('#2166ac')
    expect(STI_CLASS_COLORS.TRES_BAS).toBe('#67a9cf')
    expect(STI_CLASS_COLORS.BAS).toBe('#d1e5f0')
    expect(STI_CLASS_COLORS.NORMAL).toBe('#f7f7f7')
    expect(STI_CLASS_COLORS.HAUT).toBe('#fddbc7')
    expect(STI_CLASS_COLORS.TRES_HAUT).toBe('#ef8a62')
    expect(STI_CLASS_COLORS.EXTREMEMENT_HAUT).toBe('#b2182b')
    expect(STI_CLASS_COLORS.UNKNOWN).toBe('#6b7280')
  })

  it('STI_CLASS_ORDER has 7 entries in cold→hot order', () => {
    expect(STI_CLASS_ORDER).toHaveLength(7)
    expect(STI_CLASS_ORDER[0]).toBe('EXTREMEMENT_BAS')
    expect(STI_CLASS_ORDER[6]).toBe('EXTREMEMENT_HAUT')
  })

  it('era5StiClassColor maps known classes and falls back to grey for unknown', () => {
    expect(era5StiClassColor('EXTREMEMENT_BAS')).toBe('#2166ac')
    expect(era5StiClassColor('NORMAL')).toBe('#f7f7f7')
    expect(era5StiClassColor('EXTREMEMENT_HAUT')).toBe('#b2182b')
    expect(era5StiClassColor('UNKNOWN')).toBe('#6b7280')
    // fallback for unrecognised string
    expect(era5StiClassColor('NOT_A_CLASS')).toBe('#6b7280')
    expect(era5StiClassColor('')).toBe('#6b7280')
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

  it('classifyIndex maps z-score to McKee class (boundary values)', () => {
    expect(classifyIndex(null)).toBe('UNKNOWN')
    expect(classifyIndex(NaN)).toBe('UNKNOWN')
    // Boundaries: -1.75, -1.28, -0.84, 0.84, 1.28, 1.75
    expect(classifyIndex(-2.0)).toBe('EXTREMEMENT_BAS')
    expect(classifyIndex(-1.75)).toBe('TRES_BAS')   // -1.75 is NOT < -1.75 → TRES_BAS
    expect(classifyIndex(-1.5)).toBe('TRES_BAS')
    expect(classifyIndex(-1.28)).toBe('BAS')          // -1.28 is NOT < -1.28 → BAS
    expect(classifyIndex(-1.0)).toBe('BAS')
    expect(classifyIndex(-0.84)).toBe('NORMAL')       // -0.84 is NOT < -0.84 → NORMAL
    expect(classifyIndex(0)).toBe('NORMAL')
    // Warm-side boundaries mirror the backend half-open convention (lo <= z < hi):
    // an exact boundary lands in the warmer class.
    expect(classifyIndex(0.83)).toBe('NORMAL')
    expect(classifyIndex(0.84)).toBe('HAUT')          // 0.84 is NOT < 0.84 → HAUT
    expect(classifyIndex(1.28)).toBe('TRES_HAUT')     // 1.28 is NOT < 1.28 → TRES_HAUT
    expect(classifyIndex(1.29)).toBe('TRES_HAUT')
    expect(classifyIndex(1.75)).toBe('EXTREMEMENT_HAUT')  // 1.75 is NOT < 1.75 → EXTREMEMENT_HAUT
    expect(classifyIndex(1.76)).toBe('EXTREMEMENT_HAUT')
    expect(classifyIndex(2.5)).toBe('EXTREMEMENT_HAUT')
  })

  it('era5StiClassMatchExpr builds a match expression for index_class', () => {
    const expr = era5StiClassMatchExpr() as any[]
    expect(expr[0]).toBe('match')
    expect(expr[1]).toEqual(['get', 'index_class'])
    // Should contain all 7 classes as pairs of [class, color]
    const classes = STI_CLASS_ORDER.slice()
    for (const cls of classes) {
      const idx = expr.indexOf(cls)
      expect(idx).toBeGreaterThan(1)
      expect(expr[idx + 1]).toBe(STI_CLASS_COLORS[cls])
    }
    // Last element is the fallback (grey for UNKNOWN)
    expect(expr[expr.length - 1]).toBe(STI_CLASS_COLORS['UNKNOWN'])
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

describe('palettes d\'anomalie cohérentes', () => {
  it('Normal est le même gris neutre pour SPI et STI', () => {
    expect(SPI_CLASS_COLORS.NORMAL).toBe('#f7f7f7')
    expect(STI_CLASS_COLORS.NORMAL).toBe(SPI_CLASS_COLORS.NORMAL)
  })
  it('SPI : sec = rouge, humide = bleu', () => {
    expect(SPI_CLASS_COLORS.EXTREMEMENT_BAS).toBe('#b2182b') // très sec = rouge
    expect(SPI_CLASS_COLORS.EXTREMEMENT_HAUT).toBe('#2166ac') // très humide = bleu
  })
  it('STI : chaud = rouge, froid = bleu (axe inversé, même rouge = préoccupant)', () => {
    expect(STI_CLASS_COLORS.EXTREMEMENT_HAUT).toBe('#b2182b') // très chaud = rouge
    expect(STI_CLASS_COLORS.EXTREMEMENT_BAS).toBe('#2166ac')  // très froid = bleu
  })
})
