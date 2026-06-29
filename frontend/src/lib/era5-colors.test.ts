import { describe, it, expect } from 'vitest'
import { ERA5_VARIABLES, era5ColorExpression, era5FormatValue, era5GradientCss } from './era5-colors'

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
    expect(ERA5_VARIABLES.anomaly.prop).toBe('anomaly_c')
    const expr = era5ColorExpression('anomaly') as any[]
    expect(expr[2]).toEqual(['to-number', ['get', 'anomaly_c']])
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
