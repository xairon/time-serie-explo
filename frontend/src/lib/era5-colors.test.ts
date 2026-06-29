import { describe, it, expect } from 'vitest'
import { ERA5_VARIABLES, era5ColorExpression, era5FormatValue } from './era5-colors'

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
})
