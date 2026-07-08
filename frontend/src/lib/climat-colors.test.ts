import { describe, it, expect } from 'vitest'
import {
  CLIMAT_VARIABLES, DAILY_TEMP_STOPS, DAILY_TEMP_VARIABLE_ORDER,
  isClimatDailyVariable, isClimatIndexVariable,
  climatRawColorExpression, climatGradientCss, climatRawDomain, climatFormatValue,
} from './climat-colors'

describe('DAILY_TEMP_STOPS', () => {
  it('has the exact 8 documented °C -> color stops', () => {
    expect(DAILY_TEMP_STOPS).toEqual([
      [-10, '#1b2c6b'],
      [0, '#2e6fba'],
      [10, '#33b6a6'],
      [20, '#f7e24c'],
      [28, '#f2933d'],
      [34, '#e23b32'],
      [38, '#a31e22'],
      [42, '#7a2d8e'],
    ])
  })

  it('is sorted ascending by temperature (required for linear interpolation)', () => {
    const values = DAILY_TEMP_STOPS.map(([v]) => v)
    expect(values).toEqual([...values].sort((a, b) => a - b))
  })

  it('spans from -10°C (deep blue) to 42°C (purple)', () => {
    expect(DAILY_TEMP_STOPS[0]).toEqual([-10, '#1b2c6b'])
    expect(DAILY_TEMP_STOPS[DAILY_TEMP_STOPS.length - 1]).toEqual([42, '#7a2d8e'])
  })
})

describe('tmax/tmin/tmean share the identical daily-temp ramp', () => {
  it('all three point at the same DAILY_TEMP_STOPS array (no separate cold-only scale for Tn)', () => {
    expect(CLIMAT_VARIABLES.tmax.stops).toBe(DAILY_TEMP_STOPS)
    expect(CLIMAT_VARIABLES.tmin.stops).toBe(DAILY_TEMP_STOPS)
    expect(CLIMAT_VARIABLES.tmean.stops).toBe(DAILY_TEMP_STOPS)
  })

  it('are all kind "daily" with the matching dailyParam', () => {
    expect(CLIMAT_VARIABLES.tmax.kind).toBe('daily')
    expect(CLIMAT_VARIABLES.tmax.dailyParam).toBe('tmax')
    expect(CLIMAT_VARIABLES.tmin.dailyParam).toBe('tmin')
    expect(CLIMAT_VARIABLES.tmean.dailyParam).toBe('tmean')
  })
})

describe('DAILY_TEMP_VARIABLE_ORDER', () => {
  it('lists exactly tmax, tmin, tmean', () => {
    expect(DAILY_TEMP_VARIABLE_ORDER).toEqual(['tmax', 'tmin', 'tmean'])
  })
})

describe('isClimatDailyVariable / isClimatIndexVariable', () => {
  it('classifies daily-temp variables as daily, not index', () => {
    expect(isClimatDailyVariable('tmax')).toBe(true)
    expect(isClimatDailyVariable('tmin')).toBe(true)
    expect(isClimatDailyVariable('tmean')).toBe(true)
    expect(isClimatIndexVariable('tmax')).toBe(false)
  })

  it('classifies SPI/STI as index, not daily', () => {
    expect(isClimatDailyVariable('spi')).toBe(false)
    expect(isClimatDailyVariable('sti')).toBe(false)
  })

  it('classifies the monthly raw variables as neither index nor daily', () => {
    expect(isClimatDailyVariable('temperature')).toBe(false)
    expect(isClimatIndexVariable('temperature')).toBe(false)
  })
})

describe('climatRawColorExpression for a daily-temp variable', () => {
  it('builds a MapLibre linear-interpolate expression from the 8 stops', () => {
    const expr = climatRawColorExpression('tmax')
    expect(expr[0]).toBe('interpolate')
    expect(expr[1]).toEqual(['linear'])
    // 2 header entries + 2 per stop (value, color)
    expect(expr.length).toBe(2 + 1 + DAILY_TEMP_STOPS.length * 2)
    expect(expr[3]).toBe(-10)
    expect(expr[4]).toBe('#1b2c6b')
    expect(expr[expr.length - 2]).toBe(42)
    expect(expr[expr.length - 1]).toBe('#7a2d8e')
  })
})

describe('climatGradientCss / climatRawDomain for a daily-temp variable', () => {
  it('domain is [-10, 42]', () => {
    expect(climatRawDomain('tmean')).toEqual([-10, 42])
  })

  it('gradient CSS places the first and last stops at 0% and 100%', () => {
    const css = climatGradientCss('tmin')
    expect(css.startsWith('linear-gradient(to right, ')).toBe(true)
    expect(css).toContain('#1b2c6b 0.0%')
    expect(css).toContain('#7a2d8e 100.0%')
  })
})

describe('climatFormatValue for a daily-temp variable', () => {
  it('formats one decimal with the °C unit', () => {
    expect(climatFormatValue('tmax', 32.47)).toBe('32.5 °C')
  })

  it('returns the placeholder for null/undefined', () => {
    expect(climatFormatValue('tmax', null)).toBe('—')
    expect(climatFormatValue('tmean', undefined)).toBe('—')
  })
})
