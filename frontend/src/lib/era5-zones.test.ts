import { describe, it, expect } from 'vitest'
import { aggregateEra5ByZone, era5ZoneColorExpression, era5StiZoneColorExpression, pointInPolygonGeometry } from './era5-zones'

const square = (cx: number, cy: number) => ({
  type: 'Feature' as const,
  properties: { code: `${cx},${cy}` },
  geometry: { type: 'Polygon' as const, coordinates: [[[cx-1,cy-1],[cx+1,cy-1],[cx+1,cy+1],[cx-1,cy+1],[cx-1,cy-1]]] },
})

describe('era5-zones', () => {
  it('point-in-polygon basic', () => {
    expect(pointInPolygonGeometry(0, 0, square(0,0).geometry)).toBe(true)
    expect(pointInPolygonGeometry(5, 5, square(0,0).geometry)).toBe(false)
  })

  it('averages cell values per zone, skips nulls, omits empty zones', () => {
    const zones = [square(0,0), square(10,10)]
    const points = [
      { latitude: 0, longitude: 0, temperature_2m: 10 },
      { latitude: 0.5, longitude: 0.2, temperature_2m: 20 },
      { latitude: 0.1, longitude: -0.1, temperature_2m: null }, // skipped
      // none in the (10,10) zone
    ]
    const agg = aggregateEra5ByZone(points as any, 'temperature_2m', zones as any, 'code')
    expect(agg['0,0']).toBeCloseTo(15)        // (10+20)/2
    expect('10,10' in agg).toBe(false)        // empty zone omitted
  })

  it('builds a match expression mapping zone id to a colour with transparent fallback', () => {
    const expr = era5ZoneColorExpression('code', { '0,0': 15 }, 'temperature') as any[]
    expect(expr[0]).toBe('match')
    expect(expr[1]).toEqual(['get', 'code'])
    expect(expr).toContain('0,0')
    expect(expr[expr.length - 1]).toBe('rgba(0,0,0,0)') // fallback last
  })

  it('returns a plain transparent colour (never an empty match) when no zone has data', () => {
    // Regression: while ERA5 anomaly/STI data is still loading, zoneValues is empty.
    // An empty ['match', input, fallback] is invalid in MapLibre ("Expected at least
    // 4 arguments") and left the by-zone layer blank. Must be a constant instead.
    expect(era5ZoneColorExpression('code', {}, 'precipAnomaly')).toBe('rgba(0,0,0,0)')
    expect(era5StiZoneColorExpression('code', {})).toBe('rgba(0,0,0,0)')
    // with data, both still build a valid match
    expect(Array.isArray(era5ZoneColorExpression('code', { A: 1 }, 'precipAnomaly'))).toBe(true)
    expect(Array.isArray(era5StiZoneColorExpression('code', { A: 1 }))).toBe(true)
  })

  it('rescales stop positions when domain is provided (raw monthly case)', () => {
    // precipitation stops: [0,5,15,30,50], daily domain [0,50].
    // With monthly domain [0,200] the stops are rescaled to [0,20,60,120,200].
    // A zone mean of 120 mm/month sits at the 4th rescaled stop → should NOT
    // return the saturated darkest colour '#08306b' (which daily would give for 120>50).
    const exprDaily = era5ZoneColorExpression('code', { 'A': 120 }, 'precipitation') as any[]
    const exprMonthly = era5ZoneColorExpression('code', { 'A': 120 }, 'precipitation', [0, 200]) as any[]

    // Both expressions must keep the standard match structure
    expect(exprMonthly[0]).toBe('match')
    expect(exprMonthly[1]).toEqual(['get', 'code'])
    expect(exprMonthly[exprMonthly.length - 1]).toBe('rgba(0,0,0,0)')

    // match expression layout: ['match', ['get', idProp], id, color, ..., fallback]
    // so index 2 = first id ('A'), index 3 = first color
    expect(exprDaily[2]).toBe('A')
    // Without domain: 120 > 50 → saturated darkest blue at index 3
    expect(exprDaily[3]).toBe('#08306b')
    // With domain [0,200]: 120 maps to the 4th stop → mid-dark blue, not saturated
    expect(exprMonthly[3]).not.toBe('#08306b')
  })
})
