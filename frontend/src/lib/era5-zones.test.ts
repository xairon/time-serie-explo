import { describe, it, expect } from 'vitest'
import { aggregateEra5ByZone, era5ZoneColorExpression, pointInPolygonGeometry } from './era5-zones'

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
})
