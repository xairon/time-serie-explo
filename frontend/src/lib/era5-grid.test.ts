import { describe, it, expect } from 'vitest'
import { era5PointsToSquares, era5AnomalyPointsToSquares, ERA5_CELL_HALF } from './era5-grid'

describe('era5PointsToSquares', () => {
  it('builds one square polygon per point, centred on lat/lon', () => {
    const fc = era5PointsToSquares([
      { latitude: 48, longitude: 2, temperature_2m: 12.3, total_precipitation: 4, potential_evaporation: -3.1 },
    ])
    expect(fc.type).toBe('FeatureCollection')
    expect(fc.features).toHaveLength(1)
    const f = fc.features[0]
    expect(f.geometry.type).toBe('Polygon')
    // ring is closed (5 coords) and spans centre ± half in both axes
    const ring = f.geometry.coordinates[0]
    expect(ring).toHaveLength(5)
    expect(ring[0]).toEqual([2 - ERA5_CELL_HALF, 48 - ERA5_CELL_HALF])
    expect(ring[4]).toEqual(ring[0])
    const lons = ring.map(c => c[0])
    const lats = ring.map(c => c[1])
    expect(Math.min(...lons)).toBeCloseTo(2 - ERA5_CELL_HALF)
    expect(Math.max(...lons)).toBeCloseTo(2 + ERA5_CELL_HALF)
    expect(Math.min(...lats)).toBeCloseTo(48 - ERA5_CELL_HALF)
    expect(Math.max(...lats)).toBeCloseTo(48 + ERA5_CELL_HALF)
  })

  it('carries the three values as feature properties', () => {
    const fc = era5PointsToSquares([
      { latitude: 45, longitude: 5, temperature_2m: 9, total_precipitation: 0, potential_evaporation: -2 },
    ])
    expect(fc.features[0].properties).toEqual({
      temperature_2m: 9, total_precipitation: 0, potential_evaporation: -2,
    })
  })
})

describe('era5AnomalyPointsToSquares', () => {
  it('builds one polygon per non-null point, centred ±ERA5_CELL_HALF with a closed ring of 5 coords', () => {
    const fc = era5AnomalyPointsToSquares([
      { latitude: 48, longitude: 2, anomaly_c: 1.5 },
    ])
    expect(fc.type).toBe('FeatureCollection')
    expect(fc.features).toHaveLength(1)
    const f = fc.features[0]
    expect(f.geometry.type).toBe('Polygon')
    const ring = f.geometry.coordinates[0]
    expect(ring).toHaveLength(5)
    // ring is closed: first and last coord are equal
    expect(ring[4]).toEqual(ring[0])
    // spans centre ± half
    const lons = ring.map((c: number[]) => c[0])
    const lats = ring.map((c: number[]) => c[1])
    expect(Math.min(...lons)).toBeCloseTo(2 - ERA5_CELL_HALF)
    expect(Math.max(...lons)).toBeCloseTo(2 + ERA5_CELL_HALF)
    expect(Math.min(...lats)).toBeCloseTo(48 - ERA5_CELL_HALF)
    expect(Math.max(...lats)).toBeCloseTo(48 + ERA5_CELL_HALF)
  })

  it('carries anomaly_c in feature properties', () => {
    const fc = era5AnomalyPointsToSquares([
      { latitude: 45, longitude: 5, anomaly_c: -2.3 },
    ])
    expect(fc.features[0].properties).toEqual({ anomaly_c: -2.3 })
  })

  it('filters out points with anomaly_c === null', () => {
    const fc = era5AnomalyPointsToSquares([
      { latitude: 48, longitude: 2, anomaly_c: 1.0 },
      { latitude: 49, longitude: 3, anomaly_c: null },
    ])
    expect(fc.features).toHaveLength(1)
    expect(fc.features[0].properties.anomaly_c).toBe(1.0)
  })
})
