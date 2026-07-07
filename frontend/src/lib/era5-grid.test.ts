import { describe, it, expect } from 'vitest'
import { era5PointsToSquares, ERA5_CELL_HALF, cellCenterFromPolygon } from './era5-grid'

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

  it('carries the raw values plus the derived water balance as feature properties', () => {
    const fc = era5PointsToSquares([
      { latitude: 45, longitude: 5, temperature_2m: 9, total_precipitation: 10, potential_evaporation: -2 },
    ])
    expect(fc.features[0].properties).toEqual({
      // water_balance = P + potential_evaporation (ETP stored negative) = 10 + (-2) = 8
      temperature_2m: 9, total_precipitation: 10, potential_evaporation: -2, water_balance: 8,
    })
  })
})

describe('cellCenterFromPolygon', () => {
  // ObservatoryMap's cell popup (Task C1) uses this to build the "Analyser dans
  // Climat →" link href (/climat?lat=&lon=) from the clicked grid-square feature.
  it('recovers the exact centre of a square built by era5PointsToSquares', () => {
    const fc = era5PointsToSquares([
      { latitude: 47.4, longitude: 0.7, temperature_2m: 12, total_precipitation: 1, potential_evaporation: -1 },
    ])
    const center = cellCenterFromPolygon(fc.features[0].geometry)
    expect(center).not.toBeNull()
    expect(center!.lat).toBeCloseTo(47.4)
    expect(center!.lon).toBeCloseTo(0.7)
  })

  it('returns null for a non-Polygon geometry', () => {
    expect(cellCenterFromPolygon({ type: 'Point', coordinates: [0.7, 47.4] })).toBeNull()
  })

  it('returns null for a Polygon with an empty ring', () => {
    expect(cellCenterFromPolygon({ type: 'Polygon', coordinates: [[]] })).toBeNull()
  })
})
