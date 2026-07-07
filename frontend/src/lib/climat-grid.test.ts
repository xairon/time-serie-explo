import { describe, it, expect } from 'vitest'
import { climatMonthlyToSquares, climatIndicesToSquares } from './climat-grid'
import { ERA5_CELL_HALF } from './era5-grid'

describe('climatMonthlyToSquares', () => {
  it('builds one square per non-null point, centred on lat/lon', () => {
    const fc = climatMonthlyToSquares([
      { latitude: 48, longitude: 2, value: 12.3, mois_complet: true },
    ])
    expect(fc.features).toHaveLength(1)
    const ring = fc.features[0].geometry.coordinates[0]
    expect(ring).toHaveLength(5)
    expect(ring[0]).toEqual([2 - ERA5_CELL_HALF, 48 - ERA5_CELL_HALF])
    expect(fc.features[0].properties).toEqual({ value: 12.3 })
  })

  it('drops points with a null value', () => {
    const fc = climatMonthlyToSquares([
      { latitude: 48, longitude: 2, value: null, mois_complet: false },
    ])
    expect(fc.features).toHaveLength(0)
  })
})

describe('climatIndicesToSquares', () => {
  it('reads the requested index key (spi) and carries index_class', () => {
    const fc = climatIndicesToSquares(
      [{ latitude: 45, longitude: 5, spi: -1.8, index_class: 'TRES_BAS' }],
      'spi',
    )
    expect(fc.features).toHaveLength(1)
    expect(fc.features[0].properties).toEqual({ value: -1.8, index_class: 'TRES_BAS' })
  })

  it('reads the requested index key (sti) independently of a stray spi field', () => {
    const fc = climatIndicesToSquares(
      [{ latitude: 45, longitude: 5, sti: 1.2, spi: null, index_class: 'HAUT' }],
      'sti',
    )
    expect(fc.features[0].properties).toEqual({ value: 1.2, index_class: 'HAUT' })
  })

  it('drops points where the requested index is null', () => {
    const fc = climatIndicesToSquares(
      [{ latitude: 45, longitude: 5, spi: null, index_class: 'UNKNOWN' }],
      'spi',
    )
    expect(fc.features).toHaveLength(0)
  })
})
