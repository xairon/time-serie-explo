import { describe, it, expect } from 'vitest'
import { ERA5_CELL_HALF } from './era5-grid'
import { FRANCE_BBOX, projectCellToPixelRect } from './climat-minimap'

const { lonMin, lonMax, latMin, latMax } = FRANCE_BBOX
const lonSpan = lonMax - lonMin
const latSpan = latMax - latMin

describe('projectCellToPixelRect', () => {
  it('maps the bbox south-west corner centre to (0, height) before centring the cell', () => {
    const width = 200
    const height = 200
    const rect = projectCellToPixelRect(latMin, lonMin, width, height)
    const expectedCellW = (2 * ERA5_CELL_HALF * width) / lonSpan
    const expectedCellH = (2 * ERA5_CELL_HALF * height) / latSpan
    expect(rect.x).toBeCloseTo(0 - expectedCellW / 2, 5)
    expect(rect.y).toBeCloseTo(height - expectedCellH / 2, 5)
  })

  it('maps the bbox centre to the middle of the canvas', () => {
    const centerLon = (lonMin + lonMax) / 2
    const centerLat = (latMin + latMax) / 2
    const rect = projectCellToPixelRect(centerLat, centerLon, 200, 200)
    expect(rect.x + rect.w / 2).toBeCloseTo(100, 5)
    expect(rect.y + rect.h / 2).toBeCloseTo(100, 5)
  })

  it('flips latitude so a more northern cell renders with a smaller y (nearer the top)', () => {
    const south = projectCellToPixelRect(latMin + 1, 2, 200, 200)
    const north = projectCellToPixelRect(latMax - 1, 2, 200, 200)
    expect(north.y).toBeLessThan(south.y)
  })

  it('produces a positive, non-degenerate cell size', () => {
    const rect = projectCellToPixelRect(46.5, 2.5, 200, 200)
    expect(rect.w).toBeGreaterThan(0)
    expect(rect.h).toBeGreaterThan(0)
  })

  it('scales cell size proportionally with canvas size', () => {
    const small = projectCellToPixelRect(46.5, 2.5, 100, 100)
    const large = projectCellToPixelRect(46.5, 2.5, 200, 200)
    expect(large.w).toBeCloseTo(small.w * 2, 5)
    expect(large.h).toBeCloseTo(small.h * 2, 5)
  })
})
