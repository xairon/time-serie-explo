// GeoJSON square-polygon builders for the Climat module map (mirrors era5-grid.ts,
// but adapted to the generic response shapes of api/routers/observatory_climat.py:
// grid-monthly returns {latitude, longitude, value, mois_complet} and grid-indices
// returns {latitude, longitude, spi|sti, index_class}).
import { ERA5_CELL_HALF, buildSquarePolygon } from './era5-grid'
import type { ClimatMonthlyPoint, ClimatIndexPoint } from './observatory-types'

/** URL of the Climat module Point panel for a given cell, pre-filled via
 *  ?lat&lon (consumed by useSelectedCellParam — 2-decimal fixed, finer than the
 *  0.1° grid). Shared by every "Analyser dans Climat →" link: ObservatoryMap's
 *  cell popup (Task C1) and StationPage's Contexte climatique section (Task C2). */
export function climatDeepLink(lat: number, lon: number): string {
  return `/climat?lat=${lat.toFixed(2)}&lon=${lon.toFixed(2)}`
}

/** Convert grid-monthly points (one raw variable) into squares carrying `value`. Null values are dropped. */
export function climatMonthlyToSquares(
  points: ClimatMonthlyPoint[],
): GeoJSON.FeatureCollection<GeoJSON.Polygon, { value: number }> {
  const h = ERA5_CELL_HALF
  return {
    type: 'FeatureCollection',
    features: points
      .filter((p) => p.value != null)
      .map((p) => {
        const lon = Number(p.longitude)
        const lat = Number(p.latitude)
        return {
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [buildSquarePolygon(lon, lat, h)] },
          properties: { value: p.value as number },
        }
      }),
  }
}

/** Outline square for the cell selected on the Situation map (Task B2 Point panel
 *  highlight) — a single-feature FeatureCollection centred on lat/lon. */
export function climatSelectedCellSquare(lat: number, lon: number): GeoJSON.FeatureCollection<GeoJSON.Polygon> {
  return {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        geometry: { type: 'Polygon', coordinates: [buildSquarePolygon(lon, lat, ERA5_CELL_HALF)] },
        properties: {},
      },
    ],
  }
}

/** Convert grid-indices points (SPI or STI) into squares carrying `value` + `index_class`. */
export function climatIndicesToSquares(
  points: ClimatIndexPoint[],
  index: 'spi' | 'sti',
): GeoJSON.FeatureCollection<GeoJSON.Polygon, { value: number; index_class: string }> {
  const h = ERA5_CELL_HALF
  return {
    type: 'FeatureCollection',
    features: points
      .filter((p) => p[index] != null)
      .map((p) => {
        const lon = Number(p.longitude)
        const lat = Number(p.latitude)
        return {
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [buildSquarePolygon(lon, lat, h)] },
          properties: { value: p[index] as number, index_class: p.index_class },
        }
      }),
  }
}
