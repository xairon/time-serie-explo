import type { ERA5GridPoint } from './observatory-types'

export const ERA5_CELL_HALF = 0.05

export interface ERA5CellProps {
  temperature_2m: number | null
  total_precipitation: number | null
  potential_evaporation: number | null
}

export function era5PointsToSquares(
  points: ERA5GridPoint[],
): GeoJSON.FeatureCollection<GeoJSON.Polygon, ERA5CellProps> {
  const h = ERA5_CELL_HALF
  return {
    type: 'FeatureCollection',
    features: points.map((p) => {
      const lon = Number(p.longitude)
      const lat = Number(p.latitude)
      return {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [[
            [lon - h, lat - h],
            [lon + h, lat - h],
            [lon + h, lat + h],
            [lon - h, lat + h],
            [lon - h, lat - h],
          ]],
        },
        properties: {
          temperature_2m: p.temperature_2m,
          total_precipitation: p.total_precipitation,
          potential_evaporation: p.potential_evaporation,
        },
      }
    }),
  }
}
