import type { ERA5GridPoint, ERA5AnomalyPoint, ERA5StiPoint } from './observatory-types'

export const ERA5_CELL_HALF = 0.05

export interface ERA5CellProps {
  temperature_2m: number | null
  total_precipitation: number | null
  potential_evaporation: number | null
}

/**
 * Convert generic anomaly points (from /anomaly?variable=...) into GeoJSON squares.
 * Reads the `anomaly` field and stores it as feature property `anomaly`.
 * Used for both the temperature-anomaly and precipitation-anomaly grid layers.
 */
export function era5GenericAnomalyToSquares(
  points: ERA5AnomalyPoint[],
): GeoJSON.FeatureCollection<GeoJSON.Polygon, { anomaly: number }> {
  const h = ERA5_CELL_HALF
  return {
    type: 'FeatureCollection',
    features: points
      .filter((p) => p.anomaly != null)
      .map((p) => {
        const lon = Number(p.longitude)
        const lat = Number(p.latitude)
        const v = p.anomaly as number
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
          properties: { anomaly: v },
        }
      }),
  }
}


/**
 * Convert STI points to GeoJSON squares.
 * Filters points where index_class is null/'UNKNOWN'.
 * Feature properties: index_class (string), sti (number|null).
 */
export function era5StiToSquares(
  points: ERA5StiPoint[],
): GeoJSON.FeatureCollection<GeoJSON.Polygon, { index_class: string; sti: number | null }> {
  const h = ERA5_CELL_HALF
  return {
    type: 'FeatureCollection',
    features: points
      .filter((p) => p.index_class != null && p.index_class !== 'UNKNOWN')
      .map((p) => {
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
          properties: { index_class: p.index_class as string, sti: p.sti },
        }
      }),
  }
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
