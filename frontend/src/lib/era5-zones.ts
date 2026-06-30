import { ERA5_VARIABLES, type Era5Variable } from './era5-colors'

function pointInRing(x: number, y: number, ring: number[][]): boolean {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1]
    if (((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)) inside = !inside
  }
  return inside
}

export function pointInPolygonGeometry(lon: number, lat: number, geometry: any): boolean {
  if (!geometry) return false
  const c = geometry.coordinates
  if (geometry.type === 'Polygon') {
    const [outer, ...holes] = c as number[][][]
    if (!pointInRing(lon, lat, outer)) return false
    return !holes.some((h) => pointInRing(lon, lat, h))
  }
  if (geometry.type === 'MultiPolygon') {
    return (c as number[][][][]).some((poly) => {
      const [outer, ...holes] = poly
      if (!pointInRing(lon, lat, outer)) return false
      return !holes.some((h) => pointInRing(lon, lat, h))
    })
  }
  return false
}

export function aggregateEra5ByZone(
  points: Array<Record<string, number | null>>,
  valueKey: string,
  zoneFeatures: Array<{ properties: Record<string, unknown>; geometry: any }>,
  idProp: string,
): Record<string, number> {
  const sums: Record<string, { sum: number; n: number }> = {}
  for (const f of zoneFeatures) {
    const id = String(f.properties?.[idProp])
    for (const p of points) {
      const v = p[valueKey]
      if (v == null) continue
      const lon = Number(p['longitude']), lat = Number(p['latitude'])
      if (pointInPolygonGeometry(lon, lat, f.geometry)) {
        const acc = sums[id] ?? (sums[id] = { sum: 0, n: 0 })
        acc.sum += v; acc.n += 1
      }
    }
  }
  const out: Record<string, number> = {}
  for (const [id, { sum, n }] of Object.entries(sums)) if (n > 0) out[id] = sum / n
  return out
}

function interpColor(value: number, stops: Array<[number, string]>): string {
  if (value <= stops[0][0]) return stops[0][1]
  if (value >= stops[stops.length - 1][0]) return stops[stops.length - 1][1]
  for (let i = 0; i < stops.length - 1; i++) {
    const [v0, c0] = stops[i], [v1, c1] = stops[i + 1]
    if (value >= v0 && value <= v1) {
      const t = (value - v0) / (v1 - v0)
      const a = parseInt(c0.slice(1), 16), b = parseInt(c1.slice(1), 16)
      const r = Math.round(((a >> 16) & 255) + t * (((b >> 16) & 255) - ((a >> 16) & 255)))
      const g = Math.round(((a >> 8) & 255) + t * (((b >> 8) & 255) - ((a >> 8) & 255)))
      const bl = Math.round((a & 255) + t * ((b & 255) - (a & 255)))
      return `rgb(${r},${g},${bl})`
    }
  }
  return stops[stops.length - 1][1]
}

/**
 * Build a MapLibre 'match' colour expression for a by-zone ERA5 choropleth.
 *
 * @param domain - When provided (raw variables), the stop positions are rescaled
 *   from the original [stopsMin, stopsMax] to [domain[0], domain[1]] so that
 *   monthly aggregates (which exceed daily stop ranges) are not saturated.
 *   Omit (or pass undefined) for anomaly variables, which keep their fixed
 *   divergent stops.
 */
export function era5ZoneColorExpression(
  idProp: string,
  zoneValues: Record<string, number>,
  variable: Era5Variable,
  domain?: [number, number],
): unknown[] {
  let stops = ERA5_VARIABLES[variable].stops
  if (domain != null) {
    const [dMin, dMax] = domain
    const sMin = stops[0][0]
    const sMax = stops[stops.length - 1][0]
    const sRange = sMax - sMin
    const dRange = dMax - dMin
    stops = stops.map(([v, c]): [number, string] => [
      sRange === 0 ? dMin : dMin + ((v - sMin) / sRange) * dRange,
      c,
    ])
  }
  const expr: unknown[] = ['match', ['get', idProp]]
  for (const [id, value] of Object.entries(zoneValues)) {
    expr.push(id, interpColor(value, stops))
  }
  expr.push('rgba(0,0,0,0)') // fallback for zones with no data
  return expr
}
