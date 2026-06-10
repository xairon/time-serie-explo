// frontend/src/components/meteo/layers/sectors-layer.ts
// BSH sector choropleth + circular trend badges. Pure MapLibre, no React.
import maplibregl from 'maplibre-gl'
import { SECTOR_INSUFFICIENT_COLOR, parseTendancyCoord } from '@/lib/sector-arrows'
import { createRgbaIcon, drawTrendBadge, type TrendKind } from '@/lib/meteo-icons'

export type Trend = 'hausse' | 'stable' | 'baisse' | null

const BADGE_IMAGE: Record<TrendKind, string> = {
  hausse: 'trend-hausse',
  stable: 'trend-stable',
  baisse: 'trend-baisse',
  inconnu: 'trend-inconnu',
}

export function addSectorLayers(
  map: maplibregl.Map,
  geojson: GeoJSON.FeatureCollection,
  onSectorClick: (sectorId: number, name: string) => void,
): void {
  if (map.getSource('secteurs-bsh')) return

  for (const kind of Object.keys(BADGE_IMAGE) as TrendKind[]) {
    if (!map.hasImage(BADGE_IMAGE[kind])) {
      map.addImage(BADGE_IMAGE[kind], createRgbaIcon(drawTrendBadge(kind), 54))
    }
  }

  map.addSource('secteurs-bsh', { type: 'geojson', data: geojson, attribution: 'Secteurs © BRGM / Eaufrance' })
  map.addLayer({
    id: 'secteurs-fill', type: 'fill', source: 'secteurs-bsh',
    paint: { 'fill-color': SECTOR_INSUFFICIENT_COLOR, 'fill-opacity': 0.6 },
  })
  map.addLayer({
    id: 'secteurs-line', type: 'line', source: 'secteurs-bsh',
    paint: { 'line-color': '#475569', 'line-width': 0.8, 'line-opacity': 0.5 },
  })

  map.addSource('secteurs-trends', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
  map.addLayer({
    id: 'secteurs-trends', type: 'symbol', source: 'secteurs-trends',
    layout: {
      'icon-image': ['get', 'icon'],
      'icon-size': 1 / 3,
      'icon-allow-overlap': true,
      'icon-ignore-placement': true,
    },
  })

  map.on('click', 'secteurs-fill', (e) => {
    const f = e.features?.[0]
    if (!f) return
    onSectorClick(f.properties?.sector_id as number, (f.properties?.nom as string) ?? '')
  })
  map.on('mouseenter', 'secteurs-fill', () => { map.getCanvas().style.cursor = 'pointer' })
  map.on('mouseleave', 'secteurs-fill', () => { map.getCanvas().style.cursor = '' })
}

/** Recolor the choropleth from an explicit sector_id → hex map. */
export function updateSectorColors(map: maplibregl.Map, colorById: Record<number, string>): void {
  if (!map.getLayer('secteurs-fill')) return
  const pairs: (number | string)[] = []
  for (const [sid, hex] of Object.entries(colorById)) pairs.push(Number(sid), hex)
  map.setPaintProperty(
    'secteurs-fill', 'fill-color',
    pairs.length
      ? (['match', ['get', 'sector_id'], ...pairs, SECTOR_INSUFFICIENT_COLOR] as unknown as maplibregl.ExpressionSpecification)
      : SECTOR_INSUFFICIENT_COLOR,
  )
}

/** Rebuild trend badge points from sector geometry + a sector_id → trend map. */
export function updateTrendBadges(
  map: maplibregl.Map,
  sectorFeatures: GeoJSON.Feature[],
  trendById: Record<number, Trend>,
): void {
  const src = map.getSource('secteurs-trends') as maplibregl.GeoJSONSource | undefined
  if (!src) return
  const features = sectorFeatures
    .map((f) => {
      const sid = f.properties?.sector_id as number | undefined
      const coords = parseTendancyCoord(f.properties?.tendancy_coord as string | null | undefined)
      if (sid == null || !coords) return null
      const trend = trendById[sid]
      const kind: TrendKind = trend ?? 'inconnu'
      return {
        type: 'Feature' as const,
        geometry: { type: 'Point' as const, coordinates: coords },
        properties: { icon: BADGE_IMAGE[kind] },
      }
    })
    .filter(Boolean) as GeoJSON.Feature[]
  src.setData({ type: 'FeatureCollection', features })
}
