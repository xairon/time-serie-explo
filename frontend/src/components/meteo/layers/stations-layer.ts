// frontend/src/components/meteo/layers/stations-layer.ts
// Piezo/hydro marker layers (unclustered, minzoom-gated). Pure MapLibre, no React.
import maplibregl from 'maplibre-gl'
import type { StationGeoJSONFeature } from '@/lib/observatory-types'
import { METEO_CLASS_COLORS } from '@/lib/meteo-colors'
import { createIcon, drawStationBadge, drawPiezoGlyph, drawHydroGlyph } from '@/lib/meteo-icons'

export type StationType = 'piezo' | 'hydro'

// Markers only appear from this zoom — keeps the national view clean,
// matching the original's behavior ("visibles en zoomant").
const STATIONS_MINZOOM = 7

const MARKER_SIZE: maplibregl.ExpressionSpecification =
  ['interpolate', ['linear'], ['zoom'], 7, 0.5, 12, 0.8]

function classificationColorExpr(): maplibregl.ExpressionSpecification {
  return [
    'match', ['get', 'classification'],
    'EXTREMEMENT_BAS', METEO_CLASS_COLORS.EXTREMEMENT_BAS,
    'TRES_BAS', METEO_CLASS_COLORS.TRES_BAS,
    'BAS', METEO_CLASS_COLORS.BAS,
    'NORMAL', METEO_CLASS_COLORS.NORMAL,
    'HAUT', METEO_CLASS_COLORS.HAUT,
    'TRES_HAUT', METEO_CLASS_COLORS.TRES_HAUT,
    'EXTREMEMENT_HAUT', METEO_CLASS_COLORS.EXTREMEMENT_HAUT,
    METEO_CLASS_COLORS.UNKNOWN,
  ]
}

function toGeoJSON(features: StationGeoJSONFeature[]): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection' as const,
    features: features
      .filter(f => f.geometry.coordinates[0] != null && f.geometry.coordinates[1] != null)
      .map(f => ({
        type: 'Feature' as const,
        geometry: f.geometry,
        properties: {
          code: f.properties.code,
          type: f.properties.type,
          classification: f.properties.classification ?? '',
          commune: f.properties.commune ?? '',
          derniere_mesure: f.properties.derniere_mesure ?? '',
        },
      })),
  }
}

export function addStationLayers(
  map: maplibregl.Map,
  onStationClick: (code: string, type: StationType) => void,
): void {
  if (map.getSource('piezo-stations')) return

  map.addImage('station-badge', createIcon(drawStationBadge, 44), { sdf: true })
  map.addImage('piezo-glyph', createIcon(drawPiezoGlyph, 44))
  map.addImage('hydro-glyph', createIcon(drawHydroGlyph, 44))

  const colorExpr = classificationColorExpr()
  for (const type of ['piezo', 'hydro'] as StationType[]) {
    map.addSource(`${type}-stations`, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
    map.addLayer({
      id: `${type}-badge`, type: 'symbol', source: `${type}-stations`, minzoom: STATIONS_MINZOOM,
      layout: { 'icon-image': 'station-badge', 'icon-size': MARKER_SIZE, 'icon-allow-overlap': true },
      paint: { 'icon-color': colorExpr, 'icon-halo-color': 'rgba(15,23,42,0.45)', 'icon-halo-width': 1.2, 'icon-halo-blur': 1.2 },
    })
    map.addLayer({
      id: `${type}-glyph`, type: 'symbol', source: `${type}-stations`, minzoom: STATIONS_MINZOOM,
      layout: { 'icon-image': `${type}-glyph`, 'icon-size': MARKER_SIZE, 'icon-allow-overlap': true, 'icon-ignore-placement': true },
    })
    map.on('click', `${type}-badge`, (e) => {
      const code = e.features?.[0]?.properties?.code
      if (code) onStationClick(String(code), type)
    })
    map.on('mouseenter', `${type}-badge`, () => { map.getCanvas().style.cursor = 'pointer' })
    map.on('mouseleave', `${type}-badge`, () => { map.getCanvas().style.cursor = '' })
  }
}

export function setStationData(map: maplibregl.Map, type: StationType, features: StationGeoJSONFeature[]): void {
  ;(map.getSource(`${type}-stations`) as maplibregl.GeoJSONSource | undefined)
    ?.setData(toGeoJSON(features))
}

export function setStationVisibility(map: maplibregl.Map, type: StationType, visible: boolean): void {
  const vis = visible ? 'visible' : 'none'
  for (const id of [`${type}-badge`, `${type}-glyph`]) {
    if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis)
  }
}
