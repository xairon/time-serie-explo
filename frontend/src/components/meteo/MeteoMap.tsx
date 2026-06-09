import { useRef, useEffect, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { StationGeoJSONFeature } from '@/lib/observatory-types'
import { METEO_CLASS_COLORS } from '@/lib/meteo-colors'
import { SECTOR_INSUFFICIENT_COLOR, parseTendancyCoord } from '@/lib/sector-arrows'

const FRANCE_CENTER: [number, number] = [2.5, 46.5]
const FRANCE_ZOOM = 6

const TREND_ROTATION: Record<string, number> = { hausse: 0, stable: 90, baisse: 180 }

// Shared icon-size ramp so badge + glyph scale together.
const MARKER_SIZE: maplibregl.ExpressionSpecification = ['interpolate', ['linear'], ['zoom'], 4, 0.4, 8, 0.55, 12, 0.8]

// BRGM palette: tint station badges by their water-level classification.
// Uses METEO_CLASS_COLORS (the 7 enums + UNKNOWN), NOT the observatory palette.
function buildClassificationColorExpression(): maplibregl.ExpressionSpecification {
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

// ---- Canvas icon helpers (copied from ObservatoryMap, not imported) ----

function createSdfIcon(draw: (ctx: CanvasRenderingContext2D, size: number) => void, size = 44): ImageData {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  draw(ctx, size)
  return ctx.getImageData(0, 0, size, size)
}

// Non-SDF icon (keeps its own colors, e.g. a white glyph that must NOT be
// recolored by icon-color). Used for the type glyph drawn on top of the badge.
function createRgbaIcon(draw: (ctx: CanvasRenderingContext2D, size: number) => void, size = 44): ImageData {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  draw(ctx, size)
  return ctx.getImageData(0, 0, size, size)
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

// Unified squircle badge (SDF, tinted at runtime by classification color).
function drawStationBadge(ctx: CanvasRenderingContext2D, size: number) {
  const pad = size * 0.16
  const w = size - pad * 2
  roundRect(ctx, pad, pad, w, w, w * 0.36)
  ctx.fillStyle = '#fff'
  ctx.fill()
}

// White type glyph — piezo = groundwater: a borehole stem + downward triangle.
function drawPiezoGlyph(ctx: CanvasRenderingContext2D, size: number) {
  const cx = size / 2
  ctx.fillStyle = '#fff'
  ctx.fillRect(cx - size * 0.045, size * 0.30, size * 0.09, size * 0.13)
  const w = size * 0.16, top = size * 0.44, bot = size * 0.66
  ctx.beginPath()
  ctx.moveTo(cx - w, top)
  ctx.lineTo(cx + w, top)
  ctx.lineTo(cx, bot)
  ctx.closePath()
  ctx.fill()
}

// White type glyph — hydro = surface water: a water drop.
function drawHydroGlyph(ctx: CanvasRenderingContext2D, size: number) {
  const cx = size / 2
  const r = size * 0.15
  const bottomY = size * 0.58
  const tipY = size * 0.34
  ctx.beginPath()
  ctx.arc(cx, bottomY, r, 0.12 * Math.PI, 0.88 * Math.PI)
  ctx.lineTo(cx, tipY)
  ctx.closePath()
  ctx.fillStyle = '#fff'
  ctx.fill()
}

// White type glyph — rain = pluviomètre: a raindrop (same teardrop shape as hydro,
// slightly larger so it reads as a falling drop).
function drawRainGlyph(ctx: CanvasRenderingContext2D, size: number) {
  const cx = size / 2
  const r = size * 0.155
  const bottomY = size * 0.60
  const tipY = size * 0.32
  ctx.beginPath()
  ctx.arc(cx, bottomY, r, 0.10 * Math.PI, 0.90 * Math.PI)
  ctx.lineTo(cx, tipY)
  ctx.closePath()
  ctx.fillStyle = '#fff'
  ctx.fill()
}

// Trend arrow icon (points UP by default; rotated per sector via icon-rotate:
// hausse=0°, stable=90°, baisse=180°). Dark fill + white outline.
function drawTrendArrow(ctx: CanvasRenderingContext2D, size: number) {
  const cx = size / 2
  ctx.beginPath()
  ctx.moveTo(cx, size * 0.16)
  ctx.lineTo(size * 0.80, size * 0.74)
  ctx.lineTo(size * 0.20, size * 0.74)
  ctx.closePath()
  ctx.strokeStyle = '#ffffff'
  ctx.lineWidth = size * 0.12
  ctx.lineJoin = 'round'
  ctx.stroke()
  ctx.fillStyle = '#0f172a'
  ctx.fill()
}

// ---- GeoJSON helpers ----

function stationsToGeoJSON(features: StationGeoJSONFeature[]) {
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

interface MeteoMapProps {
  sectorColorById: Record<number, string>
  sectorTrendById: Record<number, 'hausse' | 'stable' | 'baisse' | null>
  alertSectorIds: number[]
  focusSectorId?: number | null
  visibleLayers: { bsn: boolean; piezo: boolean; rain: boolean; hydro: boolean }
  piezoFeatures: StationGeoJSONFeature[]
  hydroFeatures: StationGeoJSONFeature[]
  onSectorClick: (sectorId: number, name: string) => void
  onStationClick: (code: string, type: 'piezo' | 'hydro') => void
}

export function MeteoMap({
  sectorColorById,
  sectorTrendById,
  alertSectorIds,
  focusSectorId,
  visibleLayers,
  piezoFeatures,
  hydroFeatures,
  onSectorClick,
  onStationClick,
}: MeteoMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const mapLoadedRef = useRef(false)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [sectorsReady, setSectorsReady] = useState(false)

  // Callback props stored in refs so map event handlers always call the latest.
  const onSectorClickRef = useRef(onSectorClick); onSectorClickRef.current = onSectorClick
  const onStationClickRef = useRef(onStationClick); onStationClickRef.current = onStationClick

  // Feature refs so the init effect can seed sources without re-running.
  const piezoFeaturesRef = useRef<StationGeoJSONFeature[]>(piezoFeatures); piezoFeaturesRef.current = piezoFeatures
  const hydroFeaturesRef = useRef<StationGeoJSONFeature[]>(hydroFeatures); hydroFeaturesRef.current = hydroFeatures

  // Loaded sector geometry — kept so arrows can be rebuilt from each feature's
  // own sector_id + tendancy_coord against the active trend map.
  const sectorGeoRef = useRef<GeoJSON.FeatureCollection | null>(null)

  // Init map (once)
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          },
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
      },
      center: FRANCE_CENTER,
      zoom: FRANCE_ZOOM,
      maxBounds: [[-12, 38], [18, 54]],
      attributionControl: false,
    })

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    map.addControl(new maplibregl.FullscreenControl(), 'bottom-right')
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 100, unit: 'metric' }), 'bottom-right')
    map.addControl(new maplibregl.AttributionControl({ compact: false }), 'bottom-left')
    map.on('error', (e) => { console.error('MapLibre error:', e.error?.message ?? e) })

    map.on('load', () => {
      mapLoadedRef.current = true

      // Register marker images.
      map.addImage('piezo-marker', createSdfIcon(drawStationBadge, 44), { sdf: true })
      map.addImage('hydro-marker', createSdfIcon(drawStationBadge, 44), { sdf: true })
      map.addImage('rain-marker', createSdfIcon(drawStationBadge, 44), { sdf: true })
      map.addImage('piezo-glyph', createRgbaIcon(drawPiezoGlyph, 44))
      map.addImage('hydro-glyph', createRgbaIcon(drawHydroGlyph, 44))
      map.addImage('rain-glyph', createRgbaIcon(drawRainGlyph, 44))

      const classificationColorExpr = buildClassificationColorExpression()

      // Three independent UNCLUSTERED station sources/layers (NO clustering).
      const addUnclustered = (
        sourceId: string, badgeLayer: string, glyphLayer: string, badgeImg: string, glyphImg: string,
      ) => {
        map.addSource(sourceId, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
        map.addLayer({
          id: badgeLayer, type: 'symbol', source: sourceId,
          layout: { 'icon-image': badgeImg, 'icon-size': MARKER_SIZE, 'icon-allow-overlap': true, visibility: 'none' },
          paint: { 'icon-color': classificationColorExpr, 'icon-halo-color': 'rgba(2,6,23,0.5)', 'icon-halo-width': 1.4, 'icon-halo-blur': 1.6 },
        })
        map.addLayer({
          id: glyphLayer, type: 'symbol', source: sourceId,
          layout: { 'icon-image': glyphImg, 'icon-size': MARKER_SIZE, 'icon-allow-overlap': true, 'icon-ignore-placement': true, visibility: 'none' },
        })
      }
      addUnclustered('piezo-stations', 'piezo-layer', 'piezo-glyph-layer', 'piezo-marker', 'piezo-glyph')
      addUnclustered('hydro-stations', 'hydro-layer', 'hydro-glyph-layer', 'hydro-marker', 'hydro-glyph')
      addUnclustered('rain-stations', 'rain-layer', 'rain-glyph-layer', 'rain-marker', 'rain-glyph')

      // Seed station sources if features arrived before load.
      if (piezoFeaturesRef.current.length) {
        (map.getSource('piezo-stations') as maplibregl.GeoJSONSource | undefined)?.setData(stationsToGeoJSON(piezoFeaturesRef.current) as any)
      }
      if (hydroFeaturesRef.current.length) {
        (map.getSource('hydro-stations') as maplibregl.GeoJSONSource | undefined)?.setData(stationsToGeoJSON(hydroFeaturesRef.current) as any)
      }

      // Station click handlers (badge layers only).
      map.on('click', 'piezo-layer', (e) => { const c = e.features?.[0]?.properties?.code; if (c) onStationClickRef.current?.(String(c), 'piezo') })
      map.on('click', 'hydro-layer', (e) => { const c = e.features?.[0]?.properties?.code; if (c) onStationClickRef.current?.(String(c), 'hydro') })
      for (const layer of ['piezo-layer', 'hydro-layer', 'rain-layer']) {
        map.on('mouseenter', layer, () => { map.getCanvas().style.cursor = 'pointer' })
        map.on('mouseleave', layer, () => { map.getCanvas().style.cursor = '' })
      }

      // Sector layers (added once after fetch).
      fetch('/geo/secteurs-bsh.geojson').then(r => r.json()).then((gj) => {
        if (!mapRef.current || map.getSource('secteurs-bsh')) return
        sectorGeoRef.current = gj
        map.addSource('secteurs-bsh', { type: 'geojson', data: gj, attribution: 'Secteurs © BRGM / Eaufrance' })
        map.addLayer({ id: 'secteurs-fill', type: 'fill', source: 'secteurs-bsh', layout: { visibility: 'none' }, paint: { 'fill-color': SECTOR_INSUFFICIENT_COLOR, 'fill-opacity': 0.7 } })
        map.addLayer({ id: 'secteurs-line', type: 'line', source: 'secteurs-bsh', layout: { visibility: 'none' }, paint: { 'line-color': '#ffffff', 'line-width': 0.6 } })
        map.addLayer({ id: 'secteurs-alert-line', type: 'line', source: 'secteurs-bsh',
          layout: { visibility: 'none', 'line-join': 'round' },
          paint: { 'line-color': '#7f1d1d', 'line-width': 2.4, 'line-opacity': 0.9 },
          filter: ['in', ['get', 'sector_id'], ['literal', []]] as unknown as maplibregl.ExpressionSpecification })
        if (!map.hasImage('sector-arrow')) map.addImage('sector-arrow', createRgbaIcon(drawTrendArrow, 40))
        map.addSource('secteurs-arrows', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
        map.addLayer({
          id: 'secteurs-arrows', type: 'symbol', source: 'secteurs-arrows',
          layout: { visibility: 'none', 'icon-image': 'sector-arrow', 'icon-size': 0.7, 'icon-rotate': ['get', 'rot'], 'icon-rotation-alignment': 'map', 'icon-allow-overlap': true, 'icon-ignore-placement': true },
        })
        map.on('click', 'secteurs-fill', (e) => {
          const f = e.features?.[0]; if (!f) return
          onSectorClickRef.current?.(f.properties?.sector_id as number, (f.properties?.nom as string) ?? '')
        })
        map.on('mouseenter', 'secteurs-fill', () => { map.getCanvas().style.cursor = 'pointer' })
        map.on('mouseleave', 'secteurs-fill', () => { map.getCanvas().style.cursor = '' })
        setSectorsReady(true)
      }).catch(err => console.error('Failed to load sector geometry:', err))

      setMapLoaded(true)
    })

    mapRef.current = map
    return () => { map.remove(); mapRef.current = null; mapLoadedRef.current = false }
  }, [])

  // Recolor sector choropleth + rebuild arrows when the active source's
  // color/trend maps change. Fill is driven by an explicit sector_id → hex map
  // (source-agnostic); arrows come from each geometry feature's own
  // sector_id + tendancy_coord against the trend map.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoadedRef.current || !m.getLayer('secteurs-fill')) return

    const pairs: (number | string)[] = []
    for (const [sid, hex] of Object.entries(sectorColorById)) {
      pairs.push(Number(sid), hex)
    }
    m.setPaintProperty('secteurs-fill', 'fill-color',
      pairs.length
        ? (['match', ['get', 'sector_id'], ...pairs, SECTOR_INSUFFICIENT_COLOR] as unknown as maplibregl.ExpressionSpecification)
        : SECTOR_INSUFFICIENT_COLOR)

    const arrowFeatures = (sectorGeoRef.current?.features ?? [])
      .map((f) => {
        const sid = f.properties?.sector_id as number | undefined
        if (sid == null) return null
        const trend = sectorTrendById[sid]
        const rot = trend != null ? TREND_ROTATION[trend] : undefined
        const c = parseTendancyCoord(f.properties?.tendancy_coord as string | null | undefined)
        if (!c || rot === undefined) return null
        return { type: 'Feature' as const, geometry: { type: 'Point' as const, coordinates: c }, properties: { rot } }
      })
      .filter(Boolean) as GeoJSON.Feature[]
    ;(m.getSource('secteurs-arrows') as maplibregl.GeoJSONSource | undefined)?.setData({ type: 'FeatureCollection', features: arrowFeatures })

    if (m.getLayer('secteurs-alert-line')) {
      m.setFilter('secteurs-alert-line', ['in', ['get', 'sector_id'], ['literal', alertSectorIds]] as unknown as maplibregl.ExpressionSpecification)
    }
  }, [sectorColorById, sectorTrendById, alertSectorIds, mapLoaded, sectorsReady])

  // Sync piezo features.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoadedRef.current) return
    ;(m.getSource('piezo-stations') as maplibregl.GeoJSONSource | undefined)?.setData(stationsToGeoJSON(piezoFeatures ?? []) as any)
  }, [piezoFeatures, mapLoaded])

  // Sync hydro features.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoadedRef.current) return
    ;(m.getSource('hydro-stations') as maplibregl.GeoJSONSource | undefined)?.setData(stationsToGeoJSON(hydroFeatures ?? []) as any)
  }, [hydroFeatures, mapLoaded])

  // Toggle layer visibility.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoadedRef.current) return
    const toggle = (layers: string[], visible: boolean) => {
      const vis = visible ? 'visible' : 'none'
      for (const id of layers) { if (m.getLayer(id)) m.setLayoutProperty(id, 'visibility', vis) }
    }
    toggle(['secteurs-fill', 'secteurs-line', 'secteurs-alert-line', 'secteurs-arrows'], visibleLayers.bsn)
    toggle(['piezo-layer', 'piezo-glyph-layer'], visibleLayers.piezo)
    toggle(['rain-layer', 'rain-glyph-layer'], visibleLayers.rain)
    toggle(['hydro-layer', 'hydro-glyph-layer'], visibleLayers.hydro)
  }, [visibleLayers, mapLoaded, sectorsReady])

  // Fly to sector when focusSectorId changes.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoadedRef.current || focusSectorId == null || !sectorGeoRef.current) return
    const feature = sectorGeoRef.current.features.find(
      f => f.properties?.sector_id === focusSectorId
    )
    if (!feature || !feature.geometry) return

    const bounds = new maplibregl.LngLatBounds()
    const addCoords = (coords: number[][]) => {
      for (const [lng, lat] of coords) {
        if (typeof lng === 'number' && typeof lat === 'number' && isFinite(lng) && isFinite(lat)) {
          bounds.extend([lng, lat] as [number, number])
        }
      }
    }

    const geom = feature.geometry
    if (geom.type === 'Polygon') {
      for (const ring of geom.coordinates) addCoords(ring as number[][])
    } else if (geom.type === 'MultiPolygon') {
      for (const poly of geom.coordinates) {
        for (const ring of poly) addCoords(ring as number[][])
      }
    }

    if (!bounds.isEmpty()) {
      m.fitBounds(bounds, { padding: 60, maxZoom: 9, duration: 600 })
    }
  }, [focusSectorId, mapLoaded])

  return <div ref={containerRef} className="absolute inset-0 w-full h-full" role="application" aria-label="Carte météo des nappes" />
}
