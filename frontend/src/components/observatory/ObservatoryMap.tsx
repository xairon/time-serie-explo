import { useRef, useEffect, useCallback, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { StationGeoJSONFeature, WfsLayerId } from '@/lib/observatory-types'
import { WFS_LAYER_MAP } from '@/lib/observatory-constants'

const FRANCE_CENTER: [number, number] = [2.5, 46.5]
const FRANCE_ZOOM = 5.5

const HER2_PALETTE = [
  '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
  '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
  '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3',
  '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#e6beff',
]

const REGION_COLORS: Record<string, string> = {
  '11': '#ef4444', '24': '#f59e0b', '27': '#84cc16', '28': '#06b6d4',
  '32': '#6366f1', '44': '#a78bfa', '52': '#22c55e', '53': '#3b82f6',
  '75': '#f97316', '76': '#ec4899', '84': '#14b8a6', '93': '#e879f9', '94': '#78716c',
}

const DEPT_PALETTE = [
  '#ef4444', '#f97316', '#f59e0b', '#84cc16', '#22c55e', '#14b8a6',
  '#06b6d4', '#3b82f6', '#6366f1', '#a78bfa', '#ec4899', '#e879f9',
]

const SANDRE_DISTRICT_COLORS: Record<string, string> = {
  'A': '#64748b', 'B1': '#f97316', 'C': '#a78bfa', 'D': '#ef4444',
  'E': '#ec4899', 'F': '#22c55e', 'G': '#eab308', 'H': '#3b82f6',
}

interface Props {
  features?: StationGeoJSONFeature[]
  excludedFeatures?: StationGeoJSONFeature[]
  allFeatures?: StationGeoJSONFeature[]
  showPiezo?: boolean
  showHydro?: boolean
  onStationClick?: (code: string, type: 'piezo' | 'hydro') => void
  onEmptyClick?: () => void
  onDeptClick?: (code: string | null) => void
  activeCodeDepartement?: string
  showRegions?: boolean
  showDepts?: boolean
  showHER?: boolean
  showSandre?: boolean
  onBassinClick?: (code: string | null) => void
  activeCodeBassin?: string
  onRegionClick?: (code: string | null, stationCodes: string[] | null) => void
  onHERClick?: (code: number | null, stationCodes: string[] | null) => void
  onSpatialFilter?: (codes: string[] | null) => void
  onBboxChange?: (bbox: [number, number, number, number] | null) => void
  activeWfsLayers?: Set<WfsLayerId>
  wfsData?: Record<string, any>
  selectedStationCode?: string | null
  showTerrain?: boolean
  flyToBbox?: [number, number, number, number] | null
  onFlyToComplete?: () => void
}

function featuresToGeoJSON(features: StationGeoJSONFeature[]) {
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
          codes_bdlisa: f.properties.codes_bdlisa ?? '',
          code_site: f.properties.code_site ?? '',
        },
      })),
  }
}

function buildClassificationColorExpression(): maplibregl.ExpressionSpecification {
  return [
    'match', ['get', 'classification'],
    'EXTREMEMENT_BAS', '#991b1b',
    'TRES_BAS', '#ef4444',
    'BAS', '#f97316',
    'NORMAL', '#10b981',
    'HAUT', '#3b82f6',
    'TRES_HAUT', '#1d4ed8',
    'EXTREMEMENT_HAUT', '#312e81',
    '#6b7280',
  ]
}

function createSdfIcon(draw: (ctx: CanvasRenderingContext2D, size: number) => void, size = 40): ImageData {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  draw(ctx, size)
  return ctx.getImageData(0, 0, size, size)
}

function drawPiezoCircle(ctx: CanvasRenderingContext2D, size: number) {
  const cx = size / 2, cy = size / 2
  ctx.beginPath()
  ctx.arc(cx, cy, size * 0.38, 0, Math.PI * 2)
  ctx.fillStyle = '#fff'
  ctx.fill()
}

function drawHydroDrop(ctx: CanvasRenderingContext2D, size: number) {
  const cx = size / 2
  const r = size * 0.32
  const bottomY = size * 0.62
  const tipY = size * 0.12
  ctx.beginPath()
  ctx.arc(cx, bottomY, r, 0.15 * Math.PI, 0.85 * Math.PI)
  ctx.lineTo(cx, tipY)
  ctx.closePath()
  ctx.fillStyle = '#fff'
  ctx.fill()
}

const CLUSTER_STYLE = {
  piezo: { bg: 'rgba(6,182,212,0.85)', border: 'rgba(6,182,212,0.5)', text: '#ffffff' },
  hydro: { bg: 'rgba(99,102,241,0.85)', border: 'rgba(99,102,241,0.5)', text: '#ffffff' },
} as const

function addClusteredSource(
  map: maplibregl.Map, sourceId: string, prefix: string, iconImage: string,
  style: { bg: string; border: string; text: string },
) {
  map.addSource(sourceId, { type: 'geojson', data: { type: 'FeatureCollection', features: [] }, cluster: true, clusterMaxZoom: 9, clusterRadius: 80 })
  map.addLayer({ id: `${prefix}-clusters`, type: 'circle', source: sourceId, filter: ['has', 'point_count'], paint: { 'circle-color': style.bg, 'circle-radius': ['step', ['get', 'point_count'], 16, 10, 20, 50, 26, 200, 32, 1000, 38], 'circle-stroke-width': 2, 'circle-stroke-color': style.border } })
  map.addLayer({ id: `${prefix}-cluster-count`, type: 'symbol', source: sourceId, filter: ['has', 'point_count'], layout: { 'text-field': ['get', 'point_count_abbreviated'], 'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'], 'text-size': ['step', ['get', 'point_count'], 11, 100, 12, 1000, 13], 'text-allow-overlap': true }, paint: { 'text-color': style.text } })
  map.addLayer({ id: `${prefix}-unclustered`, type: 'symbol', source: sourceId, filter: ['!', ['has', 'point_count']], layout: { 'icon-image': iconImage, 'icon-size': ['interpolate', ['linear'], ['zoom'], 4, 0.35, 8, 0.5, 12, 0.75], 'icon-allow-overlap': true }, paint: { 'icon-color': buildClassificationColorExpression() as any, 'icon-halo-color': '#000000', 'icon-halo-width': 0.8 } })
  map.on('click', `${prefix}-clusters`, (e) => {
    const cluster = e.features?.[0]; if (!cluster) return
    const clusterId = cluster.properties?.cluster_id
    const source = map.getSource(sourceId) as maplibregl.GeoJSONSource
    source.getClusterExpansionZoom(clusterId).then((zoom) => {
      const coords = (cluster.geometry as GeoJSON.Point).coordinates
      map.easeTo({ center: coords as [number, number], zoom: zoom + 0.5 })
    })
  })
  map.on('mouseenter', `${prefix}-clusters`, () => { map.getCanvas().style.cursor = 'pointer' })
  map.on('mouseleave', `${prefix}-clusters`, () => { map.getCanvas().style.cursor = '' })
}

function pointInRing(x: number, y: number, ring: number[][]): boolean {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1]
    if (((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)) inside = !inside
  }
  return inside
}

function pointInPolygon(lon: number, lat: number, geometry: any): boolean {
  if (geometry.type === 'Polygon') {
    const [outer, ...holes] = geometry.coordinates
    if (!pointInRing(lon, lat, outer)) return false
    for (const hole of holes) { if (pointInRing(lon, lat, hole)) return false }
    return true
  }
  if (geometry.type === 'MultiPolygon') {
    return geometry.coordinates.some((poly: number[][][]) => {
      const [outer, ...holes] = poly
      if (!pointInRing(lon, lat, outer)) return false
      for (const hole of holes) { if (pointInRing(lon, lat, hole)) return false }
      return true
    })
  }
  return false
}

function stationsInGeometry(features: StationGeoJSONFeature[], geometry: any): string[] {
  return features
    .filter(f => { const [lon, lat] = f.geometry.coordinates; return lon != null && lat != null && pointInPolygon(lon, lat, geometry) })
    .map(f => f.properties.code)
}

function computeBbox(geometry: any): [number, number, number, number] {
  const coords: number[][] = []
  const collect = (g: any) => {
    if (g.type === 'Point') coords.push(g.coordinates)
    else if (g.type === 'Polygon') g.coordinates[0].forEach((c: number[]) => coords.push(c))
    else if (g.type === 'MultiPolygon') g.coordinates.forEach((p: number[][][]) => p[0].forEach((c: number[]) => coords.push(c)))
  }
  collect(geometry)
  const lons = coords.map(c => c[0])
  const lats = coords.map(c => c[1])
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)]
}

export function ObservatoryMap({
  features, excludedFeatures, allFeatures,
  showPiezo = true, showHydro = true,
  onStationClick, onEmptyClick, onDeptClick,
  activeCodeDepartement = undefined,
  showRegions = false, showDepts = false, showHER = false, showSandre = false,
  onBassinClick, activeCodeBassin,
  onRegionClick, onHERClick, onSpatialFilter, onBboxChange,
  activeWfsLayers = new Set() as Set<WfsLayerId>,
  wfsData,
  selectedStationCode = null, showTerrain = false,
  flyToBbox = null, onFlyToComplete,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const mapLoadedRef = useRef(false)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [layersReady, setLayersReady] = useState(0)

  const featuresRef = useRef<StationGeoJSONFeature[]>([])
  featuresRef.current = features ?? []
  const excludedFeaturesRef = useRef<StationGeoJSONFeature[]>([])
  excludedFeaturesRef.current = excludedFeatures ?? []
  const allFeaturesRef = useRef<StationGeoJSONFeature[]>([])
  allFeaturesRef.current = allFeatures ?? features ?? []

  const onStationClickRef = useRef(onStationClick); onStationClickRef.current = onStationClick
  const onEmptyClickRef = useRef(onEmptyClick); onEmptyClickRef.current = onEmptyClick
  const onDeptClickRef = useRef(onDeptClick); onDeptClickRef.current = onDeptClick
  const onBassinClickRef = useRef(onBassinClick); onBassinClickRef.current = onBassinClick
  const activeCodeBassinRef = useRef(activeCodeBassin); activeCodeBassinRef.current = activeCodeBassin
  const onRegionClickRef = useRef(onRegionClick); onRegionClickRef.current = onRegionClick
  const onHERClickRef = useRef(onHERClick); onHERClickRef.current = onHERClick
  const onSpatialFilterRef = useRef(onSpatialFilter); onSpatialFilterRef.current = onSpatialFilter
  const onBboxChangeRef = useRef(onBboxChange); onBboxChangeRef.current = onBboxChange

  const [tooltip, setTooltip] = useState<{ name: string; x: number; y: number } | null>(null)

  const showRegionsRef = useRef(showRegions); showRegionsRef.current = showRegions
  const showDeptsRef = useRef(showDepts); showDeptsRef.current = showDepts
  const showHERRef = useRef(showHER); showHERRef.current = showHER
  const showSandreRef = useRef(showSandre); showSandreRef.current = showSandre
  const showPiezoRef = useRef(showPiezo); showPiezoRef.current = showPiezo
  const showHydroRef = useRef(showHydro); showHydroRef.current = showHydro
  const activeCodeDeptRef = useRef<string | undefined>(activeCodeDepartement)

  const updateSource = useCallback((map: maplibregl.Map, sourceId: string, feats: StationGeoJSONFeature[]) => {
    const source = map.getSource(sourceId) as maplibregl.GeoJSONSource | undefined
    if (source) source.setData(feats.length ? featuresToGeoJSON(feats) as any : { type: 'FeatureCollection', features: [] })
  }, [])

  // Init map
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',
      center: FRANCE_CENTER, zoom: FRANCE_ZOOM, maxBounds: [[-12, 38], [18, 54]],
    })
    map.addControl(new maplibregl.NavigationControl(), 'top-right')
    map.on('error', (e) => { console.error('MapLibre error:', e.error?.message ?? e) })

    map.on('load', () => {
      mapLoadedRef.current = true
      setMapLoaded(true)

      const style = map.getStyle()
      if (style?.layers) {
        style.layers.forEach((layer) => {
          if (layer.type === 'symbol') {
            const layout = (layer as maplibregl.SymbolLayerSpecification).layout
            if (layout?.['text-field']) map.setLayoutProperty(layer.id, 'text-field', ['coalesce', ['get', 'name:fr'], ['get', 'name']])
          }
        })
      }

      map.addImage('piezo-marker', createSdfIcon(drawPiezoCircle, 40), { sdf: true })
      map.addImage('hydro-marker', createSdfIcon(drawHydroDrop, 40), { sdf: true })

      addClusteredSource(map, 'piezo-stations', 'piezo', 'piezo-marker', CLUSTER_STYLE.piezo)
      addClusteredSource(map, 'hydro-stations', 'hydro', 'hydro-marker', CLUSTER_STYLE.hydro)

      const addExcludedSource = (sourceId: string, iconImage: string, layerId: string) => {
        map.addSource(sourceId, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
        map.addLayer({ id: layerId, type: 'symbol', source: sourceId, layout: { 'icon-image': iconImage, 'icon-size': ['interpolate', ['linear'], ['zoom'], 4, 0.3, 8, 0.4, 12, 0.6], 'icon-allow-overlap': true }, paint: { 'icon-color': '#6b7280', 'icon-opacity': 0.35 } })
      }
      addExcludedSource('piezo-excluded', 'piezo-marker', 'piezo-excluded-layer')
      addExcludedSource('hydro-excluded', 'hydro-marker', 'hydro-excluded-layer')

      if (featuresRef.current.length) {
        updateSource(map, 'piezo-stations', featuresRef.current.filter(f => f.properties.type === 'piezo'))
        updateSource(map, 'hydro-stations', featuresRef.current.filter(f => f.properties.type === 'hydro'))
      }
      if (excludedFeaturesRef.current.length) {
        updateSource(map, 'piezo-excluded', excludedFeaturesRef.current.filter(f => f.properties.type === 'piezo'))
        updateSource(map, 'hydro-excluded', excludedFeaturesRef.current.filter(f => f.properties.type === 'hydro'))
      }

      Promise.all([
        fetch('/geo/regions.geojson').then(r => r.json()),
        fetch('/geo/departments.geojson').then(r => r.json()),
        fetch('/geo/her.geojson').then(r => r.json()),
        fetch('/geo/bassins.geojson').then(r => r.json()),
      ]).then(([regionsData, deptsData, herData, bassinsData]) => {
        if (!mapRef.current) return
        const addLayer = (sourceId: string, data: any, fillId: string, lineId: string, vis: string, fillPaint: any, linePaint: any) => {
          map.addSource(sourceId, { type: 'geojson', data, generateId: true })
          map.addLayer({ id: fillId, type: 'fill', source: sourceId, paint: fillPaint, layout: { visibility: vis as any } }, 'piezo-clusters')
          map.addLayer({ id: lineId, type: 'line', source: sourceId, paint: linePaint, layout: { visibility: vis as any } }, 'piezo-clusters')
        }

        // Regions
        const regionColorExpr: any = ['match', ['get', 'code'], ...Object.entries(REGION_COLORS).flatMap(([k, v]) => [k, v]), '#94a3b8']
        addLayer('regions', regionsData, 'regions-fill', 'regions-line', showRegionsRef.current ? 'visible' : 'none',
          { 'fill-color': regionColorExpr, 'fill-opacity': ['case', ['boolean', ['feature-state', 'hover'], false], 0.35, 0.15] },
          { 'line-color': regionColorExpr, 'line-width': 2, 'line-opacity': 0.7 })
        {
          let hovId: number | null = null
          map.on('mousemove', 'regions-fill', (e) => { if (!e.features?.length) return; if (hovId !== null) map.setFeatureState({ source: 'regions', id: hovId }, { hover: false }); hovId = e.features[0].id as number; map.setFeatureState({ source: 'regions', id: hovId }, { hover: true }); setTooltip({ name: e.features[0].properties?.nom ?? '', x: e.point.x, y: e.point.y }) })
          map.on('mouseleave', 'regions-fill', () => { if (hovId !== null) map.setFeatureState({ source: 'regions', id: hovId }, { hover: false }); hovId = null; setTooltip(null) })
          map.on('click', 'regions-fill', (e) => {
            const stationHits = map.queryRenderedFeatures(e.point, { layers: ['piezo-clusters', 'hydro-clusters', 'piezo-unclustered', 'hydro-unclustered', 'piezo-excluded-layer', 'hydro-excluded-layer', 'basin-stations-layer'].filter(id => !!map.getLayer(id)) })
            if (stationHits.length > 0) return
            const feat = e.features?.[0]; if (!feat) return
            const bbox = computeBbox(feat.geometry)
            map.fitBounds(bbox as maplibregl.LngLatBoundsLike, { padding: 60, duration: 500 })
            const codes = stationsInGeometry(featuresRef.current, feat.geometry)
            onSpatialFilterRef.current?.(codes.length > 0 ? codes : null); onBboxChangeRef.current?.(bbox)
          })
          map.on('mouseenter', 'regions-fill', () => { map.getCanvas().style.cursor = 'pointer' })
          map.on('mouseleave', 'regions-fill', () => { map.getCanvas().style.cursor = '' })
        }

        // Departments
        const deptColorExpr: any = ['match', ['slice', ['get', 'code'], 0, 1], '0', DEPT_PALETTE[0], '1', DEPT_PALETTE[1], '2', DEPT_PALETTE[2], '3', DEPT_PALETTE[3], '4', DEPT_PALETTE[4], '5', DEPT_PALETTE[5], '6', DEPT_PALETTE[6], '7', DEPT_PALETTE[7], '8', DEPT_PALETTE[8], '9', DEPT_PALETTE[9], DEPT_PALETTE[0]]
        addLayer('departments', deptsData, 'depts-fill', 'depts-line', showDeptsRef.current ? 'visible' : 'none',
          { 'fill-color': ['case', ['==', ['get', 'code'], activeCodeDeptRef.current ?? '$$NONE$$'], '#22d3ee', deptColorExpr] as any, 'fill-opacity': ['case', ['==', ['get', 'code'], activeCodeDeptRef.current ?? '$$NONE$$'], 0.30, ['boolean', ['feature-state', 'hover'], false], 0.25, 0.12] },
          { 'line-color': deptColorExpr as any, 'line-width': 1.5, 'line-opacity': 0.6 })
        {
          let hovId: number | null = null
          map.on('mousemove', 'depts-fill', (e) => { if (!e.features?.length) return; if (hovId !== null) map.setFeatureState({ source: 'departments', id: hovId }, { hover: false }); hovId = e.features[0].id as number; map.setFeatureState({ source: 'departments', id: hovId }, { hover: true }); setTooltip({ name: `${e.features[0].properties?.nom ?? ''} (${e.features[0].properties?.code ?? ''})`, x: e.point.x, y: e.point.y }) })
          map.on('mouseleave', 'depts-fill', () => { if (hovId !== null) map.setFeatureState({ source: 'departments', id: hovId }, { hover: false }); hovId = null; setTooltip(null) })
          map.on('click', 'depts-fill', (e) => {
            const stationHits = map.queryRenderedFeatures(e.point, { layers: ['piezo-clusters', 'hydro-clusters', 'piezo-unclustered', 'hydro-unclustered', 'piezo-excluded-layer', 'hydro-excluded-layer', 'basin-stations-layer'].filter(id => !!map.getLayer(id)) })
            if (stationHits.length > 0) return
            const feat = e.features?.[0]; if (!feat) return
            const bbox = computeBbox(feat.geometry); map.fitBounds(bbox as maplibregl.LngLatBoundsLike, { padding: 60, duration: 500 })
            const code = feat.properties?.code ?? null; const current = activeCodeDeptRef.current; const deselecting = code === current
            onDeptClickRef.current?.(deselecting ? null : code); onBboxChangeRef.current?.(deselecting ? null : bbox)
          })
          map.on('mouseenter', 'depts-fill', () => { map.getCanvas().style.cursor = 'pointer' })
          map.on('mouseleave', 'depts-fill', () => { map.getCanvas().style.cursor = '' })
        }

        // HER
        const herColorExpr: any = ['match', ['%', ['get', 'code'], HER2_PALETTE.length], ...HER2_PALETTE.flatMap((c, i) => [i, c]), '#94a3b8']
        addLayer('her', herData, 'her-fill', 'her-line', showHERRef.current ? 'visible' : 'none',
          { 'fill-color': herColorExpr, 'fill-opacity': ['case', ['boolean', ['feature-state', 'hover'], false], 0.30, 0.15] },
          { 'line-color': 'rgba(255,255,255,0.3)', 'line-width': 0.8 })
        {
          let hovId: number | null = null
          map.on('mousemove', 'her-fill', (e) => { if (!e.features?.length) return; if (hovId !== null) map.setFeatureState({ source: 'her', id: hovId }, { hover: false }); hovId = e.features[0].id as number; map.setFeatureState({ source: 'her', id: hovId }, { hover: true }); const nom = e.features[0].properties?.nom ?? ''; const her1 = e.features[0].properties?.nom_her1 ?? ''; setTooltip({ name: `${nom}${her1 ? ` - ${her1}` : ''}`, x: e.point.x, y: e.point.y }) })
          map.on('mouseleave', 'her-fill', () => { if (hovId !== null) map.setFeatureState({ source: 'her', id: hovId }, { hover: false }); hovId = null; setTooltip(null) })
          map.on('click', 'her-fill', (e) => {
            const stationHits = map.queryRenderedFeatures(e.point, { layers: ['piezo-clusters', 'hydro-clusters', 'piezo-unclustered', 'hydro-unclustered', 'piezo-excluded-layer', 'hydro-excluded-layer', 'basin-stations-layer'].filter(id => !!map.getLayer(id)) })
            if (stationHits.length > 0) return
            const feat = e.features?.[0]; if (!feat) return
            const bbox = computeBbox(feat.geometry); map.fitBounds(bbox as maplibregl.LngLatBoundsLike, { padding: 60, duration: 500 })
            const codes = stationsInGeometry(featuresRef.current, feat.geometry)
            onSpatialFilterRef.current?.(codes.length > 0 ? codes : null); onBboxChangeRef.current?.(bbox)
          })
          map.on('mouseenter', 'her-fill', () => { map.getCanvas().style.cursor = 'pointer' })
          map.on('mouseleave', 'her-fill', () => { map.getCanvas().style.cursor = '' })
        }

        // Bassins
        const sandreColorExpr: any = ['match', ['get', 'CdBH'], ...Object.entries(SANDRE_DISTRICT_COLORS).flatMap(([k, v]) => [k, v]), '#94a3b8']
        addLayer('bassins', bassinsData, 'bassins-fill', 'bassins-line', showSandreRef.current ? 'visible' : 'none',
          { 'fill-color': sandreColorExpr, 'fill-opacity': ['case', ['==', ['get', 'CdBH'], activeCodeBassinRef.current ?? '$$NONE$$'], 0.35, ['boolean', ['feature-state', 'hover'], false], 0.20, 0.10] },
          { 'line-color': sandreColorExpr, 'line-width': 1.5, 'line-opacity': 0.5 })
        {
          let hovId: number | null = null
          map.on('mousemove', 'bassins-fill', (e) => { if (!e.features?.length) return; if (hovId !== null) map.setFeatureState({ source: 'bassins', id: hovId }, { hover: false }); hovId = e.features[0].id as number; map.setFeatureState({ source: 'bassins', id: hovId }, { hover: true }); setTooltip({ name: e.features[0].properties?.LbBH ?? e.features[0].properties?.CdBH ?? '', x: e.point.x, y: e.point.y }) })
          map.on('mouseleave', 'bassins-fill', () => { if (hovId !== null) map.setFeatureState({ source: 'bassins', id: hovId }, { hover: false }); hovId = null; setTooltip(null) })
          map.on('click', 'bassins-fill', (e) => {
            const stationHits = map.queryRenderedFeatures(e.point, { layers: ['piezo-clusters', 'hydro-clusters', 'piezo-unclustered', 'hydro-unclustered', 'piezo-excluded-layer', 'hydro-excluded-layer', 'basin-stations-layer'].filter(id => !!map.getLayer(id)) })
            if (stationHits.length > 0) return
            const feat = e.features?.[0]; if (!feat) return
            const code = feat.properties?.CdBH ?? null; const current = activeCodeBassinRef.current
            const bbox = computeBbox(feat.geometry); map.fitBounds(bbox as maplibregl.LngLatBoundsLike, { padding: 60, duration: 500 })
            if (code === current) { onBassinClickRef.current?.(null); onSpatialFilterRef.current?.(null); onBboxChangeRef.current?.(null) }
            else { onBassinClickRef.current?.(code); const codes = stationsInGeometry(featuresRef.current, feat.geometry); onSpatialFilterRef.current?.(codes.length > 0 ? codes : null); onBboxChangeRef.current?.(bbox) }
          })
          map.on('mouseenter', 'bassins-fill', () => { map.getCanvas().style.cursor = 'pointer' })
          map.on('mouseleave', 'bassins-fill', () => { map.getCanvas().style.cursor = '' })
        }

        setLayersReady(4)
      }).catch(err => console.error('Failed to load reference layers:', err))

      // Click handlers for individual stations
      map.on('click', 'piezo-unclustered', (e) => { const code = e.features?.[0]?.properties?.code; if (code) onStationClickRef.current?.(code, 'piezo') })
      map.on('click', 'hydro-unclustered', (e) => { const code = e.features?.[0]?.properties?.code; if (code) onStationClickRef.current?.(code, 'hydro') })
      map.on('click', 'piezo-excluded-layer', (e) => { const code = e.features?.[0]?.properties?.code; if (code) onStationClickRef.current?.(code, 'piezo') })
      map.on('click', 'hydro-excluded-layer', (e) => { const code = e.features?.[0]?.properties?.code; if (code) onStationClickRef.current?.(code, 'hydro') })

      const pointerLayers = ['piezo-unclustered', 'hydro-unclustered', 'piezo-excluded-layer', 'hydro-excluded-layer']
      pointerLayers.forEach((layer) => {
        map.on('mouseenter', layer, () => { map.getCanvas().style.cursor = 'pointer' })
        map.on('mouseleave', layer, () => { map.getCanvas().style.cursor = '' })
      })

      map.on('click', (e) => {
        const stationLayers = ['piezo-clusters', 'hydro-clusters', 'piezo-unclustered', 'hydro-unclustered', 'piezo-excluded-layer', 'hydro-excluded-layer', 'basin-stations-layer'].filter(id => !!map.getLayer(id))
        const stationHits = map.queryRenderedFeatures(e.point, { layers: stationLayers })
        if (stationHits.length > 0) return
        const visibleSpatialLayers = ['depts-fill', 'regions-fill', 'her-fill', 'bassins-fill', ...Object.entries(WFS_LAYER_MAP).filter(([, cfg]) => cfg.geometryType === 'polygon').map(([id]) => `wfs-${id}-fill`)].filter(id => { if (!map.getLayer(id)) return false; return map.getLayoutProperty(id, 'visibility') === 'visible' })
        const spatialHits = map.queryRenderedFeatures(e.point, { layers: visibleSpatialLayers })
        if (spatialHits.length > 0) return
        onEmptyClickRef.current?.(); onDeptClickRef.current?.(null); onBassinClickRef.current?.(null); onSpatialFilterRef.current?.(null); onBboxChangeRef.current?.(null)
      })
    })

    mapRef.current = map
    return () => { map.remove(); mapRef.current = null; mapLoadedRef.current = false }
  }, [updateSource])

  // Sync features
  useEffect(() => { if (!mapRef.current || !mapLoadedRef.current) return; const all = features ?? []; updateSource(mapRef.current, 'piezo-stations', all.filter(f => f.properties.type === 'piezo')); updateSource(mapRef.current, 'hydro-stations', all.filter(f => f.properties.type === 'hydro')) }, [features, updateSource])
  useEffect(() => { if (!mapRef.current || !mapLoadedRef.current) return; const ex = excludedFeatures ?? []; updateSource(mapRef.current, 'piezo-excluded', ex.filter(f => f.properties.type === 'piezo')); updateSource(mapRef.current, 'hydro-excluded', ex.filter(f => f.properties.type === 'hydro')) }, [excludedFeatures, updateSource])

  // Toggle piezo/hydro
  useEffect(() => {
    if (!mapRef.current || !mapLoadedRef.current) return; const map = mapRef.current
    const toggle = (layers: string[], visible: boolean) => { const vis = visible ? 'visible' : 'none'; layers.forEach(id => { if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', vis) }) }
    toggle(['piezo-clusters', 'piezo-cluster-count', 'piezo-unclustered', 'piezo-excluded-layer'], showPiezo)
    toggle(['hydro-clusters', 'hydro-cluster-count', 'hydro-unclustered', 'hydro-excluded-layer'], showHydro)
  }, [showPiezo, showHydro, mapLoaded])

  // Toggle terrain
  useEffect(() => {
    if (!mapRef.current || !mapLoadedRef.current) return; const map = mapRef.current
    if (showTerrain) {
      if (!map.getSource('terrain-dem')) {
        map.addSource('terrain-dem', { type: 'raster-dem', tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'], encoding: 'terrarium', tileSize: 256, maxzoom: 15 })
        map.addLayer({ id: 'hillshading', type: 'hillshade', source: 'terrain-dem', paint: { 'hillshade-shadow-color': '#473B24', 'hillshade-highlight-color': '#ffffff', 'hillshade-exaggeration': 0.3, 'hillshade-illumination-direction': 315 } }, map.getStyle().layers.find(l => l.type === 'symbol')?.id)
      } else { if (map.getLayer('hillshading')) map.setLayoutProperty('hillshading', 'visibility', 'visible') }
    } else { if (map.getLayer('hillshading')) map.setLayoutProperty('hillshading', 'visibility', 'none') }
  }, [showTerrain, mapLoaded])

  // Toggle reference layers
  useEffect(() => {
    if (!mapRef.current || !mapLoadedRef.current) return; const map = mapRef.current
    const toggle = (fillId: string, lineId: string, visible: boolean) => { const vis = visible ? 'visible' : 'none'; if (map.getLayer(fillId)) map.setLayoutProperty(fillId, 'visibility', vis); if (map.getLayer(lineId)) map.setLayoutProperty(lineId, 'visibility', vis) }
    toggle('regions-fill', 'regions-line', showRegions); toggle('depts-fill', 'depts-line', showDepts); toggle('her-fill', 'her-line', showHER); toggle('bassins-fill', 'bassins-line', showSandre)
  }, [showRegions, showDepts, showHER, showSandre, mapLoaded, layersReady])

  // Sync active dept
  useEffect(() => {
    activeCodeDeptRef.current = activeCodeDepartement
    if (!mapRef.current || !mapLoadedRef.current) return; const map = mapRef.current; if (!map.getLayer('depts-fill')) return
    const deptColor: any = ['match', ['slice', ['get', 'code'], 0, 1], '0', DEPT_PALETTE[0], '1', DEPT_PALETTE[1], '2', DEPT_PALETTE[2], '3', DEPT_PALETTE[3], '4', DEPT_PALETTE[4], '5', DEPT_PALETTE[5], '6', DEPT_PALETTE[6], '7', DEPT_PALETTE[7], '8', DEPT_PALETTE[8], '9', DEPT_PALETTE[9], DEPT_PALETTE[0]]
    map.setPaintProperty('depts-fill', 'fill-color', ['case', ['==', ['get', 'code'], activeCodeDepartement ?? '$$NONE$$'], '#22d3ee', deptColor])
    map.setPaintProperty('depts-fill', 'fill-opacity', ['case', ['==', ['get', 'code'], activeCodeDepartement ?? '$$NONE$$'], 0.30, ['boolean', ['feature-state', 'hover'], false], 0.25, 0.10])
  }, [activeCodeDepartement])

  // Sync active bassin
  useEffect(() => {
    activeCodeBassinRef.current = activeCodeBassin
    if (!mapRef.current || !mapLoadedRef.current) return; const map = mapRef.current; if (!map.getLayer('bassins-fill')) return
    map.setPaintProperty('bassins-fill', 'fill-opacity', ['case', ['==', ['get', 'CdBH'], activeCodeBassin ?? '$$NONE$$'], 0.35, ['boolean', ['feature-state', 'hover'], false], 0.20, 0.10])
  }, [activeCodeBassin])

  // Selected station ring
  useEffect(() => {
    if (!mapRef.current || !mapLoadedRef.current) return; const map = mapRef.current; const layerId = 'selected-station-ring'
    if (selectedStationCode) {
      const feat = featuresRef.current.find(f => f.properties.code === selectedStationCode); if (!feat) return
      const data = { type: 'FeatureCollection' as const, features: [{ type: 'Feature' as const, geometry: feat.geometry, properties: {} }] }
      if (!map.getSource('selected-station')) {
        map.addSource('selected-station', { type: 'geojson', data: data as any })
        map.addLayer({ id: layerId, type: 'circle', source: 'selected-station', paint: { 'circle-radius': ['interpolate', ['linear'], ['zoom'], 4, 10, 8, 14, 12, 18], 'circle-color': 'transparent', 'circle-stroke-width': 3, 'circle-stroke-color': '#22d3ee', 'circle-opacity': 1 } })
      } else { const src = map.getSource('selected-station') as maplibregl.GeoJSONSource; src.setData(data as any); if (map.getLayer(layerId)) map.setLayoutProperty(layerId, 'visibility', 'visible') }
    } else { if (map.getLayer(layerId)) map.setLayoutProperty(layerId, 'visibility', 'none') }
  }, [selectedStationCode])

  // Sync WFS layers
  useEffect(() => {
    if (!mapRef.current || !mapLoadedRef.current) return; const map = mapRef.current
    for (const [layerId, config] of Object.entries(WFS_LAYER_MAP)) {
      const fillId = `wfs-${layerId}-fill`; const lineId = `wfs-${layerId}-line`; const isActive = activeWfsLayers.has(layerId as WfsLayerId); const data = wfsData?.[layerId]
      if (isActive && data && !map.getSource(`wfs-${layerId}`)) {
        map.addSource(`wfs-${layerId}`, { type: 'geojson', data, generateId: true })
        if (config.geometryType === 'polygon') {
          map.addLayer({ id: fillId, type: 'fill', source: `wfs-${layerId}`, minzoom: config.minZoom, paint: { 'fill-color': config.color, 'fill-opacity': ['case', ['boolean', ['feature-state', 'hover'], false], 0.30, 0.12] } }, 'piezo-clusters')
          map.addLayer({ id: lineId, type: 'line', source: `wfs-${layerId}`, minzoom: config.minZoom, paint: { 'line-color': config.color, 'line-width': 1, 'line-opacity': 0.5 } }, 'piezo-clusters')
        } else {
          map.addLayer({ id: lineId, type: 'line', source: `wfs-${layerId}`, minzoom: config.minZoom, paint: { 'line-color': config.color, 'line-width': ['interpolate', ['linear'], ['zoom'], 6, 1, 12, 3], 'line-opacity': 0.7 } }, 'piezo-clusters')
        }
        const hoverLayerId = config.geometryType === 'polygon' ? fillId : lineId; let hoveredId: number | null = null
        map.on('mousemove', hoverLayerId, (e) => { if (!e.features?.length) return; const feat = e.features[0]; if (hoveredId !== null) map.setFeatureState({ source: `wfs-${layerId}`, id: hoveredId }, { hover: false }); hoveredId = feat.id as number; map.setFeatureState({ source: `wfs-${layerId}`, id: hoveredId }, { hover: true }); const parts = config.tooltipFields.map(f => feat.properties?.[f]).filter(Boolean); setTooltip({ name: parts.join(' \u2014 ') || layerId, x: e.point.x, y: e.point.y }) })
        map.on('mouseleave', hoverLayerId, () => { if (hoveredId !== null) map.setFeatureState({ source: `wfs-${layerId}`, id: hoveredId }, { hover: false }); hoveredId = null; setTooltip(null) })
        map.on('mouseenter', hoverLayerId, () => { map.getCanvas().style.cursor = 'pointer' }); map.on('mouseleave', hoverLayerId, () => { map.getCanvas().style.cursor = '' })
        if (config.geometryType === 'polygon') {
          map.on('click', fillId, (e) => { const feat = e.features?.[0]; if (!feat) return; const bbox = computeBbox(feat.geometry); map.fitBounds(bbox as maplibregl.LngLatBoundsLike, { padding: 60, duration: 500 }); const codes = stationsInGeometry(featuresRef.current, feat.geometry); onSpatialFilterRef.current?.(codes.length > 0 ? codes : null); onBboxChangeRef.current?.(bbox) })
        }
      }
      if (data && map.getSource(`wfs-${layerId}`)) { const src = map.getSource(`wfs-${layerId}`) as maplibregl.GeoJSONSource; src.setData(data) }
      if (map.getLayer(fillId)) map.setLayoutProperty(fillId, 'visibility', isActive ? 'visible' : 'none')
      if (map.getLayer(lineId)) map.setLayoutProperty(lineId, 'visibility', isActive ? 'visible' : 'none')
    }
    const adminLayers = ['regions-fill', 'regions-line', 'depts-fill', 'depts-line', 'her-fill', 'her-line', 'bassins-fill', 'bassins-line']
    adminLayers.forEach(id => { if (map.getLayer(id)) map.moveLayer(id, 'piezo-clusters') })
  }, [activeWfsLayers, wfsData])

  // Fly to bbox
  const onFlyToCompleteRef = useRef(onFlyToComplete); onFlyToCompleteRef.current = onFlyToComplete
  useEffect(() => {
    if (!flyToBbox || !mapRef.current || !mapLoadedRef.current) return
    mapRef.current.fitBounds(flyToBbox as maplibregl.LngLatBoundsLike, { padding: 80, duration: 600 }); onFlyToCompleteRef.current?.()
  }, [flyToBbox])

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" role="application" aria-label="Carte interactive des stations hydrologiques françaises" />
      {tooltip && (
        <div className="absolute z-20 bg-gray-900/95 border border-white/10 rounded px-2 py-1 text-xs text-white pointer-events-none" style={{ left: tooltip.x + 12, top: tooltip.y - 8 }}>{tooltip.name}</div>
      )}
      <div className="absolute bottom-3 left-3 z-10 bg-gray-900/90 backdrop-blur-sm border border-white/10 rounded-lg px-3 py-2 text-[10px] space-y-1.5">
        <div className="text-gray-400 font-medium uppercase tracking-wider mb-1">État</div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
          {[
            ['#991b1b', 'Extr. bas'], ['#ef4444', 'Très bas'], ['#f97316', 'Bas'], ['#10b981', 'Normal'],
            ['#3b82f6', 'Haut'], ['#1d4ed8', 'Très haut'], ['#312e81', 'Extr. haut'], ['#6b7280', 'Non classé'],
          ].map(([color, label]) => (
            <div key={label} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
              <span className="text-gray-300">{label}</span>
            </div>
          ))}
        </div>
        <div className="border-t border-white/10 pt-1 mt-1 flex gap-3">
          <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: '#10b981' }} /><span className="text-gray-300">Piézo</span></div>
          <div className="flex items-center gap-1.5">
            <svg width="12" height="14" viewBox="0 0 12 14" className="shrink-0"><path d="M6,1 Q6,1 10,8 A4.5,4.5 0 1,1 2,8 Q6,1 6,1Z" fill="#10b981" /></svg>
            <span className="text-gray-300">Hydro</span>
          </div>
        </div>
      </div>
    </div>
  )
}
