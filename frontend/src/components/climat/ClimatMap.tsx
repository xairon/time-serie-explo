import { useRef, useEffect, useState, useCallback } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { climatMonthlyToSquares, climatIndicesToSquares, climatSelectedCellSquare } from '@/lib/climat-grid'
import { climatRawColorExpression, climatIndexColorExpression, isClimatIndexVariable } from '@/lib/climat-colors'
import type { ClimatVariable } from '@/lib/climat-colors'
import type { ClimatMonthlyPoint, ClimatIndexPoint } from '@/lib/observatory-types'
import type { SelectedCell } from '@/hooks/useClimat'

const FRANCE_CENTER: [number, number] = [2.5, 46.5]
const FRANCE_ZOOM = 5.2
const SRC = 'climat-grid'
const FILL = 'climat-grid-fill'
const SELECTED_SRC = 'climat-selected-cell'
const SELECTED_LINE = 'climat-selected-cell-line'

interface Props {
  variable: ClimatVariable
  monthlyPoints?: ClimatMonthlyPoint[]
  indexPoints?: ClimatIndexPoint[]
  /** Called with the clicked cell's centre (rounded to 0.1°) — opens the Point panel (Task B2). */
  onCellClick?: (lat: number, lon: number) => void
  /** Currently open Point panel's cell, outlined on the map for orientation. */
  selectedCell?: SelectedCell | null
}

/** Lean full-screen MapLibre map for the Climat "Situation" view — a single ERA5-grid
 *  squares layer (no stations, no admin zones, no drawer). Deliberately not the shared
 *  ObservatoryMap, which is coupled to the station/zone/sensor overlay state (see plan
 *  Task B1: "if ObservatoryMap is too coupled, build a lean ClimatMap"). Reuses the same
 *  base style + squares approach as era5-grid.ts / ObservatoryMap's era5-grid-fill layer. */
export function ClimatMap({ variable, monthlyPoints, indexPoints, onCellClick, selectedCell }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const onCellClickRef = useRef(onCellClick); onCellClickRef.current = onCellClick

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
      map.addSource(SRC, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      map.addLayer({ id: FILL, type: 'fill', source: SRC, paint: { 'fill-color': '#6b7280', 'fill-opacity': 0.65 } })
      map.addSource(SELECTED_SRC, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      map.addLayer({
        id: SELECTED_LINE, type: 'line', source: SELECTED_SRC,
        paint: { 'line-color': '#22d3ee', 'line-width': 3 },
      })
      // Click opens the Point panel (Task B2) instead of a quick popup — the panel
      // carries far more information (full history, SPI/STI, drought episodes) than
      // a one-line value ever could. The cell centre is the clicked square's centroid
      // (average of its 4 corners), rounded back to the 0.1° grid.
      map.on('click', FILL, (e) => {
        const f = e.features?.[0]
        if (!f || f.geometry.type !== 'Polygon') return
        const ring = f.geometry.coordinates[0]
        const lon = (ring[0][0] + ring[1][0] + ring[2][0] + ring[3][0]) / 4
        const lat = (ring[0][1] + ring[1][1] + ring[2][1] + ring[3][1]) / 4
        onCellClickRef.current?.(Math.round(lat * 10) / 10, Math.round(lon * 10) / 10)
      })
      map.on('mouseenter', FILL, () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mouseleave', FILL, () => { map.getCanvas().style.cursor = '' })
      mapRef.current = map
      setMapLoaded(true)
    })
    return () => { map.remove(); mapRef.current = null }
    // Map init must run exactly once — see updateLayer/selected-cell effects below for
    // everything that needs to react to prop changes without rebuilding the map.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const updateLayer = useCallback(() => {
    const map = mapRef.current
    if (!map || !mapLoaded || !map.getLayer(FILL)) return
    const isIndex = isClimatIndexVariable(variable)
    const data = isIndex
      ? climatIndicesToSquares(indexPoints ?? [], variable as 'spi' | 'sti')
      : climatMonthlyToSquares(monthlyPoints ?? [])
    ;(map.getSource(SRC) as maplibregl.GeoJSONSource).setData(data as any)
    map.setPaintProperty(
      FILL, 'fill-color',
      isIndex ? climatIndexColorExpression(variable as 'spi' | 'sti') as any : climatRawColorExpression(variable) as any,
    )
  }, [mapLoaded, variable, monthlyPoints, indexPoints])

  useEffect(() => { updateLayer() }, [updateLayer])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapLoaded || !map.getSource(SELECTED_SRC)) return
    const data = selectedCell
      ? climatSelectedCellSquare(selectedCell.lat, selectedCell.lon)
      : { type: 'FeatureCollection' as const, features: [] }
    ;(map.getSource(SELECTED_SRC) as maplibregl.GeoJSONSource).setData(data as any)
  }, [mapLoaded, selectedCell])

  return <div ref={containerRef} className="absolute inset-0" data-testid="climat-map" />
}
