import { useRef, useEffect, useState, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { climatMonthlyToSquares, climatIndicesToSquares } from '@/lib/climat-grid'
import { climatRawColorExpression, climatIndexColorExpression, climatFormatValue, isClimatIndexVariable, CLIMAT_VARIABLES } from '@/lib/climat-colors'
import type { ClimatVariable } from '@/lib/climat-colors'
import type { ClimatMonthlyPoint, ClimatIndexPoint } from '@/lib/observatory-types'

const FRANCE_CENTER: [number, number] = [2.5, 46.5]
const FRANCE_ZOOM = 5.2
const SRC = 'climat-grid'
const FILL = 'climat-grid-fill'

interface Props {
  variable: ClimatVariable
  window: number
  monthlyPoints?: ClimatMonthlyPoint[]
  indexPoints?: ClimatIndexPoint[]
}

/** Lean full-screen MapLibre map for the Climat "Situation" view — a single ERA5-grid
 *  squares layer (no stations, no admin zones, no drawer). Deliberately not the shared
 *  ObservatoryMap, which is coupled to the station/zone/sensor overlay state (see plan
 *  Task B1: "if ObservatoryMap is too coupled, build a lean ClimatMap"). Reuses the same
 *  base style + squares approach as era5-grid.ts / ObservatoryMap's era5-grid-fill layer. */
export function ClimatMap({ variable, window, monthlyPoints, indexPoints }: Props) {
  const { t } = useTranslation()
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const variableRef = useRef(variable); variableRef.current = variable
  const windowRef = useRef(window); windowRef.current = window
  const tRef = useRef(t); tRef.current = t

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
      map.on('click', FILL, (e) => {
        const f = e.features?.[0]
        if (!f) return
        const pr = f.properties as Record<string, string | number>
        const v = Number(pr.value)
        const curVar = variableRef.current
        // SPI/STI share the observatory overlay's popup label i18n keys — same wording, no duplication.
        // Reads tRef.current (not the closed-over `t`) so popups created after a language change
        // still pick up the new translations, even though this effect never re-runs.
        const curT = tRef.current
        const label = curVar === 'spi'
          ? curT('observatory.era5.popupSpiLabel', { n: windowRef.current })
          : curVar === 'sti'
            ? curT('observatory.era5.popupStiLabel', { n: windowRef.current })
            : curT(CLIMAT_VARIABLES[curVar].labelKey)
        new maplibregl.Popup({ closeButton: true })
          .setLngLat(e.lngLat)
          .setHTML(`<div style="font-size:12px;line-height:1.5"><strong>${label}</strong><div>${climatFormatValue(curVar, Number.isFinite(v) ? v : null)}</div></div>`)
          .addTo(map)
      })
      map.on('mouseenter', FILL, () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mouseleave', FILL, () => { map.getCanvas().style.cursor = '' })
      mapRef.current = map
      setMapLoaded(true)
    })
    return () => { map.remove(); mapRef.current = null }
    // Map init must run exactly once: react-i18next hands out a new `t` reference on every
    // language change, and depending on it here would tear down and rebuild the whole
    // maplibre map (style reload, viewport reset, tile cache lost). Language-dependent
    // strings read tRef.current at event time instead (see click handler above).
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

  return <div ref={containerRef} className="absolute inset-0" data-testid="climat-map" />
}
