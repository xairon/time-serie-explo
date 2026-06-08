import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { TerritorySituation } from '@/lib/observatory-types'
import { classColor, INSUFFICIENT_COLOR } from '@/lib/situation-format'
import regionsUrl from '@/assets/geo/regions.geojson?url'
import departementsUrl from '@/assets/geo/departements.geojson?url'

const FRANCE_CENTER: [number, number] = [2.4, 46.6]
const FRANCE_ZOOM = 4.7

type Props = {
  level: 'region' | 'department'
  territories: TerritorySituation[]
  onSelectRegion: (code: string) => void
  onSelectDepartment: (code: string) => void
}

function fillColorExpression(territories: TerritorySituation[]) {
  const pairs: string[] = []
  for (const t of territories) pairs.push(t.code, classColor(t.situation_class))
  if (pairs.length === 0) return INSUFFICIENT_COLOR
  return ['match', ['get', 'code'], ...pairs, INSUFFICIENT_COLOR] as unknown as maplibregl.ExpressionSpecification
}

export function TerritoryChoropleth({ level, territories, onSelectRegion, onSelectDepartment }: Props) {
  const mapRef = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const levelRef = useRef(level); levelRef.current = level
  const onRegion = useRef(onSelectRegion); onRegion.current = onSelectRegion
  const onDept = useRef(onSelectDepartment); onDept.current = onSelectDepartment

  useEffect(() => {
    if (!mapRef.current || map.current) return
    const m = new maplibregl.Map({
      container: mapRef.current,
      style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
      center: FRANCE_CENTER, zoom: FRANCE_ZOOM,
    })
    map.current = m
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    m.on('click', 'territories-fill', (e) => {
      const code = e.features?.[0]?.properties?.code as string | undefined
      if (!code) return
      if (levelRef.current === 'region') onRegion.current(code)
      else onDept.current(code)
    })
    m.on('mouseenter', 'territories-fill', () => { m.getCanvas().style.cursor = 'pointer' })
    m.on('mouseleave', 'territories-fill', () => { m.getCanvas().style.cursor = '' })
    return () => { m.remove(); map.current = null }
  }, [])

  useEffect(() => {
    const m = map.current
    if (!m) return
    const dataUrl = level === 'region' ? regionsUrl : departementsUrl
    const apply = () => {
      const src = m.getSource('territories') as maplibregl.GeoJSONSource | undefined
      if (!src) {
        m.addSource('territories', { type: 'geojson', data: dataUrl })
        m.addLayer({
          id: 'territories-fill', type: 'fill', source: 'territories',
          paint: { 'fill-color': fillColorExpression(territories), 'fill-opacity': 0.7 },
        })
        m.addLayer({
          id: 'territories-line', type: 'line', source: 'territories',
          paint: { 'line-color': '#ffffff', 'line-width': 0.5 },
        })
        return
      }
      src.setData(dataUrl)
      if (m.getLayer('territories-fill')) {
        m.setPaintProperty('territories-fill', 'fill-color', fillColorExpression(territories))
      }
    }
    if (m.isStyleLoaded()) apply()
    else m.once('load', apply)
  }, [level, territories])

  return <div ref={mapRef} className="w-full h-[480px] rounded-xl overflow-hidden" />
}
