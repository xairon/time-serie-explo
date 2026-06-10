// frontend/src/components/meteo/MeteoMiniMap.tsx
// Collapsible France overview with the main map's viewport rectangle.
import { useRef, useEffect, useState } from 'react'
import maplibregl from 'maplibre-gl'

interface Props {
  mainMap: maplibregl.Map | null
}

function boundsToPolygon(b: maplibregl.LngLatBounds): GeoJSON.Feature {
  const [w, s, e, n] = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]
  return {
    type: 'Feature',
    geometry: { type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
    properties: {},
  }
}

export function MeteoMiniMap({ mainMap }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const miniRef = useRef<maplibregl.Map | null>(null)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    if (!containerRef.current || miniRef.current || collapsed) return
    const mini = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256 },
        },
        layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
      },
      center: [2.5, 46.6],
      zoom: 3.2,
      interactive: false,
      attributionControl: false,
    })
    mini.on('load', () => {
      mini.addSource('viewport', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      mini.addLayer({ id: 'viewport-fill', type: 'fill', source: 'viewport', paint: { 'fill-color': '#3b82f6', 'fill-opacity': 0.15 } })
      mini.addLayer({ id: 'viewport-line', type: 'line', source: 'viewport', paint: { 'line-color': '#3b82f6', 'line-width': 1.5 } })
      if (mainMap) {
        ;(mini.getSource('viewport') as maplibregl.GeoJSONSource)
          .setData({ type: 'FeatureCollection', features: [boundsToPolygon(mainMap.getBounds())] })
      }
    })
    miniRef.current = mini
    return () => { mini.remove(); miniRef.current = null }
  }, [collapsed, mainMap])

  useEffect(() => {
    if (!mainMap) return
    const sync = () => {
      const src = miniRef.current?.getSource('viewport') as maplibregl.GeoJSONSource | undefined
      src?.setData({ type: 'FeatureCollection', features: [boundsToPolygon(mainMap.getBounds())] })
    }
    mainMap.on('move', sync)
    return () => { mainMap.off('move', sync) }
  }, [mainMap, collapsed])

  return (
    <div className="bg-white rounded-lg shadow-md border border-slate-200 overflow-hidden">
      <button
        onClick={() => setCollapsed(c => !c)}
        aria-label={collapsed ? 'Afficher la mini-carte' : 'Masquer la mini-carte'}
        className="w-full flex items-center justify-center py-0.5 text-slate-400 hover:text-slate-600 hover:bg-slate-50"
      >
        <svg width="10" height="6" viewBox="0 0 10 6" aria-hidden="true" style={{ transform: collapsed ? 'rotate(180deg)' : undefined }}>
          <path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" />
        </svg>
      </button>
      {!collapsed && <div ref={containerRef} style={{ width: 150, height: 112 }} />}
    </div>
  )
}
