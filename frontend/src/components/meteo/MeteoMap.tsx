// frontend/src/components/meteo/MeteoMap.tsx
import { useRef, useEffect, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { StationGeoJSONFeature } from '@/lib/observatory-types'
import { addSectorLayers, updateSectorColors, updateTrendBadges, type Trend } from './layers/sectors-layer'
import { addStationLayers, setStationData, setStationVisibility, type StationType } from './layers/stations-layer'

export const FRANCE_CENTER: [number, number] = [2.5, 46.5]
export const FRANCE_ZOOM = 5.6

interface MeteoMapProps {
  sectorColorById: Record<number, string>
  sectorTrendById: Record<number, Trend>
  visibleLayers: Record<StationType, boolean>
  piezoFeatures: StationGeoJSONFeature[]
  hydroFeatures: StationGeoJSONFeature[]
  onSectorClick: (sectorId: number, name: string) => void
  onStationClick: (code: string, type: StationType) => void
  onMapReady: (map: maplibregl.Map) => void
}

export function MeteoMap({
  sectorColorById,
  sectorTrendById,
  visibleLayers,
  piezoFeatures,
  hydroFeatures,
  onSectorClick,
  onStationClick,
  onMapReady,
}: MeteoMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [sectorsReady, setSectorsReady] = useState(false)

  // Latest callbacks for map event handlers.
  const onSectorClickRef = useRef(onSectorClick); onSectorClickRef.current = onSectorClick
  const onStationClickRef = useRef(onStationClick); onStationClickRef.current = onStationClick
  const onMapReadyRef = useRef(onMapReady); onMapReadyRef.current = onMapReady

  // Sector geometry kept for badge rebuilds.
  const sectorGeoRef = useRef<GeoJSON.FeatureCollection | null>(null)

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
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 100, unit: 'metric' }), 'bottom-right')
    map.addControl(new maplibregl.AttributionControl({ compact: false }), 'bottom-left')
    map.on('error', (e) => { console.error('MapLibre error:', e.error?.message ?? e) })

    map.on('load', () => {
      addStationLayers(map, (code, type) => onStationClickRef.current?.(code, type))

      fetch('/geo/secteurs-bsh.geojson')
        .then(r => r.json())
        .then((gj: GeoJSON.FeatureCollection) => {
          if (!mapRef.current) return
          sectorGeoRef.current = gj
          addSectorLayers(map, gj, (id, name) => onSectorClickRef.current?.(id, name))
          setSectorsReady(true)
        })
        .catch(err => console.error('Failed to load sector geometry:', err))

      setMapLoaded(true)
      onMapReadyRef.current?.(map)
    })

    mapRef.current = map
    return () => { map.remove(); mapRef.current = null }
  }, [])

  // Choropleth colors + trend badges follow the selected month.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoaded || !sectorsReady) return
    updateSectorColors(m, sectorColorById)
    updateTrendBadges(m, sectorGeoRef.current?.features ?? [], sectorTrendById)
  }, [sectorColorById, sectorTrendById, mapLoaded, sectorsReady])

  // Station data.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoaded) return
    setStationData(m, 'piezo', piezoFeatures ?? [])
  }, [piezoFeatures, mapLoaded])

  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoaded) return
    setStationData(m, 'hydro', hydroFeatures ?? [])
  }, [hydroFeatures, mapLoaded])

  // Layer visibility.
  useEffect(() => {
    const m = mapRef.current
    if (!m || !mapLoaded) return
    setStationVisibility(m, 'piezo', visibleLayers.piezo)
    setStationVisibility(m, 'hydro', visibleLayers.hydro)
  }, [visibleLayers, mapLoaded])

  return <div ref={containerRef} className="absolute inset-0 w-full h-full" role="application" aria-label="Carte météo des nappes" />
}
