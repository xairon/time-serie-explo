import { useState, useCallback, useMemo, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { ObservatoryMap } from '@/components/observatory/ObservatoryMap'
import { StationDrawer } from '@/components/observatory/StationDrawer'
import { KPIBar } from '@/components/observatory/KPIBar'
import { SearchBar } from '@/components/observatory/SearchBar'
import type { SearchAction } from '@/components/observatory/SearchBar'
import { TimelineSlider } from '@/components/observatory/TimelineSlider'
import { RightDrawer } from '@/components/observatory/RightDrawer'
import { Era5Banner } from '@/components/observatory/Era5Banner'
import { useStationsGeoJSON, useWfsLayer, useObsFilters, useSectorSituation, useSectorTimeline, useERA5Range, useERA5Snapshot, useERA5Anomaly, useERA5Monthly } from '@/hooks/useObservatory'
import type { Era5Variable } from '@/lib/era5-colors'
import type { StationGeoJSONFeature, WfsLayerId, ClassificationTimeline, SituationClass } from '@/lib/observatory-types'
import { TIMELINE_CLASSIFICATIONS } from '@/lib/observatory-types'
import { isStationActive, ACTIVE_FILTER_DAYS } from '@/lib/observatory-utils'

type Bbox = [number, number, number, number]

interface GeoJSONGeometry {
  type: string
  coordinates: number[] | number[][] | number[][][] | number[][][][]
}

interface GeoJSONFeature {
  type: 'Feature'
  geometry: GeoJSONGeometry
  properties: Record<string, unknown>
}

interface GeoJSONCollection {
  type: 'FeatureCollection'
  features: GeoJSONFeature[]
}

function computeBboxFromGeometry(geometry: GeoJSONGeometry): Bbox {
  const coords: number[][] = []
  const collect = (g: any) => { if (!g) return; switch (g.type) { case 'Point': coords.push(g.coordinates); break; case 'LineString': g.coordinates.forEach((c: number[]) => coords.push(c)); break; case 'MultiLineString': g.coordinates.forEach((line: number[][]) => line.forEach((c: number[]) => coords.push(c))); break; case 'Polygon': g.coordinates[0].forEach((c: number[]) => coords.push(c)); break; case 'MultiPolygon': g.coordinates.forEach((p: number[][][]) => p[0].forEach((c: number[]) => coords.push(c))); break } }
  collect(geometry)
  if (coords.length === 0) return [-5, 41, 10, 51]
  // Coerce to numbers (geojson coords may arrive as strings) and drop invalid
  // vertices so one bad coordinate can't poison the bbox; if none remain, return
  // NaN bounds so the map skips the fly instead of crashing on fitBounds.
  const valid = coords
    .map(c => [Number(c?.[0]), Number(c?.[1])] as [number, number])
    .filter(c => Number.isFinite(c[0]) && Number.isFinite(c[1]))
  if (valid.length === 0) return [NaN, NaN, NaN, NaN]
  if (valid.length === 1) { const [lon, lat] = valid[0]; return [lon - 0.15, lat - 0.1, lon + 0.15, lat + 0.1] }
  const lons = valid.map(c => c[0]); const lats = valid.map(c => c[1])
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)]
}

function featureIntersectsBbox(feature: GeoJSONFeature, bbox: Bbox): boolean {
  const [minLon, minLat, maxLon, maxLat] = bbox; const coords: number[][] = []
  const collectCoords = (geom: any) => { if (!geom) return; switch (geom.type) { case 'Point': coords.push(geom.coordinates); break; case 'LineString': case 'MultiPoint': geom.coordinates.forEach((c: number[]) => coords.push(c)); break; case 'Polygon': case 'MultiLineString': geom.coordinates.forEach((ring: number[][]) => ring.forEach((c: number[]) => coords.push(c))); break; case 'MultiPolygon': geom.coordinates.forEach((poly: number[][][]) => poly.forEach((ring: number[][]) => ring.forEach((c: number[]) => coords.push(c)))); break } }
  collectCoords(feature.geometry)
  const margin = 0.1; return coords.some(c => c[0] >= minLon - margin && c[0] <= maxLon + margin && c[1] >= minLat - margin && c[1] <= maxLat + margin)
}

function filterGeoJSONByBbox(geojson: GeoJSONCollection | null, bbox: Bbox): GeoJSONCollection | null {
  if (!geojson?.features) return geojson
  return { ...geojson, features: geojson.features.filter((f) => featureIntersectsBbox(f, bbox)) }
}

function pointInRing(x: number, y: number, ring: number[][]): boolean { let inside = false; for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) { const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1]; if (((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)) inside = !inside }; return inside }
function pointInGeometry(lon: number, lat: number, geometry: GeoJSONGeometry): boolean {
  const coords = geometry.coordinates as any // eslint-disable-line @typescript-eslint/no-explicit-any -- runtime type-checked via geometry.type
  if (geometry.type === 'Polygon') { const [outer, ...holes] = coords as number[][][]; if (!pointInRing(lon, lat, outer)) return false; for (const hole of holes) { if (pointInRing(lon, lat, hole)) return false }; return true }
  if (geometry.type === 'MultiPolygon') { return (coords as number[][][][]).some((poly: number[][][]) => { const [outer, ...holes] = poly; if (!pointInRing(lon, lat, outer)) return false; for (const hole of holes) { if (pointInRing(lon, lat, hole)) return false }; return true }) }
  return false
}
function stationsInGeometry(features: StationGeoJSONFeature[], geometry: GeoJSONGeometry): string[] { return features.filter(f => { const [lon, lat] = f.geometry.coordinates; return lon != null && lat != null && pointInGeometry(lon, lat, geometry) }).map(f => f.properties.code) }

export default function ObservatoryPage() {
  const { t } = useTranslation()
  const { filters, setFilter } = useObsFilters()
  const [searchParams, setSearchParams] = useSearchParams()
  const { data: geojsonData, isError: geojsonError } = useStationsGeoJSON()
  const [spatialStationCodes, setSpatialStationCodes] = useState<string[] | null>(null)
  const [selectedStation, setSelectedStation] = useState<{ code: string; type: 'piezo' | 'hydro' } | null>(null)
  const [showPiezo, setShowPiezo] = useState(true); const [showHydro, setShowHydro] = useState(true)
  const [showExcluded, setShowExcluded] = useState(false); const [showTerrain, setShowTerrain] = useState(false)

  const [era5Active, setEra5Active] = useState(false)
  const [era5Variable, setEra5Variable] = useState<Era5Variable>('precipAnomaly')
  const [era5Date, setEra5Date] = useState<string>('')
  const [era5Window, setEra5Window] = useState(3)
  const { data: era5Range } = useERA5Range(era5Active)
  useEffect(() => { if (era5Range?.max_date && !era5Date) setEra5Date(era5Range.max_date) }, [era5Range, era5Date])

  const filteredFeatures = useMemo<StationGeoJSONFeature[]>(() => {
    const all = geojsonData?.features ?? []
    return all.filter(f => {
      if (f.properties.type === 'piezo' && !showPiezo) return false; if (f.properties.type === 'hydro' && !showHydro) return false
      if (filters.activeOnly && !isStationActive(f.properties.derniere_mesure, ACTIVE_FILTER_DAYS)) return false
      if (filters.codeDepartement && f.properties.code_departement !== filters.codeDepartement) return false
      if (filters.classification?.length && !filters.classification.includes(f.properties.classification ?? '')) return false
      if (filters.codeBdlisa && f.properties.type === 'piezo') { const codes = f.properties.codes_bdlisa ?? ''; if (!codes.startsWith(filters.codeBdlisa)) return false }
      if (filters.lastMeasurementAfter && f.properties.derniere_mesure) { if (f.properties.derniere_mesure < filters.lastMeasurementAfter) return false }
      if (filters.minObservations && (f.properties.nb_observations ?? 0) < filters.minObservations) return false
      const anyReliabilityActive = filters.showFiable || filters.showIndicatif || filters.showInsuffisant
      if (anyReliabilityActive) { const fiab = f.properties.fiabilite ?? 'insuffisant'; if (fiab === 'fiable' && !filters.showFiable) return false; if (fiab === 'indicatif' && !filters.showIndicatif) return false; if (fiab === 'insuffisant' && !filters.showInsuffisant) return false }
      if (spatialStationCodes?.length) { if (!spatialStationCodes.includes(f.properties.code)) return false }
      return true
    })
  }, [geojsonData, showPiezo, showHydro, filters, spatialStationCodes])

  const [activeZoneLayer, setActiveZoneLayer] = useState<string | null>('regions')
  const [era5ByZone, setEra5ByZone] = useState(true)
  useEffect(() => {
    if (era5ByZone && !['depts', 'regions', 'her', 'bassins', 'secteurs'].includes(activeZoneLayer ?? '')) {
      setActiveZoneLayer('depts')
    }
  }, [era5ByZone]) // eslint-disable-line react-hooks/exhaustive-deps
  const [overlayLayers, setOverlayLayers] = useState<Set<string>>(new Set())
  const [activeBbox, setActiveBbox] = useState<Bbox | null>(null)
  const [flyToBbox, setFlyToBbox] = useState<Bbox | null>(null)

  const showRegions = activeZoneLayer === 'regions'; const showDepts = activeZoneLayer === 'depts'; const showSandre = activeZoneLayer === 'bassins'; const showHER = activeZoneLayer === 'her'
  const showSectors = activeZoneLayer === 'secteurs'
  const sectorType: 'piezo' | 'hydro' = showHydro && !showPiezo ? 'hydro' : 'piezo'
  const { data: sectorSituationData } = useSectorSituation(sectorType, showSectors)
  const { data: sectorTimelineData } = useSectorTimeline(sectorType, showSectors)
  const activeWfsLayers = useMemo(() => { const s = new Set<WfsLayerId>(); const wfsZones: WfsLayerId[] = ['region-hydro', 'secteur-hydro', 'sous-secteur-hydro', 'zone-hydro']; if (activeZoneLayer && wfsZones.includes(activeZoneLayer as WfsLayerId)) s.add(activeZoneLayer as WfsLayerId); overlayLayers.forEach(id => s.add(id as WfsLayerId)); return s }, [activeZoneLayer, overlayLayers])

  useEffect(() => {
    const lat = parseFloat(searchParams.get('lat') ?? ''); const lon = parseFloat(searchParams.get('lon') ?? ''); const zoom = parseFloat(searchParams.get('zoom') ?? '')
    if (!isNaN(lat) && !isNaN(lon)) { const delta = zoom > 10 ? 0.05 : 0.15; setFlyToBbox([lon - delta, lat - delta, lon + delta, lat + delta]); setSearchParams(prev => { const next = new URLSearchParams(prev); next.delete('lat'); next.delete('lon'); next.delete('zoom'); return next }, { replace: true }) }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const [timelinePeriodIndex, setTimelinePeriodIndex] = useState<number | null>(null)
  const [timelineData, setTimelineData] = useState<ClassificationTimeline | null>(null)

  const timelineMonth = (timelinePeriodIndex != null && timelineData)
    ? timelineData.periods[timelinePeriodIndex] + '-01'
    : null
  const isAnomalyVar = era5Variable === 'anomaly' || era5Variable === 'precipAnomaly'
  const anomalyApiVar = era5Variable === 'precipAnomaly' ? 'precipitation' : 'temperature'
  // daily snapshot ONLY when independent (timeline closed) and not anomaly
  const { data: era5SnapshotPoints } = useERA5Snapshot(
    era5Active && !timelineMonth && era5Date && !isAnomalyVar ? era5Date : undefined
  )
  // monthly aggregate when timeline-driven and not anomaly
  const { data: era5MonthlyPoints } = useERA5Monthly(
    era5Active && timelineMonth && !isAnomalyVar ? timelineMonth : undefined
  )
  const era5Points = timelineMonth ? era5MonthlyPoints : era5SnapshotPoints
  // anomaly: month from timeline if present, else the independent month picker
  const anomalyMonth = timelineMonth ?? (era5Date ? era5Date.slice(0, 7) + '-01' : era5Date)
  const { data: era5AnomalyPoints } = useERA5Anomaly(
    anomalyApiVar, anomalyMonth, era5Window, era5Active && isAnomalyVar
  )

  const handleOverlayToggle = useCallback((id: string) => { setOverlayLayers(prev => { const next = new Set(prev); if (next.has(id)) next.delete(id); else next.add(id); return next }) }, [])

  const regionHydro = useWfsLayer('region-hydro', activeWfsLayers.has('region-hydro')); const secteurHydro = useWfsLayer('secteur-hydro', activeWfsLayers.has('secteur-hydro'))
  const sousSecteurHydro = useWfsLayer('sous-secteur-hydro', activeWfsLayers.has('sous-secteur-hydro')); const zoneHydro = useWfsLayer('zone-hydro', activeWfsLayers.has('zone-hydro'))
  const coursEau1 = useWfsLayer('cours-eau-1', activeWfsLayers.has('cours-eau-1')); const coursEau2 = useWfsLayer('cours-eau-2', activeWfsLayers.has('cours-eau-2'))
  const planEau = useWfsLayer('plan-eau', activeWfsLayers.has('plan-eau')); const masseEauRiv = useWfsLayer('masse-eau-riv', activeWfsLayers.has('masse-eau-riv'))

  const wfsDataAll = useMemo(() => {
    const raw: Record<string, any> = {}
    if (regionHydro.data) raw['region-hydro'] = regionHydro.data; if (secteurHydro.data) raw['secteur-hydro'] = secteurHydro.data
    if (sousSecteurHydro.data) raw['sous-secteur-hydro'] = sousSecteurHydro.data; if (zoneHydro.data) raw['zone-hydro'] = zoneHydro.data
    if (coursEau1.data) raw['cours-eau-1'] = coursEau1.data; if (coursEau2.data) raw['cours-eau-2'] = coursEau2.data
    if (planEau.data) raw['plan-eau'] = planEau.data; if (masseEauRiv.data) raw['masse-eau-riv'] = masseEauRiv.data
    return raw
  }, [regionHydro.data, secteurHydro.data, sousSecteurHydro.data, zoneHydro.data, coursEau1.data, coursEau2.data, planEau.data, masseEauRiv.data])

  const wfsData = useMemo(() => { if (!activeBbox) return wfsDataAll; const filtered: Record<string, any> = {}; for (const [key, data] of Object.entries(wfsDataAll)) filtered[key] = filterGeoJSONByBbox(data, activeBbox); return filtered }, [wfsDataAll, activeBbox])

  const handleStationClick = useCallback((code: string, type: 'piezo' | 'hydro') => { setSelectedStation(prev => prev?.code === code && prev?.type === type ? null : { code, type }); setSpatialStationCodes(null); setActiveBbox(null) }, [])
  const handleEmptyClick = useCallback(() => { setSelectedStation(null); setSpatialStationCodes(null); setActiveBbox(null) }, [])
  const handleDeptClick = useCallback((code: string | null) => { setSelectedStation(null); setSpatialStationCodes(null); setFilter('dept', code ?? undefined); if (!code) setActiveBbox(null) }, [setFilter])
  const handleBassinClick = useCallback((code: string | null) => { setSelectedStation(null); setSpatialStationCodes(null); setFilter('bassin', code ?? undefined); if (!code) setActiveBbox(null) }, [setFilter])
  const handleSpatialFilter = useCallback((codes: string[] | null) => { setSelectedStation(null); setSpatialStationCodes(codes); if (!codes) setActiveBbox(null) }, [])
  const handleSectorClick = useCallback((codes: string[] | null, _name: string | null) => { setSelectedStation(null); setSpatialStationCodes(codes); if (!codes) setActiveBbox(null) }, [])
  const handleBboxChange = useCallback((bbox: Bbox | null) => { setActiveBbox(bbox) }, [])
  const handleTimelinePeriodChange = useCallback((periodIndex: number | null, timeline: ClassificationTimeline | null) => { setTimelinePeriodIndex(periodIndex); setTimelineData(timeline) }, [])

  const displayFeatures = useMemo<StationGeoJSONFeature[]>(() => {
    if (timelinePeriodIndex == null || !timelineData) return filteredFeatures
    const all = geojsonData?.features ?? []
    return all.filter(f => {
      const arr = timelineData.stations[f.properties.code]; if (!arr) return false; const clsIdx = arr[timelinePeriodIndex]; if (clsIdx === 7 || clsIdx == null) return false
      if (f.properties.type === 'piezo' && !showPiezo) return false; if (f.properties.type === 'hydro' && !showHydro) return false
      if (filters.codeDepartement && f.properties.code_departement !== filters.codeDepartement) return false
      if (filters.codeBdlisa && f.properties.type === 'piezo') { const codes = f.properties.codes_bdlisa ?? ''; if (!codes.startsWith(filters.codeBdlisa)) return false }
      if (spatialStationCodes?.length) { if (!spatialStationCodes.includes(f.properties.code)) return false }
      if (filters.classification?.length) { const cls = TIMELINE_CLASSIFICATIONS[clsIdx]; if (!filters.classification.includes(cls)) return false }
      return true
    }).map(f => { const cls = TIMELINE_CLASSIFICATIONS[timelineData.stations[f.properties.code][timelinePeriodIndex]]; return { ...f, properties: { ...f.properties, classification: cls === 'UNKNOWN' ? null : cls } } })
  }, [filteredFeatures, timelinePeriodIndex, timelineData, geojsonData, showPiezo, showHydro, filters, spatialStationCodes])

  const displaySectorSituation = useMemo(() => {
    const base = sectorSituationData ?? []
    if (timelinePeriodIndex == null || !timelineData || !sectorTimelineData) return base
    const period = timelineData.periods[timelinePeriodIndex]
    const sIdx = sectorTimelineData.periods.indexOf(period)
    // During historical replay a period with no sector data must read as
    // "insufficient" (grey), never fall back to the current-month colours.
    if (sIdx < 0) return base.map(s => ({ ...s, situation_class: null, trend: null, insufficient: true }))
    const CLS = ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT'] as const
    const TR: Record<number, 'baisse' | 'stable' | 'hausse'> = { [-1]: 'baisse', [0]: 'stable', [1]: 'hausse' }
    return base.map(s => {
      const ci = sectorTimelineData.sectors[s.code]?.[sIdx]
      const ti = sectorTimelineData.trends[s.code]?.[sIdx]
      const insufficient = ci == null || ci === 7
      return { ...s, situation_class: insufficient ? null : (CLS[ci] as SituationClass), trend: (ti != null ? TR[ti] : null), insufficient }
    })
  }, [sectorSituationData, sectorTimelineData, timelinePeriodIndex, timelineData])

  const excludedFeatures = useMemo<StationGeoJSONFeature[]>(() => {
    const activeSet = new Set(displayFeatures.map(f => f.properties.code)); const all = geojsonData?.features ?? []
    if (timelinePeriodIndex != null && timelineData) {
      if (!filters.activeOnly) return []
      return all.filter(f => { if (activeSet.has(f.properties.code)) return false; if (f.properties.type === 'piezo' && !showPiezo) return false; if (f.properties.type === 'hydro' && !showHydro) return false; const arr = timelineData.stations[f.properties.code]; if (!arr) return false; let existed = false; for (let i = 0; i < timelinePeriodIndex; i++) { if (arr[i] != null && arr[i] !== 7) { existed = true; break } }; if (!existed) return false; if (filters.codeDepartement && f.properties.code_departement !== filters.codeDepartement) return false; if (spatialStationCodes?.length) { if (!spatialStationCodes.includes(f.properties.code)) return false }; return true })
    }
    return all.filter(f => { if (activeSet.has(f.properties.code)) return false; if (f.properties.type === 'piezo' && !showPiezo) return false; if (f.properties.type === 'hydro' && !showHydro) return false; if (filters.codeDepartement && f.properties.code_departement !== filters.codeDepartement) return false; if (filters.classification?.length && !filters.classification.includes(f.properties.classification ?? '')) return false; if (spatialStationCodes?.length) { if (!spatialStationCodes.includes(f.properties.code)) return false }; return true })
  }, [displayFeatures, geojsonData, showPiezo, showHydro, filters, spatialStationCodes, timelinePeriodIndex, timelineData])

  const handleSearchAction = useCallback((action: SearchAction) => {
    switch (action.kind) {
      case 'station': if (action.stationType === 'piezo') setShowPiezo(true); if (action.stationType === 'hydro') setShowHydro(true); setSelectedStation({ code: action.code, type: action.stationType! }); if (action.geometry) setFlyToBbox(computeBboxFromGeometry(action.geometry)); break
      case 'department': setSelectedStation(null); setSpatialStationCodes(null); setActiveZoneLayer('depts'); setFilter('dept', action.code); if (action.geometry) { const bbox = computeBboxFromGeometry(action.geometry); setActiveBbox(bbox); setFlyToBbox(bbox) }; break
      case 'region': { setSelectedStation(null); setActiveZoneLayer('regions'); if (action.geometry) { const bbox = computeBboxFromGeometry(action.geometry); setActiveBbox(bbox); setFlyToBbox(bbox); const codes = stationsInGeometry(geojsonData?.features ?? [], action.geometry); setSpatialStationCodes(codes.length > 0 ? codes : null) }; break }
      case 'bassin': { setSelectedStation(null); setActiveZoneLayer('bassins'); if (action.geometry) { const bbox = computeBboxFromGeometry(action.geometry); setActiveBbox(bbox); setFlyToBbox(bbox); const codes = stationsInGeometry(geojsonData?.features ?? [], action.geometry); setSpatialStationCodes(codes.length > 0 ? codes : null); setFilter('bassin', action.code) }; break }
      case 'her': { setSelectedStation(null); setActiveZoneLayer('her'); if (action.geometry) { const bbox = computeBboxFromGeometry(action.geometry); setActiveBbox(bbox); setFlyToBbox(bbox); const codes = stationsInGeometry(geojsonData?.features ?? [], action.geometry); setSpatialStationCodes(codes.length > 0 ? codes : null) }; break }
      case 'wfs': if (action.wfsLayerId) { const wfsZones = ['region-hydro', 'secteur-hydro', 'sous-secteur-hydro', 'zone-hydro']; if (wfsZones.includes(action.wfsLayerId)) setActiveZoneLayer(action.wfsLayerId); else setOverlayLayers(prev => new Set(prev).add(action.wfsLayerId!)) }; if (action.geometry) setFlyToBbox(computeBboxFromGeometry(action.geometry)); break
    }
  }, [setFilter, geojsonData])

  const stationCounts = useMemo(() => {
    const all = geojsonData?.features ?? []
    return {
      filteredPiezo: showPiezo ? filteredFeatures.filter(f => f.properties.type === 'piezo').length : 0,
      filteredHydro: showHydro ? filteredFeatures.filter(f => f.properties.type === 'hydro').length : 0,
      totalPiezo: showPiezo ? all.filter(f => f.properties.type === 'piezo').length : 0,
      totalHydro: showHydro ? all.filter(f => f.properties.type === 'hydro').length : 0,
    }
  }, [filteredFeatures, geojsonData, showPiezo, showHydro])

  return (
    <div className="relative h-full">
      {geojsonError && (<div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 bg-red-900/90 text-red-200 px-4 py-2 rounded-lg text-sm">{t('mainPages.observatory.loadError')} <button onClick={() => window.location.reload()} className="underline ml-2">{t('mainPages.observatory.retry')}</button></div>)}
      <ObservatoryMap features={displayFeatures} excludedFeatures={showExcluded ? excludedFeatures : []} allFeatures={geojsonData?.features} showPiezo={showPiezo} showHydro={showHydro} onStationClick={handleStationClick} onEmptyClick={handleEmptyClick} onDeptClick={handleDeptClick} activeCodeDepartement={filters.codeDepartement} showRegions={showRegions} showDepts={showDepts} showHER={showHER} showSandre={showSandre} onBassinClick={handleBassinClick} activeCodeBassin={filters.codeBassin} onSpatialFilter={handleSpatialFilter} onBboxChange={handleBboxChange} activeWfsLayers={activeWfsLayers} wfsData={wfsData} selectedStationCode={selectedStation?.code ?? null} showTerrain={showTerrain} flyToBbox={flyToBbox} onFlyToComplete={() => setFlyToBbox(null)} showSectors={showSectors} sectorSituation={displaySectorSituation} onSectorClick={handleSectorClick} stationsInGeometry={(geom) => stationsInGeometry(geojsonData?.features ?? [], geom)}
        era5Active={era5Active} era5Points={era5Points} era5Variable={era5Variable} era5AnomalyPoints={era5AnomalyPoints} era5Window={era5Window} era5ByZone={era5ByZone} era5Granularity={timelineMonth != null ? 'monthly' : 'daily'} />
      <Era5Banner era5Active={era5Active} era5Variable={era5Variable} era5Window={era5Window} era5Period={timelineMonth ?? era5Date} era5Granularity={timelineMonth != null ? 'monthly' : 'daily'} />
      <SearchBar features={geojsonData?.features} wfsData={wfsDataAll} onSearchAction={handleSearchAction} />
      <RightDrawer showPiezo={showPiezo} setShowPiezo={setShowPiezo} showHydro={showHydro} setShowHydro={setShowHydro} showExcluded={showExcluded} setShowExcluded={setShowExcluded} showTerrain={showTerrain} setShowTerrain={setShowTerrain} filters={filters} setFilter={setFilter} filteredPiezo={stationCounts.filteredPiezo} totalPiezo={stationCounts.totalPiezo} filteredHydro={stationCounts.filteredHydro} totalHydro={stationCounts.totalHydro} activeZoneLayer={activeZoneLayer} onZoneLayerChange={setActiveZoneLayer} overlayLayers={overlayLayers} onOverlayToggle={handleOverlayToggle} onResetSpatial={() => { setSpatialStationCodes(null); setActiveBbox(null) }} hasSpatialFilter={spatialStationCodes != null && spatialStationCodes.length > 0}
        era5Active={era5Active} setEra5Active={setEra5Active} era5Variable={era5Variable} setEra5Variable={setEra5Variable} era5Date={era5Date} setEra5Date={setEra5Date} era5EffectiveDate={timelineMonth ?? era5Date} era5MinDate={era5Range?.min_date} era5MaxDate={era5Range?.max_date} era5Window={era5Window} setEra5Window={setEra5Window} era5ByZone={era5ByZone} setEra5ByZone={setEra5ByZone} era5TimelineDriven={timelineMonth != null} era5Granularity={timelineMonth != null ? 'monthly' : 'daily'} />
      {selectedStation && <StationDrawer code={selectedStation.code} type={selectedStation.type} onClose={() => setSelectedStation(null)} />}
      <TimelineSlider onPeriodChange={handleTimelinePeriodChange} showMeteo={era5Active} onMeteoChange={setEra5Active} showHydro={showHydro} onHydroChange={setShowHydro} showPiezo={showPiezo} onPiezoChange={setShowPiezo} />
      <KPIBar filteredPiezo={stationCounts.filteredPiezo} filteredHydro={stationCounts.filteredHydro} totalPiezo={stationCounts.totalPiezo} totalHydro={stationCounts.totalHydro} />
    </div>
  )
}
