// frontend/src/pages/MeteoNappesPage.tsx
// Clone fidèle MétéEAU Nappes — données Junon (IPS réf. fixe 1991-2020).
import { useState, useMemo, useCallback } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import type maplibregl from 'maplibre-gl'
import { MeteoMap, FRANCE_CENTER, FRANCE_ZOOM } from '@/components/meteo/MeteoMap'
import { MeteoTypePanel } from '@/components/meteo/MeteoTypePanel'
import { MeteoLegend } from '@/components/meteo/MeteoLegend'
import { MeteoTimeline } from '@/components/meteo/MeteoTimeline'
import { MeteoSearchBar, type SearchTarget } from '@/components/meteo/MeteoSearchBar'
import { MeteoMiniMap } from '@/components/meteo/MeteoMiniMap'
import { AboutModal } from '@/components/meteo/AboutModal'
import { SectorPopup } from '@/components/meteo/SectorPopup'
import { StationPopup } from '@/components/meteo/StationPopup'
import type { StationType } from '@/components/meteo/layers/stations-layer'
import type { Trend } from '@/components/meteo/layers/sectors-layer'
import { useSectorSituation, useSectorTimeline, useStationsGeoJSON } from '@/hooks/useObservatory'
import { meteoClassColor, METEO_CLASS_LABELS } from '@/lib/meteo-colors'
import { SECTOR_INSUFFICIENT_COLOR } from '@/lib/sector-arrows'
import type { SectorSituation, SituationClass, StationGeoJSONFeature } from '@/lib/observatory-types'

// Timeline payload class indices → enums (index 7 / null = insufficient).
const CLS = ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT'] as const
const TR: Record<number, 'baisse' | 'stable' | 'hausse'> = { [-1]: 'baisse', [0]: 'stable', [1]: 'hausse' }

function capitalize(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s
}

export default function MeteoNappesPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [visible, setVisible] = useState<Record<StationType, boolean>>({ piezo: true, hydro: true })
  const [selectedSectorId, setSelectedSectorId] = useState<number | null>(null)
  const [selectedSectorName, setSelectedSectorName] = useState<string | null>(null)
  const [selectedStation, setSelectedStation] = useState<{ code: string; type: StationType } | null>(null)
  const [searchLabel, setSearchLabel] = useState<string | null>(null)
  const [aboutOpen, setAboutOpen] = useState(false)
  const [map, setMap] = useState<maplibregl.Map | null>(null)

  // Réseau officiel MétéEAU Nappes (450 stations du bulletin BRGM) pour coller aux cartes officielles
  const { data: sectorSituationData } = useSectorSituation('piezo', true, 'meteeau')
  const { data: timeline } = useSectorTimeline('piezo', true, 'meteeau')
  const { data: geojsonData } = useStationsGeoJSON()

  // Sector names come from the static geometry.
  const { data: sectorGeo } = useQuery({
    queryKey: ['secteurs-bsh-geo'],
    queryFn: () => fetch('/geo/secteurs-bsh.geojson').then((r) => r.json()),
    staleTime: Infinity,
  })
  const nameById: Record<number, string> = useMemo(
    () => Object.fromEntries((sectorGeo?.features ?? []).map((f: { properties: { sector_id: number; nom: string } }) => [f.properties.sector_id, f.properties.nom])),
    [sectorGeo],
  )

  const piezoFeatures = useMemo<StationGeoJSONFeature[]>(
    () => (geojsonData?.features ?? []).filter((f) => f.properties.type === 'piezo'),
    [geojsonData],
  )
  const hydroFeatures = useMemo<StationGeoJSONFeature[]>(
    () => (geojsonData?.features ?? []).filter((f) => f.properties.type === 'hydro'),
    [geojsonData],
  )
  const allStations = useMemo(
    () => piezoFeatures.concat(hydroFeatures),
    [piezoFeatures, hydroFeatures],
  )

  // Selected month: ?month= param, else latest available.
  const periods = timeline?.periods ?? []
  const latest = periods.length ? periods[periods.length - 1] : null
  const urlMonth = searchParams.get('month')
  const effectivePeriod = (urlMonth && periods.includes(urlMonth)) ? urlMonth : latest

  const setPeriod = useCallback((p: string) => {
    setSearchParams(prev => {
      const next = new URLSearchParams(prev)
      if (latest && p === latest) next.delete('month')
      else next.set('month', p)
      return next
    }, { replace: true })
  }, [setSearchParams, latest])

  // Recolor for the selected month from the timeline payload (no refetch).
  const displaySectorSituation = useMemo<SectorSituation[]>(() => {
    const base = sectorSituationData ?? []
    if (!timeline || effectivePeriod == null || effectivePeriod === latest) return base
    const sIdx = timeline.periods.indexOf(effectivePeriod)
    if (sIdx < 0) return base
    return base.map((s) => {
      const ci = timeline.sectors[s.code]?.[sIdx]
      const ti = timeline.trends[s.code]?.[sIdx]
      const insufficient = ci == null || ci === 7
      return {
        ...s,
        situation_class: insufficient ? null : (CLS[ci] as SituationClass),
        // No trend for insufficient sectors — the timeline backend always emits
        // code 0 ('stable'), which would render a misleading arrow next to "no data".
        trend: insufficient ? null : (ti != null ? TR[ti] : null),
        insufficient,
      }
    })
  }, [sectorSituationData, timeline, effectivePeriod, latest])

  const { sectorColorById, sectorTrendById } = useMemo(() => {
    const colorById: Record<number, string> = {}
    const trendById: Record<number, Trend> = {}
    for (const s of displaySectorSituation) {
      const sid = Number(s.code)
      colorById[sid] = s.insufficient ? SECTOR_INSUFFICIENT_COLOR : meteoClassColor(s.situation_class)
      trendById[sid] = s.trend
    }
    return { sectorColorById: colorById, sectorTrendById: trendById }
  }, [displaySectorSituation])

  const onSectorClick = useCallback((id: number, name: string) => {
    setSelectedSectorId(id)
    setSelectedSectorName(name)
    setSelectedStation(null)
  }, [])

  const onStationClick = useCallback((code: string, type: StationType) => {
    setSelectedStation({ code, type })
    setSelectedSectorId(null)
  }, [])

  const onToggle = useCallback((k: StationType) => {
    setVisible((v) => ({ ...v, [k]: !v[k] }))
  }, [])

  const onSearchSelect = useCallback((t: SearchTarget) => {
    setSearchLabel(t.label)
    map?.flyTo({ center: [t.lng, t.lat], zoom: t.zoom, duration: 1200 })
  }, [map])

  const onSearchReset = useCallback(() => {
    setSearchLabel(null)
    map?.flyTo({ center: FRANCE_CENTER, zoom: FRANCE_ZOOM, duration: 1000 })
  }, [map])

  const selectedIps = useMemo<SectorSituation | null>(() => {
    if (selectedSectorId == null) return null
    return displaySectorSituation.find((s) => Number(s.code) === selectedSectorId) ?? null
  }, [displaySectorSituation, selectedSectorId])

  // When replaying a past month, per-station classification for that month is not
  // available (the timeline is per-sector only), so neutralize the marker classes/popups.
  // Otherwise the markers keep their latest-month colour on top of a historically
  // recoloured sector map — a visibly inconsistent carte.
  const isHistorical = effectivePeriod != null && effectivePeriod !== latest
  const neutralize = useCallback(
    (fs: StationGeoJSONFeature[]) =>
      isHistorical ? fs.map((f) => ({ ...f, properties: { ...f.properties, classification: null } })) : fs,
    [isHistorical],
  )
  const displayPiezoFeatures = useMemo(() => neutralize(piezoFeatures), [neutralize, piezoFeatures])
  const displayHydroFeatures = useMemo(() => neutralize(hydroFeatures), [neutralize, hydroFeatures])

  const selectedStationFeature = useMemo<StationGeoJSONFeature | null>(() => {
    if (!selectedStation) return null
    const pool = selectedStation.type === 'piezo' ? displayPiezoFeatures : displayHydroFeatures
    return pool.find((f) => f.properties.code === selectedStation.code) ?? null
  }, [selectedStation, displayPiezoFeatures, displayHydroFeatures])

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-slate-100">
      <MeteoMap
        sectorColorById={sectorColorById}
        sectorTrendById={sectorTrendById}
        visibleLayers={visible}
        piezoFeatures={displayPiezoFeatures}
        hydroFeatures={displayHydroFeatures}
        onSectorClick={onSectorClick}
        onStationClick={onStationClick}
        onMapReady={setMap}
      />

      {/* Search — top-left */}
      <div className="absolute top-3 left-3 z-20">
        <MeteoSearchBar stations={allStations} onSelect={onSearchSelect} />
      </div>

      {/* Reset chip — top-center, after a search */}
      {searchLabel && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20">
          <button
            onClick={onSearchReset}
            className="flex items-center gap-1.5 bg-slate-800 text-white text-xs rounded px-2.5 py-1.5 shadow-md hover:bg-slate-700"
          >
            réinitialiser <span aria-hidden="true">×</span>
          </button>
        </div>
      )}

      {/* À propos — top-right */}
      <div className="absolute top-3 right-3 z-20">
        <button
          onClick={() => setAboutOpen(true)}
          className="bg-white text-xs text-slate-600 rounded-full shadow-md border border-slate-200 px-3.5 py-2 hover:bg-slate-50"
        >
          À propos
        </button>
      </div>

      {/* Left panels: Type + legend */}
      <div className="absolute left-3 top-1/2 -translate-y-1/2 z-10 space-y-2">
        <MeteoTypePanel visible={visible} onToggle={onToggle} />
        <MeteoLegend />
      </div>

      {/* Junon logo → back to the app (where the original puts its own logo) */}
      <Link
        to="/"
        className="absolute bottom-14 left-1/2 -translate-x-1/2 z-10 bg-white/90 rounded-full shadow border border-slate-200 px-3 py-1 text-xs font-bold text-slate-700 hover:bg-white"
      >
        Junon
      </Link>

      {/* Minimap — bottom-right above the timeline */}
      <div className="absolute bottom-14 right-3 z-10">
        <MeteoMiniMap mainMap={map} />
      </div>

      {/* Timeline — bottom */}
      {periods.length > 0 && effectivePeriod && (
        <MeteoTimeline periods={periods} selected={effectivePeriod} onChange={setPeriod} />
      )}

      {/* Popups (one at a time) */}
      {selectedIps && (
        <div className="absolute top-16 right-4 z-20">
          <SectorPopup
            name={selectedSectorName ?? nameById[Number(selectedIps.code)] ?? selectedIps.name}
            code={selectedIps.code}
            classLabel={
              selectedIps.insufficient || !selectedIps.situation_class
                ? 'Données insuffisantes'
                : capitalize(METEO_CLASS_LABELS[selectedIps.situation_class] ?? '')
            }
            trend={selectedIps.trend}
            colorHex={selectedIps.insufficient ? SECTOR_INSUFFICIENT_COLOR : meteoClassColor(selectedIps.situation_class)}
            metrics={
              // pct/counts describe the latest month only; the timeline recolor
              // doesn't carry per-month metrics, so hide them for past months
              // rather than pin the current month's numbers to a historical verdict.
              effectivePeriod === latest
                ? {
                    pctBelowNormal: selectedIps.pct_below_normal,
                    nEligible: selectedIps.n_eligible,
                    nProvisoire: selectedIps.n_provisoire,
                  }
                : undefined
            }
            onClose={() => { setSelectedSectorId(null); setSelectedSectorName(null) }}
          />
        </div>
      )}

      {selectedStationFeature && (
        <div className="absolute top-16 right-4 z-20">
          <StationPopup
            code={selectedStationFeature.properties.code}
            commune={selectedStationFeature.properties.commune ?? undefined}
            classification={selectedStationFeature.properties.classification}
            type={selectedStation?.type}
            derniereMesure={selectedStationFeature.properties.derniere_mesure}
            onClose={() => setSelectedStation(null)}
          />
        </div>
      )}

      {aboutOpen && <AboutModal onClose={() => setAboutOpen(false)} />}
    </div>
  )
}
