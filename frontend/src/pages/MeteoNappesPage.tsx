import { useState, useMemo, useCallback } from 'react'
import { MeteoMap } from '@/components/meteo/MeteoMap'
import { MeteoLayersPanel } from '@/components/meteo/MeteoLayersPanel'
import type { LayerKey } from '@/components/meteo/MeteoLayersPanel'
import { MeteoLegend } from '@/components/meteo/MeteoLegend'
import { SituationTimelineSlider } from '@/components/meteo/SituationTimelineSlider'
import { SectorPopup } from '@/components/meteo/SectorPopup'
import { StationPopup } from '@/components/meteo/StationPopup'
import { useSectorSituation, useSectorTimeline, useStationsGeoJSON } from '@/hooks/useObservatory'
import type { SectorSituation, SituationClass, StationGeoJSONFeature } from '@/lib/observatory-types'

// 7-class index → enum (index 7 = no data / insufficient).
const CLS = ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT'] as const
const TR: Record<number, 'baisse' | 'stable' | 'hausse'> = { [-1]: 'baisse', [0]: 'stable', [1]: 'hausse' }

export default function MeteoNappesPage() {
  const [visible, setVisible] = useState<Record<LayerKey, boolean>>({ bsn: true, piezo: true, rain: false, hydro: false })
  const [selectedPeriod, setSelectedPeriod] = useState<string | null>(null)
  const [selectedSectorId, setSelectedSectorId] = useState<number | null>(null)
  const [selectedStation, setSelectedStation] = useState<{ code: string; type: 'piezo' | 'hydro' } | null>(null)

  // BRGM météo des nappes is groundwater → 'piezo' sectors.
  const { data: sectorSituationData } = useSectorSituation('piezo', true)
  const { data: timeline } = useSectorTimeline('piezo', true)
  const { data: geojsonData } = useStationsGeoJSON()

  const piezoFeatures = useMemo<StationGeoJSONFeature[]>(
    () => (geojsonData?.features ?? []).filter((f) => f.properties.type === 'piezo'),
    [geojsonData],
  )
  const hydroFeatures = useMemo<StationGeoJSONFeature[]>(
    () => (geojsonData?.features ?? []).filter((f) => f.properties.type === 'hydro'),
    [geojsonData],
  )

  // Latest published period; the slider defaults to it internally too.
  const periods = timeline?.periods ?? []
  const effectivePeriod = selectedPeriod ?? (periods.length ? periods[periods.length - 1] : null)

  // Recolor the choropleth for the selected period (mirror ObservatoryPage).
  const displaySectorSituation = useMemo<SectorSituation[]>(() => {
    const base = sectorSituationData ?? []
    if (!timeline || effectivePeriod == null) return base
    const isLatest = periods.length > 0 && effectivePeriod === periods[periods.length - 1]
    if (isLatest) return base
    const sIdx = timeline.periods.indexOf(effectivePeriod)
    if (sIdx < 0) return base
    return base.map((s) => {
      const ci = timeline.sectors[s.code]?.[sIdx]
      const ti = timeline.trends[s.code]?.[sIdx]
      const insufficient = ci == null || ci === 7
      return {
        ...s,
        situation_class: insufficient ? null : (CLS[ci] as SituationClass),
        trend: ti != null ? TR[ti] : null,
        insufficient,
      }
    })
  }, [sectorSituationData, timeline, effectivePeriod, periods])

  const onSectorClick = useCallback((id: number, _name: string) => {
    setSelectedSectorId(id)
    setSelectedStation(null)
  }, [])

  const onStationClick = useCallback((code: string, type: 'piezo' | 'hydro') => {
    setSelectedStation({ code, type })
    setSelectedSectorId(null)
  }, [])

  const onToggle = useCallback((k: LayerKey) => {
    setVisible((v) => ({ ...v, [k]: !v[k] }))
  }, [])

  const selectedSector = useMemo<SectorSituation | null>(() => {
    if (selectedSectorId == null) return null
    return displaySectorSituation.find((s) => Number(s.code) === selectedSectorId) ?? null
  }, [displaySectorSituation, selectedSectorId])

  const selectedStationFeature = useMemo<StationGeoJSONFeature | null>(() => {
    if (!selectedStation) return null
    const pool = selectedStation.type === 'piezo' ? piezoFeatures : hydroFeatures
    return pool.find((f) => f.properties.code === selectedStation.code) ?? null
  }, [selectedStation, piezoFeatures, hydroFeatures])

  return (
    <div className="fixed inset-0">
      <MeteoMap
        sectorSituation={displaySectorSituation}
        visibleLayers={visible}
        piezoFeatures={piezoFeatures}
        hydroFeatures={hydroFeatures}
        onSectorClick={onSectorClick}
        onStationClick={onStationClick}
      />
      <MeteoLayersPanel visible={visible} onToggle={onToggle} />
      <MeteoLegend />
      <SituationTimelineSlider periods={periods} selectedPeriod={effectivePeriod} onChange={setSelectedPeriod} />

      {selectedSector && (
        <div className="absolute top-4 right-4 z-20">
          <SectorPopup sector={selectedSector} onClose={() => setSelectedSectorId(null)} />
        </div>
      )}

      {selectedStationFeature && (
        <div className="absolute top-4 right-4 z-20">
          <StationPopup
            code={selectedStationFeature.properties.code}
            commune={selectedStationFeature.properties.commune ?? undefined}
            classification={selectedStationFeature.properties.classification}
            derniereMesure={selectedStationFeature.properties.derniere_mesure}
            onClose={() => setSelectedStation(null)}
          />
        </div>
      )}
    </div>
  )
}
