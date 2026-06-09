import { useState, useMemo, useCallback } from 'react'
import { MeteoMap } from '@/components/meteo/MeteoMap'
import { MeteoLayersPanel } from '@/components/meteo/MeteoLayersPanel'
import type { LayerKey } from '@/components/meteo/MeteoLayersPanel'
import { MeteoLegend } from '@/components/meteo/MeteoLegend'
import { SituationTimelineSlider } from '@/components/meteo/SituationTimelineSlider'
import { SectorPopup } from '@/components/meteo/SectorPopup'
import { StationPopup } from '@/components/meteo/StationPopup'
import { useSectorSituation, useSectorTimeline, useStationsGeoJSON, useBrgmSectors } from '@/hooks/useObservatory'
import { meteoClassColor, METEO_CLASS_LABELS } from '@/lib/meteo-colors'
import { SECTOR_INSUFFICIENT_COLOR } from '@/lib/sector-arrows'
import type { SectorSituation, SituationClass, StationGeoJSONFeature, BrgmSector } from '@/lib/observatory-types'

// 7-class index → enum (index 7 = no data / insufficient).
const CLS = ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT'] as const
const TR: Record<number, 'baisse' | 'stable' | 'hausse'> = { [-1]: 'baisse', [0]: 'stable', [1]: 'hausse' }

// BRGM published class index (0=no aquifer / grey) → our 7-enum + UNKNOWN.
const brgmClassToEnum: Record<number, string> = {
  0: 'UNKNOWN',
  1: 'EXTREMEMENT_BAS',
  2: 'TRES_BAS',
  3: 'BAS',
  4: 'NORMAL',
  5: 'HAUT',
  6: 'TRES_HAUT',
  7: 'EXTREMEMENT_HAUT',
}

function capitalize(s: string): string {
  if (!s) return s
  return s.charAt(0).toUpperCase() + s.slice(1)
}

type Source = 'brgm' | 'ips'

export default function MeteoNappesPage() {
  const [source, setSource] = useState<Source>('brgm')
  const [visible, setVisible] = useState<Record<LayerKey, boolean>>({ bsn: true, piezo: true, rain: false, hydro: false })
  const [selectedPeriod, setSelectedPeriod] = useState<string | null>(null)
  const [selectedSectorId, setSelectedSectorId] = useState<number | null>(null)
  const [selectedSectorName, setSelectedSectorName] = useState<string | null>(null)
  const [selectedStation, setSelectedStation] = useState<{ code: string; type: 'piezo' | 'hydro' } | null>(null)

  // BRGM exact published per-sector colors/trend (default source).
  const { data: brgmSectors } = useBrgmSectors(source === 'brgm')

  // Our fixed-reference IPS (only fetched when that source is active).
  // BRGM météo des nappes is groundwater → 'piezo' sectors.
  const { data: sectorSituationData } = useSectorSituation('piezo', source === 'ips')
  const { data: timeline } = useSectorTimeline('piezo', source === 'ips')
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

  // Source-agnostic explicit fill + trend maps consumed by MeteoMap.
  const { sectorColorById, sectorTrendById } = useMemo(() => {
    const colorById: Record<number, string> = {}
    const trendById: Record<number, 'hausse' | 'stable' | 'baisse' | null> = {}
    if (source === 'brgm') {
      for (const b of brgmSectors ?? []) {
        colorById[b.sector_id] = b.color
        trendById[b.sector_id] = b.trend
      }
    } else {
      for (const s of displaySectorSituation) {
        const sid = Number(s.code)
        colorById[sid] = s.insufficient ? SECTOR_INSUFFICIENT_COLOR : meteoClassColor(s.situation_class)
        trendById[sid] = s.trend
      }
    }
    return { sectorColorById: colorById, sectorTrendById: trendById }
  }, [source, brgmSectors, displaySectorSituation])

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

  const onSourceChange = useCallback((next: Source) => {
    setSource(next)
    setSelectedSectorId(null)
    setSelectedStation(null)
    if (next === 'brgm') setSelectedPeriod(null)
  }, [])

  // Sector geometry carries `nom`; the click handler passes it through.
  const selectedBrgm = useMemo<BrgmSector | null>(() => {
    if (selectedSectorId == null || source !== 'brgm') return null
    return (brgmSectors ?? []).find((b) => b.sector_id === selectedSectorId) ?? null
  }, [brgmSectors, selectedSectorId, source])

  const selectedIps = useMemo<SectorSituation | null>(() => {
    if (selectedSectorId == null || source !== 'ips') return null
    return displaySectorSituation.find((s) => Number(s.code) === selectedSectorId) ?? null
  }, [displaySectorSituation, selectedSectorId, source])

  const selectedStationFeature = useMemo<StationGeoJSONFeature | null>(() => {
    if (!selectedStation) return null
    const pool = selectedStation.type === 'piezo' ? piezoFeatures : hydroFeatures
    return pool.find((f) => f.properties.code === selectedStation.code) ?? null
  }, [selectedStation, piezoFeatures, hydroFeatures])

  // Build SectorPopup props for whichever source is active.
  const sectorPopupProps = useMemo(() => {
    if (selectedBrgm) {
      const classKey = brgmClassToEnum[selectedBrgm.brgm_class] ?? 'UNKNOWN'
      return {
        name: selectedSectorName ?? '',
        code: String(selectedBrgm.sector_id),
        classLabel: capitalize(METEO_CLASS_LABELS[classKey] ?? METEO_CLASS_LABELS.UNKNOWN),
        trend: selectedBrgm.trend,
        colorHex: selectedBrgm.color,
        metrics: { ips: selectedBrgm.ips },
      }
    }
    if (selectedIps) {
      const classKey = selectedIps.situation_class ?? 'UNKNOWN'
      return {
        name: selectedIps.name,
        code: selectedIps.code,
        classLabel: capitalize(METEO_CLASS_LABELS[classKey] ?? METEO_CLASS_LABELS.UNKNOWN),
        trend: selectedIps.trend,
        colorHex: selectedIps.insufficient ? SECTOR_INSUFFICIENT_COLOR : meteoClassColor(selectedIps.situation_class),
        metrics: {
          pctBelowNormal: selectedIps.pct_below_normal,
          nEligible: selectedIps.n_eligible,
          nProvisoire: selectedIps.n_provisoire,
        },
      }
    }
    return null
  }, [selectedBrgm, selectedIps, selectedSectorName])

  // Capture the clicked sector's display name (BRGM record has no name).
  const onSectorClickWithName = useCallback((id: number, name: string) => {
    setSelectedSectorName(name)
    onSectorClick(id, name)
  }, [onSectorClick])

  return (
    <div className="fixed inset-0">
      <MeteoMap
        sectorColorById={sectorColorById}
        sectorTrendById={sectorTrendById}
        visibleLayers={visible}
        piezoFeatures={piezoFeatures}
        hydroFeatures={hydroFeatures}
        onSectorClick={onSectorClickWithName}
        onStationClick={onStationClick}
      />
      <MeteoLayersPanel visible={visible} onToggle={onToggle} />
      <MeteoLegend />

      {/* Data-source toggle */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex rounded-lg bg-white shadow-md border border-slate-200 overflow-hidden text-xs font-medium">
        <button
          onClick={() => onSourceChange('brgm')}
          className={`px-3 py-2 transition-colors ${source === 'brgm' ? 'bg-slate-800 text-white' : 'text-slate-600 hover:bg-slate-50'}`}
          aria-pressed={source === 'brgm'}
        >
          MétéEau (BRGM)
        </button>
        <button
          onClick={() => onSourceChange('ips')}
          className={`px-3 py-2 transition-colors ${source === 'ips' ? 'bg-slate-800 text-white' : 'text-slate-600 hover:bg-slate-50'}`}
          aria-pressed={source === 'ips'}
        >
          Notre IPS (réf. fixe)
        </button>
      </div>

      {/* Slider only for our IPS source (BRGM = current published snapshot). */}
      {source === 'ips' && (
        <SituationTimelineSlider periods={periods} selectedPeriod={effectivePeriod} onChange={setSelectedPeriod} />
      )}

      {sectorPopupProps && (
        <div className="absolute top-16 right-4 z-20">
          <SectorPopup {...sectorPopupProps} onClose={() => setSelectedSectorId(null)} />
        </div>
      )}

      {selectedStationFeature && (
        <div className="absolute top-16 right-4 z-20">
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
