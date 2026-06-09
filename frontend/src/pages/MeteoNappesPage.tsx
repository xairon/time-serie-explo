import { useState, useMemo, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { MeteoMap } from '@/components/meteo/MeteoMap'
import { MeteoLayersPanel } from '@/components/meteo/MeteoLayersPanel'
import type { LayerKey } from '@/components/meteo/MeteoLayersPanel'
import { MeteoLegend } from '@/components/meteo/MeteoLegend'
import { MeteoNationalBanner } from '@/components/meteo/MeteoNationalBanner'
import { MeteoCriticalList } from '@/components/meteo/MeteoCriticalList'
import { SituationTimelineSlider } from '@/components/meteo/SituationTimelineSlider'
import { SectorPopup } from '@/components/meteo/SectorPopup'
import { StationPopup } from '@/components/meteo/StationPopup'
import {
  useSectorSituation,
  useSectorTimeline,
  useStationsGeoJSON,
  useBrgmTimeline,
  useNationalSituation,
} from '@/hooks/useObservatory'
import {
  meteoClassColor,
  METEO_CLASS_LABELS,
  BRGM_CLASS_TO_ENUM,
  isCriticalClass,
  classSeverityRank,
  summarizeAlert,
} from '@/lib/meteo-colors'
import { SECTOR_INSUFFICIENT_COLOR } from '@/lib/sector-arrows'
import type { SectorSituation, SituationClass, StationGeoJSONFeature } from '@/lib/observatory-types'

// 7-class index → enum (index 7 = no data / insufficient).
const CLS = ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT'] as const
const TR: Record<number, 'baisse' | 'stable' | 'hausse'> = { [-1]: 'baisse', [0]: 'stable', [1]: 'hausse' }

function capitalize(s: string): string {
  if (!s) return s
  return s.charAt(0).toUpperCase() + s.slice(1)
}

type Source = 'brgm' | 'ips'
type Trend = 'hausse' | 'stable' | 'baisse' | null

interface SectorView {
  sectorId: number
  classEnum: string | null
  trend: Trend
  colorHex: string
  name: string
}

export default function MeteoNappesPage() {
  const [source, setSource] = useState<Source>('brgm')
  // Stations OFF by default — only the sector choropleth shows initially.
  const [visible, setVisible] = useState<Record<LayerKey, boolean>>({ bsn: true, piezo: false, rain: false, hydro: false })
  const [selectedPeriod, setSelectedPeriod] = useState<string | null>(null)
  const [selectedSectorId, setSelectedSectorId] = useState<number | null>(null)
  const [selectedSectorName, setSelectedSectorName] = useState<string | null>(null)
  const [selectedStation, setSelectedStation] = useState<{ code: string; type: 'piezo' | 'hydro' } | null>(null)
  const [focusSectorId, setFocusSectorId] = useState<number | null>(null)

  // BRGM exact published per-sector colors/trend across monthly windows.
  const { data: brgmTimeline } = useBrgmTimeline(source === 'brgm')

  // Our fixed-reference IPS (only fetched when that source is active).
  // BRGM météo des nappes is groundwater → 'piezo' sectors.
  const { data: sectorSituationData } = useSectorSituation('piezo', source === 'ips')
  const { data: timeline } = useSectorTimeline('piezo', source === 'ips')
  const { data: national } = useNationalSituation('piezo', source === 'ips')
  const { data: geojsonData } = useStationsGeoJSON()

  // BRGM records carry no sector name — fetch the geojson once to map id → nom.
  const { data: sectorGeo } = useQuery({
    queryKey: ['secteurs-bsh-geo'],
    queryFn: () => fetch('/geo/secteurs-bsh.geojson').then((r) => r.json()),
    staleTime: Infinity,
  })
  const nameById: Record<number, string> = useMemo(
    () => Object.fromEntries((sectorGeo?.features ?? []).map((f: any) => [f.properties.sector_id, f.properties.nom])),
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

  // Periods for the active source; effective = explicit selection or latest.
  const periods = source === 'brgm' ? (brgmTimeline?.periods ?? []) : (timeline?.periods ?? [])
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

  // Normalized per-sector views for the active source — single source of truth
  // for the choropleth fill, the alert highlight and the critical list.
  const views = useMemo<SectorView[]>(() => {
    if (source === 'brgm') {
      const window = (effectivePeriod && brgmTimeline?.windows[effectivePeriod]) || {}
      return Object.entries(window).map(([sidStr, w]) => ({
        sectorId: Number(sidStr),
        classEnum: BRGM_CLASS_TO_ENUM[w.brgm_class] ?? 'UNKNOWN',
        trend: w.trend,
        colorHex: w.color,
        name: nameById[Number(sidStr)] ?? `Secteur ${sidStr}`,
      }))
    }
    return displaySectorSituation.map((s) => {
      const sid = Number(s.code)
      return {
        sectorId: sid,
        classEnum: s.insufficient ? 'UNKNOWN' : s.situation_class,
        trend: s.trend,
        colorHex: s.insufficient ? SECTOR_INSUFFICIENT_COLOR : meteoClassColor(s.situation_class),
        name: nameById[sid] ?? s.name,
      }
    })
  }, [source, brgmTimeline, effectivePeriod, displaySectorSituation, nameById])

  // Derive everything the map/alert UI needs from the normalized views.
  const { sectorColorById, sectorTrendById, alertSectorIds, criticalList, alertSummary } = useMemo(() => {
    const colorById: Record<number, string> = {}
    const trendById: Record<number, Trend> = {}
    for (const v of views) {
      colorById[v.sectorId] = v.colorHex
      trendById[v.sectorId] = v.trend
    }
    const alertIds = views.filter((v) => isCriticalClass(v.classEnum)).map((v) => v.sectorId)
    const list = views
      .filter((v) => isCriticalClass(v.classEnum))
      .sort(
        (a, b) =>
          classSeverityRank(a.classEnum) - classSeverityRank(b.classEnum) || a.name.localeCompare(b.name),
      )
      .map((v) => ({
        code: String(v.sectorId),
        name: v.name,
        classLabel: METEO_CLASS_LABELS[v.classEnum ?? 'UNKNOWN'] ?? '',
        colorHex: v.colorHex,
        trend: v.trend,
      }))
    const summary = summarizeAlert(views.map((v) => ({ classEnum: v.classEnum, trend: v.trend })))
    return {
      sectorColorById: colorById,
      sectorTrendById: trendById,
      alertSectorIds: alertIds,
      criticalList: list,
      alertSummary: summary,
    }
  }, [views])

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
    setFocusSectorId(null)
    setSelectedPeriod(null)
  }, [])

  // Select + fly-to a sector from the critical list.
  const handleSelectSector = useCallback((code: string) => {
    const id = Number(code)
    setFocusSectorId(id)
    setSelectedSectorId(id)
    setSelectedSectorName(nameById[id] ?? null)
    setSelectedStation(null)
  }, [nameById])

  // Selected BRGM sector for the active period (from the timeline window).
  const selectedBrgm = useMemo(() => {
    if (selectedSectorId == null || source !== 'brgm') return null
    const w = (effectivePeriod && brgmTimeline?.windows[effectivePeriod]?.[String(selectedSectorId)]) || null
    if (!w) return null
    return { sector_id: selectedSectorId, brgm_class: w.brgm_class, trend: w.trend, color: w.color, ips: null as number | null }
  }, [brgmTimeline, effectivePeriod, selectedSectorId, source])

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
      const classKey = BRGM_CLASS_TO_ENUM[selectedBrgm.brgm_class] ?? 'UNKNOWN'
      return {
        name: selectedSectorName ?? nameById[selectedBrgm.sector_id] ?? '',
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
  }, [selectedBrgm, selectedIps, selectedSectorName, nameById])

  // Capture the clicked sector's display name (BRGM record has no name).
  const onSectorClickWithName = useCallback((id: number, name: string) => {
    setSelectedSectorName(name)
    onSectorClick(id, name)
  }, [onSectorClick])

  return (
    <div className="relative h-full">
      <MeteoMap
        sectorColorById={sectorColorById}
        sectorTrendById={sectorTrendById}
        alertSectorIds={alertSectorIds}
        focusSectorId={focusSectorId}
        visibleLayers={visible}
        piezoFeatures={piezoFeatures}
        hydroFeatures={hydroFeatures}
        onSectorClick={onSectorClickWithName}
        onStationClick={onStationClick}
      />

      {/* National summary banner — top, full-width. */}
      <div className="absolute top-2 left-1/2 -translate-x-1/2 w-[min(96vw,900px)] z-20">
        <MeteoNationalBanner
          dominantClassLabel={alertSummary.dominantClassLabel}
          dominantColorHex={meteoClassColor(alertSummary.dominantClass as any)}
          trendSummary={alertSummary.trendSummary}
          criticalCount={alertSummary.criticalCount}
          totalSectors={views.length}
          pctBelowNormal={source === 'ips' ? (national?.pct_below_normal ?? null) : null}
        />
      </div>

      {/* Data-source toggle — just below the banner, right-aligned. */}
      <div className="absolute top-[72px] left-1/2 translate-x-[calc(min(48vw,450px)-100%)] z-20 flex rounded-lg bg-white shadow-md border border-slate-200 overflow-hidden text-xs font-medium">
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

      <MeteoLayersPanel visible={visible} onToggle={onToggle} />
      <MeteoLegend />

      {/* Critical-sectors list — left panel, stacked below the layers panel. */}
      <div className="absolute top-[290px] left-3 z-10 w-64">
        <MeteoCriticalList sectors={criticalList} onSelect={handleSelectSector} />
      </div>

      {/* Monthly window slider — both sources are period-aware. */}
      {periods.length > 0 && (
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
