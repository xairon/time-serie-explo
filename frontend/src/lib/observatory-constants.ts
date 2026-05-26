// Observatory constants — ported from junondashboard
import type { WfsLayerConfig, WfsLayerId } from './observatory-types'

export const CLASSIFICATION_COLORS: Record<string, string> = {
  EXTREMEMENT_BAS: '#991b1b',
  TRES_BAS: '#ef4444',
  BAS: '#f97316',
  NORMAL: '#10b981',
  HAUT: '#3b82f6',
  TRES_HAUT: '#1d4ed8',
  EXTREMEMENT_HAUT: '#312e81',
  UNKNOWN: '#6b7280',
}

export const CLASSIFICATION_LABELS: Record<string, string> = {
  EXTREMEMENT_BAS: 'Extremely low',
  TRES_BAS: 'Very low',
  BAS: 'Low',
  NORMAL: 'Normal',
  HAUT: 'High',
  TRES_HAUT: 'Very high',
  EXTREMEMENT_HAUT: 'Extremely high',
  UNKNOWN: 'Unclassified',
}

export const CLASSIFICATION_ORDER = ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'NORMAL', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT'] as const

export const TREND_LABELS: Record<string, string> = {
  HAUSSE_FORTE: 'Strong rise',
  HAUSSE_SIGNIFICATIVE: 'Significant rise',
  STABLE: 'Stable',
  BAISSE_SIGNIFICATIVE: 'Significant decline',
  BAISSE_FORTE: 'Strong decline',
}

export const TREND_COLORS: Record<string, string> = {
  HAUSSE_FORTE: '#3b82f6',
  HAUSSE_SIGNIFICATIVE: '#60a5fa',
  STABLE: '#10b981',
  BAISSE_SIGNIFICATIVE: '#f97316',
  BAISSE_FORTE: '#ef4444',
}

export const TREND_ORDER = ['BAISSE_FORTE', 'BAISSE_SIGNIFICATIVE', 'STABLE', 'HAUSSE_SIGNIFICATIVE', 'HAUSSE_FORTE'] as const

// --- WFS Layer Configuration ---

const SANDRE_ZONAGE_COLORS: Record<string, string> = {
  'region-hydro': '#3b82f6',
  'secteur-hydro': '#06b6d4',
  'sous-secteur-hydro': '#14b8a6',
  'zone-hydro': '#10b981',
}

export const WFS_LAYERS: WfsLayerConfig[] = [
  {
    id: 'region-hydro',
    label: 'Hydrographic regions',
    group: 'sandre',
    minZoom: 0,
    geometryType: 'polygon',
    color: SANDRE_ZONAGE_COLORS['region-hydro'],
    tooltipFields: ['CdRegionHydro', 'LbRegionHydro'],
  },
  {
    id: 'secteur-hydro',
    label: 'Hydrographic sectors',
    group: 'sandre',
    minZoom: 6,
    geometryType: 'polygon',
    color: SANDRE_ZONAGE_COLORS['secteur-hydro'],
    tooltipFields: ['CdSecteurHydro', 'LbSecteurHydro'],
  },
  {
    id: 'sous-secteur-hydro',
    label: 'Hydrographic sub-sectors',
    group: 'sandre',
    minZoom: 7,
    geometryType: 'polygon',
    color: SANDRE_ZONAGE_COLORS['sous-secteur-hydro'],
    tooltipFields: ['CdSousSecteurHydro', 'LbSousSecteurHydro'],
  },
  {
    id: 'zone-hydro',
    label: 'Hydrographic zones',
    group: 'sandre',
    minZoom: 9,
    geometryType: 'polygon',
    color: SANDRE_ZONAGE_COLORS['zone-hydro'],
    tooltipFields: ['CdZoneHydro', 'LbZoneHydro', 'Superficie'],
  },
  {
    id: 'cours-eau-1',
    label: 'Major rivers (>100km)',
    group: 'carthage',
    minZoom: 6,
    geometryType: 'line',
    color: '#60a5fa',
    tooltipFields: ['NomEntiteHydrographique', 'CdEntiteHydrographique'],
  },
  {
    id: 'cours-eau-2',
    label: 'Secondary rivers (50-100km)',
    group: 'carthage',
    minZoom: 8,
    geometryType: 'line',
    color: '#93c5fd',
    tooltipFields: ['NomEntiteHydrographique', 'CdEntiteHydrographique'],
  },
  {
    id: 'plan-eau',
    label: 'Lakes',
    group: 'carthage',
    minZoom: 8,
    geometryType: 'polygon',
    color: '#38bdf8',
    tooltipFields: ['NomEntiteHydrographique', 'CdEntiteHydrographique', 'Superficie'],
  },
  {
    id: 'masse-eau-riv',
    label: 'Water bodies (WFD)',
    group: 'hydroeco',
    minZoom: 8,
    geometryType: 'line',
    color: '#c084fc',
    tooltipFields: ['CdMasseDEau', 'NomMasseDEau'],
  },
]

export const WFS_LAYER_MAP = Object.fromEntries(
  WFS_LAYERS.map(l => [l.id, l])
) as Record<WfsLayerId, WfsLayerConfig>

export interface LayerGroup {
  id: string
  label: string
  icon: string
  mode: 'radio' | 'checkbox'
  layers: WfsLayerConfig[]
}

export const LAYER_GROUPS: LayerGroup[] = [
  {
    id: 'sandre',
    label: 'SANDRE Zoning',
    icon: '\u{1F30A}',
    mode: 'radio',
    layers: WFS_LAYERS.filter(l => l.group === 'sandre'),
  },
  {
    id: 'carthage',
    label: 'Hydrographic network',
    icon: '\u{1F3DE}',
    mode: 'checkbox',
    layers: WFS_LAYERS.filter(l => l.group === 'carthage'),
  },
  {
    id: 'hydroeco',
    label: 'Hydro-ecology',
    icon: '\u{1F9EA}',
    mode: 'checkbox',
    layers: WFS_LAYERS.filter(l => l.group === 'hydroeco'),
  },
]
