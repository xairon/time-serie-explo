import { fetchJson } from './observatory-api'
import type { TerritorySituation, SectorSituation, SectorTimeline } from './observatory-types'

export const situationApi = {
  territories: (level: 'region' | 'department', type: 'piezo' | 'hydro') =>
    fetchJson<TerritorySituation[]>('/observatory/situation/territories', { level, type }),
  sectors: (type: 'piezo' | 'hydro', month?: string) =>
    fetchJson<SectorSituation[]>('/observatory/situation/sectors', { type, ...(month ? { month } : {}) }),
  sectorsTimeline: (type: 'piezo' | 'hydro') =>
    fetchJson<SectorTimeline>('/observatory/situation/sectors/timeline', { type }),
}
