import { fetchJson } from './observatory-api'
import type { NationalSituation, TerritorySituation } from './observatory-types'

export const situationApi = {
  national: (type: 'piezo' | 'hydro') =>
    fetchJson<NationalSituation>('/observatory/situation/national', { type }),
  territories: (level: 'region' | 'department', type: 'piezo' | 'hydro') =>
    fetchJson<TerritorySituation[]>('/observatory/situation/territories', { level, type }),
}
