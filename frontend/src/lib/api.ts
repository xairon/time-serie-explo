import { API_BASE } from './constants'
import type {
  HealthStatus,
  DatasetSummary,
  StationInfo,
  ModelSummary,
  TrainingConfig,
  ForecastResult,
  CounterfactualResult,
  AvailableModel,
} from './types'

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 30_000)
  try {
    const res = await fetch(url, {
      ...init,
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
        ...init?.headers,
      },
    })
    if (!res.ok) {
      let detail = ''
      try {
        const body = await res.json() as { detail?: unknown }
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      } catch { /* ignore parse errors */ }
      throw new Error(`API ${res.status}${detail ? `: ${detail}` : ''}`)
    }
    return await res.json() as T
  } finally {
    clearTimeout(timeoutId)
  }
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  return fetchJson<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

async function deleteJson<T>(path: string): Promise<T> {
  return fetchJson<T>(path, { method: 'DELETE' })
}

export const api = {
  health: () => fetchJson<HealthStatus>('/health'),

  db: {
    schemas: () => fetchJson<string[]>('/db/schemas'),
    tables: (schema: string) =>
      fetchJson<{ tables: string[]; views: string[] }>(`/db/tables?schema=${schema}`),
    columns: (table: string, schema: string) =>
      fetchJson<{
        columns: { name: string; type: string; nullable: boolean }[]
        row_count: number
        date_columns: string[]
      }>(`/db/columns?table=${table}&schema=${schema}`),
    distinct: (table: string, column: string, schema: string) =>
      fetchJson<string[]>(`/db/distinct?table=${table}&column=${column}&schema=${schema}`),
    dateRange: (table: string, column: string, schema: string) =>
      fetchJson<{ min: string | null; max: string | null }>(
        `/db/date-range?table=${table}&column=${column}&schema=${schema}`,
      ),
    searchStations: (params: {
      q?: string
      departement?: string
      tendance?: string
      alerte?: string
      limit?: number
    }) => {
      const sp = new URLSearchParams()
      if (params.q) sp.set('q', params.q)
      if (params.departement) sp.set('departement', params.departement)
      if (params.tendance) sp.set('tendance', params.tendance)
      if (params.alerte) sp.set('alerte', params.alerte)
      if (params.limit) sp.set('limit', String(params.limit))
      return fetchJson<{ stations: StationInfo[]; total: number }>(
        `/db/stations/search?${sp.toString()}`,
      )
    },
    stationFilters: () =>
      fetchJson<{
        departements: string[]
        tendances: string[]
        alertes: string[]
        classifications: string[]
      }>('/db/stations/filters'),
  },

  datasets: {
    list: () => fetchJson<DatasetSummary[]>('/datasets'),
    get: (id: string) => fetchJson<DatasetSummary>(`/datasets/${id}`),
    create: (body: FormData) =>
      fetch(`${API_BASE}/datasets`, { method: 'POST', body }).then(async (res) => {
        if (!res.ok) throw new Error(`API ${res.status}`)
        return res.json() as Promise<DatasetSummary>
      }),
    delete: (id: string) => deleteJson<{ ok: boolean }>(`/datasets/${id}`),
    importDB: (body: {
      table_name: string
      schema_name: string
      columns: string[]
      date_column?: string
      start_date?: string
      end_date?: string
      filters?: Record<string, string[]>
      dataset_name?: string
    }) => postJson<DatasetSummary>('/datasets/import-db', body),
  },

  training: {
    start: (config: TrainingConfig) =>
      postJson<{ task_id: string }>('/training/start', config),
    status: (taskId: string) =>
      fetchJson<{ status: string; metrics?: Record<string, number> }>(`/training/${taskId}/status`),
    stop: (taskId: string) =>
      postJson<{ ok: boolean }>(`/training/${taskId}/stop`, {}),
    stream: (taskId: string) =>
      new EventSource(`${API_BASE}/training/${taskId}/stream`),
    availableModels: () =>
      fetchJson<AvailableModel[]>('/training/models'),
  },

  models: {
    list: () => fetchJson<ModelSummary[]>('/models'),
    get: (id: string) => fetchJson<ModelSummary>(`/models/${id}`),
    delete: (id: string) => deleteJson<{ ok: boolean }>(`/models/${id}`),
  },

  forecasting: {
    run: (body: { model_id: string; horizon: number; dataset_id: string }) =>
      postJson<ForecastResult>('/forecasting/run', body),
  },

  explainability: {
    featureImportance: (modelId: string) =>
      fetchJson<{ features: string[]; importances: number[] }>(
        `/explainability/${modelId}/feature-importance`,
      ),
    shap: (modelId: string) =>
      fetchJson<{ shap_values: number[][]; feature_names: string[] }>(
        `/explainability/${modelId}/shap`,
      ),
    attention: (modelId: string) =>
      fetchJson<{ attention_weights: number[][] }>(
        `/explainability/${modelId}/attention`,
      ),
  },

  counterfactual: {
    run: (body: {
      model_id: string
      dataset_id: string
      modifications: Record<string, number>
    }) => postJson<CounterfactualResult>('/counterfactual/run', body),
  },
}
