import { API_BASE } from './constants'
import type {
  HealthStatus,
  DatasetSummary,
  DatasetPreview,
  DatasetProfile,
  StationInfo,
  ModelSummary,
  ModelDetail,
  TrainingConfig,
  TrainingResult,
  ForecastResult,
  ForecastResultRaw,
  ForecastTimePoint,
  CounterfactualResult,
  AvailableModel,
  ExplainResult,
  IPSReference,
} from './types'

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 60_000)
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

// --- Transform helpers ---

/** Extract value from a serialized TimeSeries point {time, col: value} */
function extractValue(point: ForecastTimePoint): number | null {
  for (const [key, val] of Object.entries(point)) {
    if (key !== 'time' && typeof val === 'number') return val
    if (key !== 'time' && val === null) return null
  }
  return null
}

/** Transform backend ForecastResultRaw to frontend ForecastResult */
function transformForecastResult(raw: ForecastResultRaw): ForecastResult {
  const dates = raw.target?.map((p) => p.time) ?? raw.predictions?.map((p) => p.time) ?? []
  const actuals = raw.target?.map(extractValue) ?? []
  const predictions = raw.predictions?.map(extractValue) ?? []

  return {
    dates,
    predictions,
    actuals,
    metrics: raw.metrics ?? {},
    confidence_low: [],
    confidence_high: [],
    predictions_onestep: raw.predictions_onestep?.map(extractValue) ?? null,
    metrics_onestep: raw.metrics_onestep ?? null,
    predictions_exact: raw.predictions_exact?.map(extractValue) ?? null,
    metrics_exact: raw.metrics_exact ?? null,
  }
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
    preview: (id: string, n: number = 50) =>
      fetchJson<DatasetPreview>(`/datasets/${id}/preview?n=${n}`),
    profile: (id: string) => fetchJson<DatasetProfile>(`/datasets/${id}/profile`),
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
      fetchJson<TrainingResult>(`/training/${taskId}/status`),
    cancel: (taskId: string) =>
      postJson<{ status: string; task_id: string }>(`/training/${taskId}/cancel`, {}),
    stream: (taskId: string) =>
      new EventSource(`${API_BASE}/training/${taskId}/stream`),
    history: () =>
      fetchJson<{ task_id: string; status: string; config: Record<string, unknown>; created_at: number }[]>(
        '/training/history',
      ),
  },

  models: {
    list: () => fetchJson<ModelSummary[]>('/models'),
    get: (id: string) => fetchJson<ModelDetail>(`/models/${id}`),
    delete: (id: string) => deleteJson<{ ok: boolean }>(`/models/${id}`),
    available: () => fetchJson<AvailableModel[]>('/models/available'),
    downloadUrl: (id: string) => `${API_BASE}/models/${id}/download`,
  },

  forecasting: {
    single: (body: { model_id: string; start_date?: string; use_covariates?: boolean; horizon?: number; dataset_id?: string }) =>
      postJson<ForecastResultRaw>('/forecasting/single', body).then(transformForecastResult),
    rolling: (body: { model_id: string; start_date: string; forecast_horizon: number; stride?: number; use_covariates?: boolean }) =>
      postJson<ForecastResultRaw>('/forecasting/rolling', body).then(transformForecastResult),
    comparison: (body: { model_id: string; start_date: string; forecast_horizon: number; use_covariates?: boolean }) =>
      postJson<ForecastResultRaw>('/forecasting/comparison', body).then(transformForecastResult),
    global: (body: { model_id: string; use_covariates?: boolean }) =>
      postJson<ForecastResultRaw>('/forecasting/global', body).then(transformForecastResult),
    run: (body: { model_id: string; horizon?: number; dataset_id?: string }) =>
      postJson<ForecastResultRaw>('/forecasting/run', body).then(transformForecastResult),
  },

  explainability: {
    featureImportance: (modelId: string) =>
      fetchJson<ExplainResult>(`/explainability/${modelId}/feature-importance`),
    featureImportancePost: (body: { model_id: string; method: string; n_permutations?: number }) =>
      postJson<ExplainResult>('/explainability/feature-importance', body),
    attention: (body: { model_id: string }) =>
      postJson<ExplainResult>('/explainability/attention', body),
    shap: (body: { model_id: string; n_samples?: number }) =>
      postJson<ExplainResult>('/explainability/shap', body),
    gradients: (body: { model_id: string; method?: string; target_step?: number; n_steps?: number }) =>
      postJson<ExplainResult>('/explainability/gradients', body),
  },

  counterfactual: {
    run: (body: {
      model_id: string
      method?: string
      target_ips_class?: string
      from_ips_class?: string
      to_ips_class?: string
      modifications?: Record<string, number>
      lambda_prox?: number
      n_iter?: number
      lr?: number
      cc_rate?: number
      device?: string
      n_trials?: number
      seed?: number
      k_sigma?: number
      lambda_smooth?: number
    }) => postJson<CounterfactualResult>('/counterfactual/run', body),
    stream: (taskId: string) =>
      new EventSource(`${API_BASE}/counterfactual/${taskId}/stream`),
    ipsReference: (modelId: string, window: number = 3) =>
      fetchJson<IPSReference>(`/counterfactual/ips-reference?model_id=${modelId}&window=${window}`),
  },
}
