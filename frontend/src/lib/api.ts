import { API_BASE } from './constants'
import type {
  HealthStatus,
  DatasetSummary,
  DatasetPreview,
  DatasetProfile,
  StationInfo,
  ModelSummary,
  ModelDetail,
  ModelTestInfo,
  TrainingConfig,
  TrainingResult,
  ForecastResult,
  ForecastResultRaw,
  ForecastTimePoint,
  CounterfactualResult,
  AvailableModel,
  ExplainResult,
  LagImportanceResult,
  ResidualAnalysisResult,
  SeasonalityResult,
  IPSReference,
  IPSBoundsResponse,
  PastasValidationResult,
  PastasOptions,
  PastasFitResponse,
  PastasModelSummary,
  PastasScenarioResponse,
  PastasStationPreview,
  ScenarioPresetsData,
  SavedScenario,
  PastasCompareResponse,
  DiagnoseResult,
  AutoFitResult,
  CompareAIResult,
} from './types'

async function fetchJson<T>(path: string, init?: RequestInit & { timeout?: number }): Promise<T> {
  const url = `${API_BASE}${path}`
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), init?.timeout ?? 60_000)
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

async function postJson<T>(path: string, body: unknown, timeout?: number): Promise<T> {
  return fetchJson<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    timeout,
  })
}

async function deleteJson(path: string): Promise<void> {
  const url = `${API_BASE}${path}`
  const res = await fetch(url, { method: 'DELETE' })
  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json() as { detail?: unknown }
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* ignore */ }
    throw new Error(`API ${res.status}${detail ? `: ${detail}` : ''}`)
  }
}

async function patchJson<T>(path: string, body: unknown): Promise<T> {
  return fetchJson<T>(path, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
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

  // Use predictions_exact as fallback for predictions_onestep (comparison endpoint)
  const onestepRaw = raw.predictions_onestep ?? raw.predictions_exact
  const onestepMetrics = raw.metrics_onestep ?? raw.metrics_exact

  return {
    dates,
    predictions,
    actuals,
    metrics: raw.metrics ?? {},
    confidence_low: [],
    confidence_high: [],
    predictions_onestep: onestepRaw?.map(extractValue) ?? null,
    metrics_onestep: onestepMetrics ?? null,
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
    update: (id: string, body: { target_variable?: string; covariates?: string[]; preprocessing?: Record<string, unknown> }) =>
      patchJson<DatasetSummary>(`/datasets/${id}`, body),
    delete: (id: string) => deleteJson(`/datasets/${id}`),
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
    delete: (id: string) => deleteJson(`/models/${id}`),
    available: () => fetchJson<AvailableModel[]>('/models/available'),
    downloadUrl: (id: string) => `${API_BASE}/models/${id}/download`,
    testInfo: (id: string) => fetchJson<ModelTestInfo>(`/models/${id}/test-info`),
  },

  forecasting: {
    single: (body: { model_id: string; start_date?: string; use_covariates?: boolean; horizon?: number; dataset_id?: string }) =>
      postJson<ForecastResultRaw>('/forecasting/single', body).then(transformForecastResult),
    rolling: (body: { model_id: string; start_date: string; forecast_horizon: number; stride?: number; use_covariates?: boolean }) =>
      postJson<ForecastResultRaw>('/forecasting/rolling', body, 300_000).then(transformForecastResult),
    comparison: (body: { model_id: string; start_date: string; forecast_horizon: number; use_covariates?: boolean }) =>
      postJson<ForecastResultRaw>('/forecasting/comparison', body, 300_000).then(transformForecastResult),
    global: (body: { model_id: string; use_covariates?: boolean }) =>
      postJson<ForecastResultRaw>('/forecasting/global', body, 300_000).then(transformForecastResult),
    run: (body: { model_id: string; horizon?: number; dataset_id?: string }) =>
      postJson<ForecastResultRaw>('/forecasting/run', body, 300_000).then(transformForecastResult),
  },

  explainability: {
    featureImportance: (modelId: string) =>
      fetchJson<ExplainResult>(`/explainability/${modelId}/feature-importance`),
    featureImportancePost: (body: { model_id: string; method: string; n_permutations?: number }) =>
      postJson<ExplainResult>('/explainability/feature-importance', body),
    permutationImportance: (body: { model_id: string; n_permutations?: number }) =>
      postJson<ExplainResult>('/explainability/feature-importance', { ...body, method: 'permutation' }),
    attention: (body: { model_id: string }) =>
      postJson<ExplainResult>('/explainability/attention', body),
    shap: (body: { model_id: string; n_samples?: number }) =>
      postJson<ExplainResult>('/explainability/shap', body),
    gradients: (body: { model_id: string; method?: string; target_step?: number; n_steps?: number }) =>
      postJson<ExplainResult>('/explainability/gradients', body),
    lagImportance: (modelId: string) =>
      fetchJson<LagImportanceResult>(`/explainability/${modelId}/lag-importance`),
    residuals: (modelId: string) =>
      fetchJson<ResidualAnalysisResult>(`/explainability/${modelId}/residuals`),
    seasonality: (modelId: string) =>
      fetchJson<SeasonalityResult>(`/explainability/${modelId}/seasonality`),
  },

  pumpingDetection: {
    analyze: (body: { dataset_id: string; config?: Record<string, unknown> }) =>
      postJson<{ task_id: string }>('/pumping-detection/analyze', body),
    stream: (taskId: string) =>
      new EventSource(`${API_BASE}/pumping-detection/${taskId}/stream`),
    results: (taskId: string) =>
      fetchJson<Record<string, unknown>>(`/pumping-detection/${taskId}/results`),
    cancel: (taskId: string) =>
      postJson<{ status: string }>(`/pumping-detection/${taskId}/cancel`, {}),
    bnpeContext: (lat: number, lon: number, radiusKm: number = 5) =>
      fetchJson<Record<string, unknown>>(`/pumping-detection/bnpe-context?lat=${lat}&lon=${lon}&radius_km=${radiusKm}`),
  },

  pastas: {
    options: () => fetchJson<PastasOptions>('/pastas/options'),
    stationInfo: (codeBss: string) => fetchJson<Record<string, unknown>>(`/pastas/station-info?code_bss=${encodeURIComponent(codeBss)}`),
    siblings: (codeBss: string) => fetchJson<{
      siblings: { code_bss: string; lat: number; lon: number; nom_commune?: string }[]
      bdlisa_code?: string
    }>(`/pastas/siblings?code_bss=${encodeURIComponent(codeBss)}`),
    preview: (codeBss: string) => fetchJson<PastasStationPreview>(`/pastas/preview?code_bss=${encodeURIComponent(codeBss)}`, { timeout: 30_000 }),
    fit: (body: {
      code_bss: string
      tmin?: string
      tmax?: string
      recharge?: { type: string; kwargs?: Record<string, unknown> }
      response?: { type: string; kwargs?: Record<string, unknown> }
      noise?: { type: string }
      solver?: { type: string; kwargs?: Record<string, unknown> }
      name?: string
      val_split?: number
      include_temp?: boolean
    }) => postJson<PastasFitResponse>('/pastas/fit', body, 120_000),
    models: (codeBss?: string) => {
      const params = codeBss ? `?code_bss=${codeBss}` : ''
      return fetchJson<PastasModelSummary[]>(`/pastas/models${params}`)
    },
    model: (runId: string) => fetchJson<PastasFitResponse>(`/pastas/models/${runId}`),
    deleteModel: (runId: string) => deleteJson(`/pastas/models/${runId}`),
    simulate: (body: {
      run_id: string
      tmin: string
      tmax: string
      modifications: Array<Record<string, unknown>>
    }) => postJson<PastasScenarioResponse>('/pastas/simulate', body, 120_000),
    scenarioPresets: (params?: { aquifer_family?: string; tmin?: string; tmax?: string }) => {
      const qs = new URLSearchParams()
      if (params?.aquifer_family) qs.set('aquifer_family', params.aquifer_family)
      if (params?.tmin) qs.set('tmin', params.tmin)
      if (params?.tmax) qs.set('tmax', params.tmax)
      const query = qs.toString()
      return fetchJson<ScenarioPresetsData>(`/pastas/scenario-presets${query ? `?${query}` : ''}`)
    },
    savedScenarios: (runId: string) =>
      fetchJson<SavedScenario[]>(`/pastas/models/${runId}/scenarios`),
    saveScenario: (runId: string, body: { name: string; description?: string; modifications: Array<Record<string, unknown>>; tmin?: string; tmax?: string }) =>
      postJson<{ status: string; name: string }>(`/pastas/models/${runId}/scenarios`, body),
    deleteScenario: (runId: string, name: string) =>
      deleteJson(`/pastas/models/${runId}/scenarios/${encodeURIComponent(name)}`),
    adaptiveBounds: (runId: string, tFinalDays?: number) => {
      const qs = tFinalDays ? `?t_final_days=${tFinalDays}` : ''
      return fetchJson<import('./types').AdaptiveBoundsData>(`/pastas/models/${runId}/adaptive-bounds${qs}`)
    },
    diagnostics: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/diagnostics`),
    outlierDiagnostics: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/outlier-diagnostics`),
    confidenceBands: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/confidence-bands`),
    recession: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/recession`),
    baseflow: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/baseflow`),
    spectral: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/spectral`),
    decomposition: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/decomposition`),
    crossCorrelation: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/cross-correlation`),
    regionalResiduals: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/regional-residuals`),
    inputQuality: (runId: string) => fetchJson<Record<string, unknown>>(`/pastas/models/${runId}/input-quality`),
    signatures: (runId: string) => fetchJson<{ observed: Record<string, number>; simulated: Record<string, number>; categories?: Record<string, string[]> }>(`/pastas/models/${runId}/signatures`),
    compare: (runIds: string[]) =>
      postJson<PastasCompareResponse>('/pastas/compare', { run_ids: runIds }, 60_000),
    diagnose: (codeBss: string) =>
      postJson<DiagnoseResult>('/pastas/diagnose', { code_bss: codeBss }),
    autoFit: (body: {
      code_bss: string
      warm_up_years?: number
      val_split?: number
      include_temp?: boolean
      add_trend?: boolean | null
    }) => postJson<AutoFitResult>('/pastas/auto-fit', body, 300_000),
    compareAI: (body: { pastas_run_id: string; ai_model_id: string }) =>
      postJson<CompareAIResult>('/pastas/compare-ai', body),
  },

  counterfactual: {
    run: (body: {
      model_id: string
      method?: string
      target_ips_class?: string
      target_ips_classes?: Record<string, string>
      from_ips_class?: string
      to_ips_class?: string
      start_idx?: number
      modifications?: Record<string, number>
      lambda_prox?: number
      n_iter?: number
      lr?: number
      cc_rate?: number
      device?: string
      n_trials?: number
      seed?: number
      num_distractors?: number
      tau?: number
    }) => postJson<CounterfactualResult>('/counterfactual/run', body),
    stream: (taskId: string) =>
      new EventSource(`${API_BASE}/counterfactual/${taskId}/stream`),
    ipsReference: (modelId: string, window: number = 3) =>
      fetchJson<IPSReference>(`/counterfactual/ips-reference?model_id=${modelId}&window=${window}`),
    ipsBounds: (modelId: string, window: number = 1) =>
      fetchJson<IPSBoundsResponse>(`/counterfactual/ips-bounds?model_id=${modelId}&window=${window}`),
    pastasValidate: (body: { model_id: string; cf_task_id: string; gamma?: number }) =>
      postJson<PastasValidationResult>('/counterfactual/pastas-validate', body),
  },
}
