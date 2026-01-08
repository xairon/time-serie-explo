/**
 * API Client for Junon Time Series Backend
 */

const API_BASE = '/api/v1';

export interface ApiError {
    detail: string;
}

async function handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
        const error: ApiError = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'API request failed');
    }
    return response.json();
}

// ============================================
// Data Sources
// ============================================

export interface DatabaseConnection {
    host: string;
    port: number;
    database: string;
    user: string;
    password: string;
}

export interface TableInfo {
    name: string;
    schema: string;
    type: 'table' | 'view';
    row_count?: number;
}

export interface ColumnInfo {
    name: string;
    type: string;
    nullable: boolean;
    is_date: boolean;
    is_numeric: boolean;
}

export interface ConnectionResult {
    success: boolean;
    message: string;
    version?: string;
    schemas?: string[];
}

export const sourcesApi = {
    testConnection: (conn: DatabaseConnection) =>
        fetch(`${API_BASE}/sources/db/connect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(conn),
        }).then(handleResponse<ConnectionResult>),

    listTables: (conn: DatabaseConnection, schema = 'public') => {
        const params = new URLSearchParams({
            host: conn.host,
            port: String(conn.port),
            database: conn.database,
            user: conn.user,
            password: conn.password,
            schema,
        });
        return fetch(`${API_BASE}/sources/db/tables?${params}`).then(handleResponse<{ tables: TableInfo[]; views: TableInfo[] }>);
    },

    getTableSchema: (tableName: string, conn: DatabaseConnection, schema = 'public') => {
        const params = new URLSearchParams({
            host: conn.host,
            port: String(conn.port),
            database: conn.database,
            user: conn.user,
            password: conn.password,
            schema,
        });
        return fetch(`${API_BASE}/sources/db/schema/${tableName}?${params}`).then(handleResponse);
    },

    uploadCsv: (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        return fetch(`${API_BASE}/sources/upload`, {
            method: 'POST',
            body: formData,
        }).then(handleResponse);
    },

    queryTableData: (params: {
        host: string;
        port: number;
        database: string;
        user: string;
        password: string;
        schema: string;
        table_name: string;
        limit?: number;
    }) => fetch(`${API_BASE}/sources/db/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    }).then(handleResponse<{ columns: string[]; data: Record<string, unknown>[]; total_rows: number }>),

    getStations: (params: {
        host: string;
        port: number;
        database: string;
        user: string;
        password: string;
        schema: string;
        table_name: string;
        station_column: string;
        date_column?: string;
        lat_column?: string;
        lon_column?: string;
        limit?: number;
    }) => fetch(`${API_BASE}/sources/db/stations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    }).then(handleResponse<{ stations: Record<string, unknown>[]; total_stations: number }>),
};

// ============================================
// Datasets
// ============================================

export interface DatasetInfo {
    id: string;
    name: string;
    source_type: 'csv' | 'database';
    source_file?: string;
    date_column: string;
    target_column: string;
    covariate_columns: string[];
    station_column?: string;
    date_range: [string, string];
    row_count: number;
    created_at: string;
}

export const datasetsApi = {
    list: () =>
        fetch(`${API_BASE}/datasets`).then(handleResponse<{ datasets: DatasetInfo[]; total: number }>),

    get: (id: string) =>
        fetch(`${API_BASE}/datasets/${id}`).then(handleResponse<DatasetInfo>),

    preview: (id: string, limit = 100) =>
        fetch(`${API_BASE}/datasets/${id}/preview?limit=${limit}`).then(handleResponse),

    statistics: (id: string) =>
        fetch(`${API_BASE}/datasets/${id}/statistics`).then(handleResponse),

    delete: (id: string) =>
        fetch(`${API_BASE}/datasets/${id}`, { method: 'DELETE' }).then(handleResponse),
};

// ============================================
// Models
// ============================================

export interface ModelInfo {
    model_id: string;
    model_name: string;
    model_type: 'single' | 'global';
    stations: string[];
    primary_station?: string;
    hyperparams: Record<string, unknown>;
    input_chunk_length: number;
    output_chunk_length: number;
    metrics?: Record<string, number>;
    created_at: string;
    path: string;
}

export const modelsApi = {
    list: (params?: { model_type?: string; model_name?: string; station?: string }) => {
        const queryParams = new URLSearchParams();
        if (params?.model_type) queryParams.set('model_type', params.model_type);
        if (params?.model_name) queryParams.set('model_name', params.model_name);
        if (params?.station) queryParams.set('station', params.station);
        return fetch(`${API_BASE}/models?${queryParams}`).then(handleResponse<{ models: ModelInfo[]; total: number }>);
    },

    get: (id: string) =>
        fetch(`${API_BASE}/models/${id}`).then(handleResponse<ModelInfo>),

    delete: (id: string) =>
        fetch(`${API_BASE}/models/${id}`, { method: 'DELETE' }).then(handleResponse),

    metrics: (id: string) =>
        fetch(`${API_BASE}/models/${id}/metrics`).then(handleResponse),
};

// ============================================
// Training
// ============================================

export interface TrainingConfig {
    model_name: string;
    train_ratio: number;
    val_ratio: number;
    input_chunk_length: number;
    output_chunk_length: number;
    epochs: number;
    batch_size: number;
    learning_rate: number;
    model_hyperparams?: Record<string, unknown>;
}

export interface TrainingRequest {
    dataset_id: string;
    stations: string[];
    training_strategy: 'independent' | 'global';
    config: TrainingConfig;
    use_covariates: boolean;
    save_model: boolean;
}

export interface TrainingJobStatus {
    job_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    current_epoch?: number;
    total_epochs?: number;
    train_loss?: number;
    val_loss?: number;
    started_at?: string;
    completed_at?: string;
    model_id?: string;
    error_message?: string;
}

export const trainingApi = {
    start: (request: TrainingRequest) =>
        fetch(`${API_BASE}/training/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        }).then(handleResponse<{ job_id: string; status: string; message: string }>),

    status: (jobId: string) =>
        fetch(`${API_BASE}/training/${jobId}/status`).then(handleResponse<TrainingJobStatus>),

    cancel: (jobId: string) =>
        fetch(`${API_BASE}/training/${jobId}`, { method: 'DELETE' }).then(handleResponse),

    list: (status?: string) => {
        const params = status ? `?status=${status}` : '';
        return fetch(`${API_BASE}/training${params}`).then(handleResponse<{ jobs: TrainingJobStatus[]; total: number }>);
    },
};

// ============================================
// WebSocket for Training Progress
// ============================================

export interface TrainingProgressEvent {
    event_type: 'progress' | 'metric' | 'log' | 'completed' | 'error';
    job_id: string;
    epoch?: number;
    total_epochs?: number;
    progress?: number;
    train_loss?: number;
    val_loss?: number;
    message?: string;
    model_id?: string;
    error?: string;
}

export function connectTrainingWebSocket(
    jobId: string,
    onMessage: (event: TrainingProgressEvent) => void,
    onError?: (error: Event) => void,
    onClose?: () => void
): WebSocket {
    const ws = new WebSocket(`ws://${window.location.host}/ws/training/${jobId}`);

    ws.onmessage = (event) => {
        const data: TrainingProgressEvent = JSON.parse(event.data);
        onMessage(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
        onClose?.();
    };

    return ws;
}

// ============================================
// Forecasting
// ============================================

export interface ForecastRequest {
    model_id: string;
    start_date: string;
    horizon: number;
    use_covariates: boolean;
}

export interface ForecastResponse {
    model_id: string;
    station: string;
    start_date: string;
    horizon: number;
    dates: string[];
    values: number[];
    lower_bound?: number[];
    upper_bound?: number[];
    metrics?: Record<string, number>;
}

export const forecastingApi = {
    predict: (request: ForecastRequest) =>
        fetch(`${API_BASE}/forecasting/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        }).then(handleResponse<ForecastResponse>),

    availableModels: () =>
        fetch(`${API_BASE}/forecasting/available-models`).then(handleResponse<{ models: ModelInfo[] }>),
};

// ============================================
// Health Check
// ============================================

export const healthApi = {
    check: () =>
        fetch('/health').then(handleResponse<{ status: string; version: string }>),
};
