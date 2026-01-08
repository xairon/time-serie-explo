/**
 * Zustand store for global application state
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

// Types
export interface DatasetConfig {
    id: string
    name: string
    sourceType: 'csv' | 'database'
    sourceFile?: string
    dateColumn: string
    targetColumn: string
    covariateColumns: string[]
    stationColumn?: string
    stations?: string[]
    preprocessing: {
        fillMethod: string
        normalization: string
        datetimeFeatures: boolean
        lags: number[]
    }
}

export interface ModelInfo {
    modelId: string
    modelName: string
    modelType: 'single' | 'global'
    stations: string[]
    metrics?: Record<string, number>
    createdAt: string
    input_chunk_length?: number
    output_chunk_length?: number
}

export interface TrainingJob {
    jobId: string
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
    progress: number
    currentEpoch?: number
    totalEpochs?: number
    trainLoss?: number
    valLoss?: number
    modelId?: string
    error?: string
}

interface AppState {
    // UI state
    activePage: string
    setActivePage: (page: string) => void
    sidebarCollapsed: boolean
    toggleSidebar: () => void

    // Database connection
    dbConnection: {
        host: string
        port: number
        database: string
        user: string
        password: string
        connected: boolean
    } | null
    setDbConnection: (conn: AppState['dbConnection']) => void

    // Current dataset being prepared
    currentDataset: DatasetConfig | null
    setCurrentDataset: (dataset: DatasetConfig | null) => void

    // Raw data for exploration
    rawData: {
        columns: string[]
        rows: Record<string, unknown>[]
        totalRows: number
        sourceName: string
    } | null
    setRawData: (data: AppState['rawData']) => void

    // Training jobs
    activeJobs: TrainingJob[]
    addJob: (job: TrainingJob) => void
    updateJob: (jobId: string, updates: Partial<TrainingJob>) => void
    removeJob: (jobId: string) => void

    // Selected model for forecasting
    selectedModel: ModelInfo | null
    setSelectedModel: (model: ModelInfo | null) => void
}

export const useAppStore = create<AppState>()(
    persist(
        (set) => ({
            // UI state
            activePage: 'datasets',
            setActivePage: (page) => set({ activePage: page }),
            sidebarCollapsed: false,
            toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

            // Database connection
            dbConnection: null,
            setDbConnection: (conn) => set({ dbConnection: conn }),

            // Current dataset
            currentDataset: null,
            setCurrentDataset: (dataset) => set({ currentDataset: dataset }),

            // Raw data
            rawData: null,
            setRawData: (data) => set({ rawData: data }),

            // Training jobs
            activeJobs: [],
            addJob: (job) => set((s) => ({ activeJobs: [...s.activeJobs, job] })),
            updateJob: (jobId, updates) =>
                set((s) => ({
                    activeJobs: s.activeJobs.map((j) =>
                        j.jobId === jobId ? { ...j, ...updates } : j
                    ),
                })),
            removeJob: (jobId) =>
                set((s) => ({ activeJobs: s.activeJobs.filter((j) => j.jobId !== jobId) })),

            // Selected model
            selectedModel: null,
            setSelectedModel: (model) => set({ selectedModel: model }),
        }),
        {
            name: 'junon-storage',
            // Only persist UI preferences, not data state - resets on refresh
            partialize: (state) => ({
                sidebarCollapsed: state.sidebarCollapsed,
            }),
        }
    )
)
