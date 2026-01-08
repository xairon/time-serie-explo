/**
 * Datasets Page - Data loading, exploration, and configuration
 * Replicates functionality from 1_Dataset_Preparation.py
 */

import React, { useState, useCallback, useMemo, useEffect } from 'react'
import Plot from 'react-plotly.js'
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import {
    Upload,
    Database,
    FolderOpen,
    Table,
    BarChart2,
    AlertTriangle,
    Settings,
    Check,
    FileSpreadsheet,
    RefreshCw,
    MapPin,
} from 'lucide-react'
import { Card, Button, Input, Select, Tabs, Badge, Metric, DataTable, Progress, EmptyState } from '../components/ui'
import { useAppStore } from '../store/appStore'
import { datasetsApi, sourcesApi } from '../api/client'

// Fix Leaflet default marker icon issue
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

type TabId = 'upload' | 'database' | 'saved'
type ExploreTab = 'preview' | 'statistics' | 'quality' | 'map' | 'configure'

export function DatasetsPage() {
    const { rawData, setRawData, dbConnection, setDbConnection, currentDataset, setCurrentDataset } = useAppStore()

    // Source selection
    const [activeSourceTab, setActiveSourceTab] = useState<TabId>('upload')
    const [exploreTab, setExploreTab] = useState<ExploreTab>('preview')

    // Upload state
    const [uploading, setUploading] = useState(false)
    const [uploadError, setUploadError] = useState<string | null>(null)

    // Database state
    const [dbForm, setDbForm] = useState({
        host: 'localhost',
        port: '5433',
        database: 'postgres',
        user: 'postgres',
        password: 'postgres_default_pass_2024',
        schema: 'staging',
    })
    const [connecting, setConnecting] = useState(false)
    const [connectionError, setConnectionError] = useState<string | null>(null)
    const [availableSchemas, setAvailableSchemas] = useState<string[]>(['public'])
    const [tables, setTables] = useState<{ name: string; type: string }[]>([])
    const [selectedTable, setSelectedTable] = useState('')

    // Saved datasets
    const [savedDatasets, setSavedDatasets] = useState<{ id: string; name: string }[]>([])
    const [loadingDatasets, setLoadingDatasets] = useState(false)

    // Configuration state
    const [config, setConfig] = useState({
        dateColumn: '',
        targetColumn: '',
        covariateColumns: [] as string[],
        stationColumn: '',
        hasMultipleStations: false,
        fillMethod: 'interpolation',
        normalization: 'minmax',
        addTimeFeatures: false,
        addLags: false,
        lagValues: '1,7,30',
    })

    // Statistics
    const [stats, setStats] = useState<Record<string, number>>({})

    // File upload handler
    const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return

        setUploading(true)
        setUploadError(null)

        try {
            const result = await sourcesApi.uploadCsv(file) as { columns: string[]; path: string }

            // Read the full file for preview
            const text = await file.text()
            const lines = text.split('\n')
            const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''))

            const rows = lines.slice(1, 101).map(line => {
                const values = line.split(',').map(v => v.trim().replace(/"/g, ''))
                const row: Record<string, string> = {}
                headers.forEach((h, i) => { row[h] = values[i] || '' })
                return row
            }).filter(row => Object.values(row).some(v => v))

            setRawData({
                columns: result.columns || headers,
                rows,
                totalRows: lines.length - 1,
                sourceName: file.name,
            })

            // Auto-detect columns
            const dateCol = headers.find(h =>
                /date|time|timestamp|jour|datetime/i.test(h)
            ) || headers[0]

            const numericCols = headers.filter(h => {
                const vals = rows.slice(0, 10).map(r => r[h])
                return vals.some(v => !isNaN(parseFloat(v)))
            })

            setConfig(prev => ({
                ...prev,
                dateColumn: dateCol,
                targetColumn: numericCols[0] || '',
                covariateColumns: numericCols.slice(1, 6),
            }))

        } catch (err) {
            setUploadError(err instanceof Error ? err.message : 'Upload failed')
        } finally {
            setUploading(false)
        }
    }, [setRawData])

    // Database connection
    const handleConnect = useCallback(async () => {
        setConnecting(true)
        setConnectionError(null)
        try {
            console.log('Connecting to database...', dbForm)
            const result = await sourcesApi.testConnection({
                host: dbForm.host,
                port: parseInt(dbForm.port),
                database: dbForm.database,
                user: dbForm.user,
                password: dbForm.password,
            })
            console.log('Connection result:', result)

            if (result.success) {
                // Save available schemas
                if (result.schemas && result.schemas.length > 0) {
                    setAvailableSchemas(result.schemas)
                }

                setDbConnection({
                    ...dbForm,
                    port: parseInt(dbForm.port),
                    connected: true,
                })

                // Fetch tables for selected schema
                const tablesResult = await sourcesApi.listTables(
                    {
                        host: dbForm.host,
                        port: parseInt(dbForm.port),
                        database: dbForm.database,
                        user: dbForm.user,
                        password: dbForm.password,
                    },
                    dbForm.schema
                )

                setTables([
                    ...tablesResult.tables.map((t: { name: string }) => ({ name: t.name, type: 'table' })),
                    ...tablesResult.views.map((v: { name: string }) => ({ name: v.name, type: 'view' })),
                ])
            } else {
                setConnectionError(result.message || 'Connection failed')
            }
        } catch (err) {
            console.error('Connection error:', err)
            setConnectionError(err instanceof Error ? err.message : 'Connection failed')
        } finally {
            setConnecting(false)
        }
    }, [dbForm, setDbConnection])

    // Reload tables when schema changes (while connected)
    const loadTablesForSchema = useCallback(async (schema: string) => {
        if (!dbConnection) return
        try {
            const tablesResult = await sourcesApi.listTables(
                {
                    host: dbConnection.host,
                    port: dbConnection.port,
                    database: dbConnection.database,
                    user: dbConnection.user,
                    password: dbConnection.password,
                },
                schema
            )
            setTables([
                ...tablesResult.tables.map((t: { name: string }) => ({ name: t.name, type: 'table' })),
                ...tablesResult.views.map((v: { name: string }) => ({ name: v.name, type: 'view' })),
            ])
            setSelectedTable('') // Reset selection when schema changes
        } catch (err) {
            console.error('Failed to load tables for schema:', err)
        }
    }, [dbConnection])

    // Effect: reload tables when schema changes
    useEffect(() => {
        if (dbConnection?.connected && dbForm.schema) {
            loadTablesForSchema(dbForm.schema)
        }
    }, [dbForm.schema, dbConnection?.connected, loadTablesForSchema])

    // Load table data from database
    const [loadingTable, setLoadingTable] = useState(false)
    const handleLoadTableData = useCallback(async () => {
        if (!dbConnection || !selectedTable) return

        setLoadingTable(true)
        try {
            console.log('Loading table data:', selectedTable)

            // Query the table data via API
            const response = await fetch(`/api/v1/sources/db/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    host: dbConnection.host,
                    port: dbConnection.port,
                    database: dbConnection.database,
                    user: dbConnection.user,
                    password: dbConnection.password,
                    schema: dbForm.schema,
                    table_name: selectedTable,
                    limit: 1000,
                }),
            })

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}))
                const errorMsg = typeof errorData.detail === 'string'
                    ? errorData.detail
                    : (errorData.detail?.msg || JSON.stringify(errorData.detail) || `HTTP ${response.status}`)
                throw new Error(errorMsg)
            }

            const result = await response.json()

            // Set raw data for exploration
            setRawData({
                sourceName: `${dbConnection.database}.${dbForm.schema}.${selectedTable}`,
                columns: result.columns,
                rows: result.data,
                totalRows: result.total_rows,
            })

            // Auto-detect column types
            const datePatterns = ['date', 'time', 'timestamp', 'datetime', 'dt']
            const dateCol = result.columns.find((c: string) =>
                datePatterns.some(p => c.toLowerCase().includes(p))
            ) || result.columns[0]

            const numericCols = result.columns.filter((c: string) => {
                const vals = result.data.slice(0, 10).map((r: Record<string, any>) => r[c])
                return vals.some((v: any) => !isNaN(parseFloat(String(v))))
            })

            setConfig(prev => ({
                ...prev,
                dateColumn: dateCol,
                targetColumn: numericCols[0] || '',
                covariateColumns: numericCols.slice(1, 6),
            }))

            console.log('Table data loaded:', result.total_rows, 'rows')
        } catch (err) {
            console.error('Failed to load table data:', err)
            const errorMessage = err instanceof Error
                ? err.message
                : (typeof err === 'string' ? err : 'Failed to load table data')
            alert(errorMessage)
        } finally {
            setLoadingTable(false)
        }
    }, [dbConnection, selectedTable, dbForm.schema, setRawData])


    // Map State
    const [mapStations, setMapStations] = useState<any[]>([])
    const [mapLoading, setMapLoading] = useState(false)
    const [mapConfig, setMapConfig] = useState({
        stationCol: '',
        latCol: '',
        lonCol: ''
    })

    // Auto-detect map columns when rawData changes
    useEffect(() => {
        if (rawData?.columns) {
            const cols = rawData.columns.map(c => c.toLowerCase())
            const station = rawData.columns.find(c => ['station', 'code', 'site', 'bss'].some(k => c.toLowerCase().includes(k))) || ''
            const lat = rawData.columns.find(c => ['lat', 'y', 'nord'].some(k => c.toLowerCase().includes(k))) || ''
            const lon = rawData.columns.find(c => ['lon', 'x', 'est', 'lng'].some(k => c.toLowerCase().includes(k))) || ''

            setMapConfig({
                stationCol: station,
                latCol: lat,
                lonCol: lon
            })
        }
    }, [rawData])

    // Fetch full station list for map
    const handleLoadMapData = useCallback(async () => {
        if (!dbConnection || !selectedTable || !mapConfig.stationCol) return

        setMapLoading(true)
        try {
            const result = await sourcesApi.getStations({
                host: dbConnection.host,
                port: dbConnection.port,
                database: dbConnection.database,
                user: dbConnection.user,
                password: dbConnection.password,
                schema: dbForm.schema,
                table_name: selectedTable,
                station_column: mapConfig.stationCol,
                lat_column: mapConfig.latCol || undefined,
                lon_column: mapConfig.lonCol || undefined,
                limit: 5000
            })
            setMapStations(result.stations)
        } catch (err: any) {
            console.error(err)
            alert(`Error loading map data: ${err.message}`)
        } finally {
            setMapLoading(false)
        }
    }, [dbConnection, selectedTable, dbForm.schema, mapConfig])

    // Auto-load map data when tab is switched and config is ready
    useEffect(() => {
        if (exploreTab === 'map' && mapConfig.stationCol && mapStations.length === 0 && !mapLoading) {
            handleLoadMapData()
        }
    }, [exploreTab, mapConfig.stationCol, mapStations.length, mapLoading, handleLoadMapData])

    // Load saved datasets
    const loadSavedDatasets = useCallback(async () => {
        setLoadingDatasets(true)
        try {
            const result = await datasetsApi.list()
            setSavedDatasets(result.datasets.map(d => ({ id: d.id, name: d.name })))
        } catch (err) {
            console.error(err)
        } finally {
            setLoadingDatasets(false)
        }
    }, [])

    // Validate and save configuration
    const handleValidate = useCallback(() => {
        if (!rawData) return

        setCurrentDataset({
            id: `dataset-${Date.now()}`,
            name: rawData.sourceName.replace('.csv', ''),
            sourceType: 'csv',
            sourceFile: rawData.sourceName,
            dateColumn: config.dateColumn,
            targetColumn: config.targetColumn,
            covariateColumns: config.covariateColumns,
            stationColumn: config.hasMultipleStations ? config.stationColumn : undefined,
            preprocessing: {
                fillMethod: config.fillMethod,
                normalization: config.normalization,
                datetimeFeatures: config.addTimeFeatures,
                lags: config.addLags ? config.lagValues.split(',').map(v => parseInt(v.trim())) : [],
            },
        })
    }, [rawData, config, setCurrentDataset])

    // If dataset is configured, show summary
    if (currentDataset) {
        return (
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold text-white">Dataset Ready</h1>
                        <p className="text-slate-400 mt-1">Your dataset is configured and ready for training</p>
                    </div>
                    <Button variant="secondary" onClick={() => setCurrentDataset(null)}>
                        <RefreshCw className="w-4 h-4" />
                        Prepare New Dataset
                    </Button>
                </div>

                <Card className="p-6">
                    <div className="flex items-center gap-2 mb-6">
                        <Check className="w-6 h-6 text-green-400" />
                        <h2 className="text-xl font-semibold text-white">{currentDataset.name}</h2>
                        <Badge variant="success">Ready</Badge>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
                        <div>
                            <span className="text-sm text-slate-500 block mb-1">Source</span>
                            <div className="text-white font-medium truncate" title={currentDataset.sourceFile || 'Database'}>
                                {currentDataset.sourceFile?.split(/[/\\]/).pop() || 'Database'}
                            </div>
                        </div>
                        <div>
                            <span className="text-sm text-slate-500 block mb-1">Target Variable</span>
                            <div className="text-cyan-400 font-medium truncate" title={currentDataset.targetColumn}>
                                {currentDataset.targetColumn}
                            </div>
                        </div>
                        <div>
                            <span className="text-sm text-slate-500 block mb-1">Normalization</span>
                            <Badge variant="info" className="uppercase text-xs">
                                {currentDataset.preprocessing.normalization}
                            </Badge>
                        </div>
                        <div>
                            <span className="text-sm text-slate-500 block mb-1">Covariates</span>
                            <div className="text-white font-medium">
                                {currentDataset.covariateColumns.length} features
                            </div>
                        </div>
                        <div>
                            <span className="text-sm text-slate-500 block mb-1">Time Config</span>
                            <div className="text-slate-300 text-sm">
                                {currentDataset.preprocessing.datetimeFeatures ? 'Date Features On' : 'Date Features Off'}
                            </div>
                        </div>
                        <div>
                            <span className="text-sm text-slate-500 block mb-1">Lags</span>
                            <div className="text-slate-300 text-sm">
                                {currentDataset.preprocessing.lags?.length
                                    ? `${currentDataset.preprocessing.lags.length} lag sets`
                                    : 'None'}
                            </div>
                        </div>
                        <div>
                            <span className="text-sm text-slate-500 block mb-1">Imputation</span>
                            <div className="text-slate-300 text-sm">
                                {currentDataset.preprocessing.fillMethod}
                            </div>
                        </div>
                    </div>

                    <div className="flex gap-3 pt-6 border-t border-slate-700/50">
                        <Button onClick={() => useAppStore.getState().setActivePage('training')} className="px-8">
                            🚀 Go to Training
                        </Button>
                    </div>
                </Card>
            </div>
        )
    }

    // If raw data loaded, show exploration
    if (rawData) {
        return (
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold text-white">Exploring: {rawData.sourceName}</h1>
                        <p className="text-slate-400 mt-1">{rawData.totalRows.toLocaleString()} rows • {rawData.columns.length} columns</p>
                    </div>
                    <Button variant="secondary" onClick={() => setRawData(null)}>
                        ← Back to Sources
                    </Button>
                </div>

                <Tabs
                    tabs={[
                        { id: 'preview', label: 'Preview', icon: <Table className="w-4 h-4" /> },
                        { id: 'statistics', label: 'Statistics', icon: <BarChart2 className="w-4 h-4" /> },
                        { id: 'quality', label: 'Quality', icon: <AlertTriangle className="w-4 h-4" /> },
                        { id: 'map', label: 'Map', icon: <MapPin className="w-4 h-4" /> },
                        { id: 'configure', label: 'Configure', icon: <Settings className="w-4 h-4" /> },
                    ]}
                    activeTab={exploreTab}
                    onChange={(id) => setExploreTab(id as ExploreTab)}
                />

                <Card className="p-6">
                    {/* Preview Tab */}
                    {exploreTab === 'preview' && (
                        <DataTable
                            columns={rawData.columns.map(c => ({ key: c, label: c }))}
                            data={rawData.rows}
                            maxRows={100}
                        />
                    )}

                    {/* Statistics Tab */}
                    {exploreTab === 'statistics' && (() => {
                        // Compute stats for each numeric column
                        const numericColumns = rawData.columns.filter(col => {
                            const vals = rawData.rows.slice(0, 20).map(r => parseFloat(String(r[col])))
                            return vals.filter(v => !isNaN(v)).length > vals.length / 2
                        })

                        const computeStats = (col: string) => {
                            const values = rawData.rows.map(r => parseFloat(String(r[col]))).filter(v => !isNaN(v))
                            if (values.length === 0) return null
                            const sorted = [...values].sort((a, b) => a - b)
                            const sum = values.reduce((a, b) => a + b, 0)
                            const mean = sum / values.length
                            const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length
                            const std = Math.sqrt(variance)
                            const q25 = sorted[Math.floor(values.length * 0.25)]
                            const q50 = sorted[Math.floor(values.length * 0.5)]
                            const q75 = sorted[Math.floor(values.length * 0.75)]
                            return {
                                count: values.length,
                                min: Math.min(...values),
                                max: Math.max(...values),
                                mean,
                                std,
                                q25,
                                q50,
                                q75,
                            }
                        }

                        return (
                            <div className="space-y-6">
                                <div className="grid grid-cols-4 gap-4">
                                    <Metric label="Total Rows" value={rawData.totalRows.toLocaleString()} />
                                    <Metric label="Columns" value={rawData.columns.length} />
                                    <Metric label="Numeric" value={numericColumns.length} />
                                    <Metric label="Categorical" value={rawData.columns.length - numericColumns.length} />
                                </div>

                                {/* Stats Table */}
                                <div className="overflow-x-auto">
                                    <table className="w-full text-sm">
                                        <thead>
                                            <tr className="border-b border-slate-700">
                                                <th className="text-left py-2 px-3 text-slate-400 font-medium">Column</th>
                                                <th className="text-right py-2 px-3 text-slate-400 font-medium">Count</th>
                                                <th className="text-right py-2 px-3 text-slate-400 font-medium">Min</th>
                                                <th className="text-right py-2 px-3 text-slate-400 font-medium">Max</th>
                                                <th className="text-right py-2 px-3 text-slate-400 font-medium">Mean</th>
                                                <th className="text-right py-2 px-3 text-slate-400 font-medium">Std</th>
                                                <th className="text-right py-2 px-3 text-slate-400 font-medium">Q25</th>
                                                <th className="text-right py-2 px-3 text-slate-400 font-medium">Median</th>
                                                <th className="text-right py-2 px-3 text-slate-400 font-medium">Q75</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {numericColumns.map(col => {
                                                const stats = computeStats(col)
                                                if (!stats) return null
                                                const fmt = (v: number) => v.toFixed(v > 1000 ? 0 : 2)
                                                return (
                                                    <tr key={col} className="border-b border-slate-800 hover:bg-slate-800/30">
                                                        <td className="py-2 px-3 text-white font-medium">{col}</td>
                                                        <td className="py-2 px-3 text-right text-slate-300">{stats.count}</td>
                                                        <td className="py-2 px-3 text-right text-cyan-400">{fmt(stats.min)}</td>
                                                        <td className="py-2 px-3 text-right text-cyan-400">{fmt(stats.max)}</td>
                                                        <td className="py-2 px-3 text-right text-blue-400">{fmt(stats.mean)}</td>
                                                        <td className="py-2 px-3 text-right text-slate-400">{fmt(stats.std)}</td>
                                                        <td className="py-2 px-3 text-right text-purple-400">{fmt(stats.q25)}</td>
                                                        <td className="py-2 px-3 text-right text-purple-400">{fmt(stats.q50)}</td>
                                                        <td className="py-2 px-3 text-right text-purple-400">{fmt(stats.q75)}</td>
                                                    </tr>
                                                )
                                            })}
                                        </tbody>
                                    </table>
                                </div>

                                {/* Histograms for first 4 numeric columns */}
                                <div className="grid grid-cols-2 gap-4">
                                    {numericColumns.slice(0, 4).map(col => {
                                        const values = rawData.rows.map(r => parseFloat(String(r[col]))).filter(v => !isNaN(v))
                                        if (values.length === 0) return null
                                        return (
                                            <div key={col} className="bg-slate-800/30 rounded-xl p-4">
                                                <h4 className="text-sm font-medium text-slate-300 mb-2">{col}</h4>
                                                <Plot
                                                    data={[{
                                                        x: values,
                                                        type: 'histogram',
                                                        marker: { color: '#3b82f6' },
                                                    } as Plotly.Data]}
                                                    layout={{
                                                        height: 180,
                                                        margin: { t: 10, r: 10, b: 30, l: 40 },
                                                        paper_bgcolor: 'transparent',
                                                        plot_bgcolor: 'transparent',
                                                        xaxis: { color: '#94a3b8', gridcolor: '#334155' },
                                                        yaxis: { color: '#94a3b8', gridcolor: '#334155' },
                                                    }}
                                                    config={{ displayModeBar: false }}
                                                    style={{ width: '100%' }}
                                                />
                                            </div>
                                        )
                                    })}
                                </div>
                            </div>
                        )
                    })()}

                    {/* Quality Tab */}
                    {exploreTab === 'quality' && (() => {
                        // Compute quality metrics
                        const totalCells = rawData.rows.length * rawData.columns.length
                        let missingCells = 0
                        const columnQuality: Record<string, { missing: number; unique: number; outliers: number }> = {}

                        rawData.columns.forEach(col => {
                            const values = rawData.rows.map(r => r[col])
                            const missing = values.filter(v => v === null || v === undefined || v === '' || v === 'null' || v === 'NaN').length
                            const unique = new Set(values.filter(v => v != null && v !== '')).size

                            // Detect outliers for numeric columns using IQR
                            let outliers = 0
                            const numericVals = values.map(v => parseFloat(String(v))).filter(v => !isNaN(v))
                            if (numericVals.length > 10) {
                                const sorted = [...numericVals].sort((a, b) => a - b)
                                const q1 = sorted[Math.floor(numericVals.length * 0.25)]
                                const q3 = sorted[Math.floor(numericVals.length * 0.75)]
                                const iqr = q3 - q1
                                outliers = numericVals.filter(v => v < q1 - 1.5 * iqr || v > q3 + 1.5 * iqr).length
                            }

                            missingCells += missing
                            columnQuality[col] = { missing, unique, outliers }
                        })

                        const completeness = ((totalCells - missingCells) / totalCells * 100)

                        // Detect duplicate rows (based on first 5 columns)
                        const rowKeys = rawData.rows.map(r => rawData.columns.slice(0, 5).map(c => r[c]).join('|'))
                        const uniqueRows = new Set(rowKeys).size
                        const duplicateRows = rawData.rows.length - uniqueRows
                        const duplicatePct = (duplicateRows / rawData.rows.length * 100)

                        // Columns with issues
                        const problematicCols = Object.entries(columnQuality).filter(([_, q]) =>
                            q.missing > rawData.rows.length * 0.1 || q.outliers > rawData.rows.length * 0.05
                        )

                        return (
                            <div className="space-y-6">
                                {/* Overview Metrics */}
                                <div className="grid grid-cols-4 gap-4">
                                    <div className="bg-slate-800/30 rounded-xl p-4">
                                        <p className="text-sm text-slate-400">Completeness</p>
                                        <p className={`text-2xl font-bold ${completeness > 95 ? 'text-green-400' : completeness > 80 ? 'text-yellow-400' : 'text-red-400'}`}>
                                            {completeness.toFixed(1)}%
                                        </p>
                                    </div>
                                    <div className="bg-slate-800/30 rounded-xl p-4">
                                        <p className="text-sm text-slate-400">Missing Cells</p>
                                        <p className="text-2xl font-bold text-white">{missingCells.toLocaleString()}</p>
                                        <p className="text-xs text-slate-500">of {totalCells.toLocaleString()}</p>
                                    </div>
                                    <div className="bg-slate-800/30 rounded-xl p-4">
                                        <p className="text-sm text-slate-400">Duplicate Rows</p>
                                        <p className={`text-2xl font-bold ${duplicatePct < 1 ? 'text-green-400' : duplicatePct < 5 ? 'text-yellow-400' : 'text-red-400'}`}>
                                            {duplicatePct.toFixed(1)}%
                                        </p>
                                        <p className="text-xs text-slate-500">{duplicateRows} rows</p>
                                    </div>
                                    <div className="bg-slate-800/30 rounded-xl p-4">
                                        <p className="text-sm text-slate-400">Problematic Columns</p>
                                        <p className={`text-2xl font-bold ${problematicCols.length === 0 ? 'text-green-400' : 'text-yellow-400'}`}>
                                            {problematicCols.length}
                                        </p>
                                        <p className="text-xs text-slate-500">of {rawData.columns.length}</p>
                                    </div>
                                </div>

                                {/* Column Quality Table */}
                                <div>
                                    <h4 className="text-lg font-medium text-white mb-4">Column Quality Analysis</h4>
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-sm">
                                            <thead>
                                                <tr className="border-b border-slate-700">
                                                    <th className="text-left py-2 px-3 text-slate-400 font-medium">Column</th>
                                                    <th className="text-right py-2 px-3 text-slate-400 font-medium">Missing</th>
                                                    <th className="text-right py-2 px-3 text-slate-400 font-medium">Missing %</th>
                                                    <th className="text-right py-2 px-3 text-slate-400 font-medium">Unique</th>
                                                    <th className="text-right py-2 px-3 text-slate-400 font-medium">Outliers</th>
                                                    <th className="text-center py-2 px-3 text-slate-400 font-medium">Status</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {Object.entries(columnQuality).map(([col, q]) => {
                                                    const missingPct = (q.missing / rawData.rows.length * 100)
                                                    const hasIssue = q.missing > rawData.rows.length * 0.1 || q.outliers > rawData.rows.length * 0.05
                                                    return (
                                                        <tr key={col} className="border-b border-slate-800 hover:bg-slate-800/30">
                                                            <td className="py-2 px-3 text-white font-medium">{col}</td>
                                                            <td className="py-2 px-3 text-right text-slate-300">{q.missing}</td>
                                                            <td className={`py-2 px-3 text-right ${missingPct > 10 ? 'text-red-400' : missingPct > 0 ? 'text-yellow-400' : 'text-green-400'}`}>
                                                                {missingPct.toFixed(1)}%
                                                            </td>
                                                            <td className="py-2 px-3 text-right text-slate-300">{q.unique}</td>
                                                            <td className={`py-2 px-3 text-right ${q.outliers > 0 ? 'text-orange-400' : 'text-slate-400'}`}>
                                                                {q.outliers}
                                                            </td>
                                                            <td className="py-2 px-3 text-center">
                                                                {hasIssue
                                                                    ? <span className="text-yellow-400">⚠️</span>
                                                                    : <span className="text-green-400">✓</span>
                                                                }
                                                            </td>
                                                        </tr>
                                                    )
                                                })}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>

                                {/* Recommendations */}
                                {problematicCols.length > 0 && (
                                    <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4">
                                        <h4 className="font-medium text-yellow-400 mb-2">⚠️ Recommendations</h4>
                                        <ul className="text-sm text-slate-300 space-y-1">
                                            {problematicCols.map(([col, q]) => (
                                                <li key={col}>
                                                    <strong>{col}</strong>:
                                                    {q.missing > rawData.rows.length * 0.1 && ` ${(q.missing / rawData.rows.length * 100).toFixed(0)}% missing values - consider imputation`}
                                                    {q.outliers > rawData.rows.length * 0.05 && ` ${q.outliers} outliers detected - review for errors`}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </div>
                        )
                    })()}

                    {/* Map Tab */}
                    {exploreTab === 'map' && (
                        <div className="space-y-4">
                            <div className="grid grid-cols-4 gap-4 bg-slate-800/30 p-4 rounded-xl">
                                <Select
                                    label="Station ID Column"
                                    value={mapConfig.stationCol}
                                    onChange={v => setMapConfig(prev => ({ ...prev, stationCol: v }))}
                                    options={rawData.columns.map(c => ({ value: c, label: c }))}
                                    placeholder="Select station column"
                                />
                                <Select
                                    label="Latitude Column"
                                    value={mapConfig.latCol}
                                    onChange={v => setMapConfig(prev => ({ ...prev, latCol: v }))}
                                    options={rawData.columns.map(c => ({ value: c, label: c }))}
                                    placeholder="Select latitude"
                                />
                                <Select
                                    label="Longitude Column"
                                    value={mapConfig.lonCol}
                                    onChange={v => setMapConfig(prev => ({ ...prev, lonCol: v }))}
                                    options={rawData.columns.map(c => ({ value: c, label: c }))}
                                    placeholder="Select longitude"
                                />
                                <div className="flex items-end">
                                    <Button
                                        onClick={handleLoadMapData}
                                        loading={mapLoading}
                                        disabled={!mapConfig.stationCol}
                                        className="w-full"
                                    >
                                        <RefreshCw className="w-4 h-4" />
                                        Refresh Map Data
                                    </Button>
                                </div>
                            </div>

                            <div className="h-[600px] rounded-xl overflow-hidden relative z-0 border border-slate-700/50">
                                {mapStations.length > 0 ? (
                                    <MapContainer
                                        center={[
                                            mapStations.reduce((acc, s) => acc + (s.latitude || 46.2276), 0) / mapStations.length || 46.2276,
                                            mapStations.reduce((acc, s) => acc + (s.longitude || 2.2137), 0) / mapStations.length || 2.2137
                                        ]}
                                        zoom={6}
                                        style={{ height: '100%', width: '100%' }}
                                    >
                                        <TileLayer
                                            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                                            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                                        />
                                        {mapStations.map((station, idx) => (
                                            station.latitude && station.longitude && (
                                                <Marker
                                                    key={idx}
                                                    position={[station.latitude, station.longitude]}
                                                    eventHandlers={{
                                                        click: () => {
                                                            // Logic to select station and load its timeseries will be implemented in next step
                                                            console.log('Clicked station:', station)
                                                        },
                                                    }}
                                                >
                                                    <Popup>
                                                        <div className="text-slate-900 min-w-[200px]">
                                                            <strong className="text-lg">{station.station}</strong>
                                                            <div className="mt-2 text-sm text-slate-600 space-y-1">
                                                                <p>Available Records: <b>{station.row_count}</b></p>
                                                                {station.min_date && <p>Range: {station.min_date} to {station.max_date}</p>}
                                                            </div>
                                                            <button
                                                                className="mt-3 w-full px-3 py-1 bg-blue-600 text-white rounded text-xs font-medium hover:bg-blue-700"
                                                                onClick={(e) => {
                                                                    e.stopPropagation()
                                                                    // TODO: Load station charts
                                                                }}
                                                            >
                                                                Analyze Station Data
                                                            </button>
                                                        </div>
                                                    </Popup>
                                                </Marker>
                                            )
                                        ))}
                                    </MapContainer>
                                ) : (
                                    <div className="flex items-center justify-center h-full bg-slate-900/50">
                                        <div className="text-center">
                                            <MapPin className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                                            <h3 className="text-lg font-medium text-slate-300">Map Not Configured</h3>
                                            <p className="text-slate-400 mt-2 max-w-sm mx-auto">
                                                Select the columns containing Station ID, Latitude, and Longitude above, then click Refresh.
                                            </p>
                                        </div>
                                    </div>
                                )}
                            </div>

                            <div className="flex justify-between text-sm text-slate-400 px-2">
                                <span>{mapStations.length} stations loaded</span>
                                <span>{mapLoading ? 'Loading...' : 'Ready'}</span>
                            </div>
                        </div>
                    )}

                    {/* Configure Tab */}
                    {exploreTab === 'configure' && (
                        <div className="space-y-8">
                            {/* Time Column */}
                            <div>
                                <h4 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                                    <span className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center text-blue-400">1</span>
                                    Time Column
                                </h4>
                                <Select
                                    label="Select the column containing dates/timestamps"
                                    value={config.dateColumn}
                                    onChange={(v) => setConfig(prev => ({ ...prev, dateColumn: v }))}
                                    options={rawData.columns.map(c => ({ value: c, label: c }))}
                                />
                            </div>

                            {/* Station Column */}
                            <div>
                                <h4 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                                    <span className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center text-blue-400">2</span>
                                    Station Column (Optional)
                                </h4>
                                <label className="flex items-center gap-3 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={config.hasMultipleStations}
                                        onChange={(e) => setConfig(prev => ({ ...prev, hasMultipleStations: e.target.checked }))}
                                        className="w-5 h-5 rounded bg-slate-800 border-slate-600"
                                    />
                                    <span className="text-slate-300">Data contains multiple stations/locations</span>
                                </label>
                                {config.hasMultipleStations && (
                                    <Select
                                        label="Station identifier column"
                                        value={config.stationColumn}
                                        onChange={(v) => setConfig(prev => ({ ...prev, stationColumn: v }))}
                                        options={rawData.columns.filter(c => c !== config.dateColumn).map(c => ({ value: c, label: c }))}
                                        className="mt-4"
                                    />
                                )}
                            </div>

                            {/* Variables */}
                            <div>
                                <h4 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                                    <span className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center text-blue-400">3</span>
                                    Variables Selection
                                </h4>
                                <div className="grid grid-cols-2 gap-4">
                                    <Select
                                        label="🎯 Target Variable (to predict)"
                                        value={config.targetColumn}
                                        onChange={(v) => setConfig(prev => ({ ...prev, targetColumn: v }))}
                                        options={rawData.columns
                                            .filter(c => c !== config.dateColumn && c !== config.stationColumn)
                                            .map(c => ({ value: c, label: c }))}
                                    />
                                    <div>
                                        <label className="block text-sm font-medium text-slate-300 mb-1.5">
                                            📈 Covariates (features)
                                        </label>
                                        <div className="flex flex-wrap gap-2 p-3 bg-slate-900/50 rounded-xl border border-slate-700/50 max-h-32 overflow-y-auto">
                                            {rawData.columns
                                                .filter(c => c !== config.dateColumn && c !== config.stationColumn && c !== config.targetColumn)
                                                .map(col => (
                                                    <button
                                                        key={col}
                                                        onClick={() => setConfig(prev => ({
                                                            ...prev,
                                                            covariateColumns: prev.covariateColumns.includes(col)
                                                                ? prev.covariateColumns.filter(c => c !== col)
                                                                : [...prev.covariateColumns, col]
                                                        }))}
                                                        className={`px-2 py-1 rounded-lg text-sm transition-all ${config.covariateColumns.includes(col)
                                                            ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                                                            : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                                                            }`}
                                                    >
                                                        {col}
                                                    </button>
                                                ))}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Preprocessing */}
                            <div>
                                <h4 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                                    <span className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center text-blue-400">4</span>
                                    Preprocessing
                                </h4>
                                <div className="grid grid-cols-2 gap-4">
                                    <Select
                                        label="Missing Values"
                                        value={config.fillMethod}
                                        onChange={(v) => setConfig(prev => ({ ...prev, fillMethod: v }))}
                                        options={[
                                            { value: 'interpolation', label: 'Linear Interpolation' },
                                            { value: 'ffill', label: 'Forward Fill' },
                                            { value: 'bfill', label: 'Backward Fill' },
                                            { value: 'drop', label: 'Drop Rows' },
                                        ]}
                                    />
                                    <Select
                                        label="Normalization"
                                        value={config.normalization}
                                        onChange={(v) => setConfig(prev => ({ ...prev, normalization: v }))}
                                        options={[
                                            { value: 'minmax', label: 'MinMax (0-1)' },
                                            { value: 'standard', label: 'StandardScaler (z-score)' },
                                            { value: 'none', label: 'None' },
                                        ]}
                                    />
                                </div>
                                <div className="flex gap-6 mt-4">
                                    <label className="flex items-center gap-2 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={config.addTimeFeatures}
                                            onChange={(e) => setConfig(prev => ({ ...prev, addTimeFeatures: e.target.checked }))}
                                            className="w-4 h-4 rounded bg-slate-800 border-slate-600"
                                        />
                                        <span className="text-slate-300">🕐 Add Time Features</span>
                                    </label>
                                    <label className="flex items-center gap-2 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={config.addLags}
                                            onChange={(e) => setConfig(prev => ({ ...prev, addLags: e.target.checked }))}
                                            className="w-4 h-4 rounded bg-slate-800 border-slate-600"
                                        />
                                        <span className="text-slate-300">📉 Add Target Lags</span>
                                    </label>
                                </div>
                            </div>

                            {/* Actions */}
                            <div className="flex gap-4 pt-4 border-t border-slate-700/50">
                                <Button onClick={handleValidate} className="flex-1">
                                    <Check className="w-4 h-4" />
                                    Validate & Configure Dataset
                                </Button>
                            </div>
                        </div>
                    )}
                </Card>
            </div>
        )
    }

    // Data source selection
    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-3xl font-bold text-white">Dataset Preparation</h1>
                <p className="text-slate-400 mt-1">Load, explore, and prepare your time series data</p>
            </div>

            <Tabs
                tabs={[
                    { id: 'upload', label: 'Upload CSV', icon: <Upload className="w-4 h-4" /> },
                    { id: 'database', label: 'From Database', icon: <Database className="w-4 h-4" /> },
                    { id: 'saved', label: 'Saved Datasets', icon: <FolderOpen className="w-4 h-4" /> },
                ]}
                activeTab={activeSourceTab}
                onChange={(id) => {
                    setActiveSourceTab(id as TabId)
                    if (id === 'saved') loadSavedDatasets()
                }}
            />

            <Card className="p-8">
                {/* Upload Tab */}
                {activeSourceTab === 'upload' && (
                    <div className="space-y-6">
                        <div className="border-2 border-dashed border-slate-700 rounded-xl p-12 text-center hover:border-blue-500/50 transition-colors">
                            <input
                                type="file"
                                accept=".csv"
                                onChange={handleFileUpload}
                                className="hidden"
                                id="file-upload"
                                disabled={uploading}
                            />
                            <label htmlFor="file-upload" className="cursor-pointer">
                                <FileSpreadsheet className="w-12 h-12 mx-auto text-slate-500 mb-4" />
                                <p className="text-lg text-slate-300 mb-2">
                                    {uploading ? 'Uploading...' : 'Drop your CSV file here or click to browse'}
                                </p>
                                <p className="text-sm text-slate-500">
                                    Expected: date column, numeric target, optional covariates
                                </p>
                            </label>
                        </div>
                        {uploadError && (
                            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400">
                                {uploadError}
                            </div>
                        )}
                    </div>
                )}

                {/* Database Tab */}
                {activeSourceTab === 'database' && (
                    <div className="space-y-6">
                        {!dbConnection?.connected ? (
                            <>
                                <h3 className="text-lg font-medium text-white">Connect to PostgreSQL</h3>
                                <div className="grid grid-cols-2 gap-4">
                                    <Input
                                        label="Host"
                                        value={dbForm.host}
                                        onChange={(v) => setDbForm(prev => ({ ...prev, host: v }))}
                                    />
                                    <Input
                                        label="Port"
                                        value={dbForm.port}
                                        onChange={(v) => setDbForm(prev => ({ ...prev, port: v }))}
                                    />
                                    <Input
                                        label="Database"
                                        value={dbForm.database}
                                        onChange={(v) => setDbForm(prev => ({ ...prev, database: v }))}
                                    />
                                    <Input
                                        label="Username"
                                        value={dbForm.user}
                                        onChange={(v) => setDbForm(prev => ({ ...prev, user: v }))}
                                    />
                                    <Input
                                        label="Password"
                                        type="password"
                                        value={dbForm.password}
                                        onChange={(v) => setDbForm(prev => ({ ...prev, password: v }))}
                                    />
                                </div>
                                <Button onClick={handleConnect} loading={connecting}>
                                    <Database className="w-4 h-4" />
                                    {connecting ? 'Connecting...' : 'Connect'}
                                </Button>
                                {connectionError && (
                                    <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400">
                                        ❌ {connectionError}
                                    </div>
                                )}
                            </>
                        ) : (
                            <>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="w-3 h-3 bg-green-500 rounded-full" />
                                        <span className="text-white font-medium">Connected to {dbConnection.database}</span>
                                    </div>
                                    <Button variant="ghost" onClick={() => setDbConnection(null)}>
                                        Disconnect
                                    </Button>
                                </div>
                                <Select
                                    label="Schema"
                                    value={dbForm.schema}
                                    onChange={(v) => setDbForm(prev => ({ ...prev, schema: v }))}
                                    options={availableSchemas.map(s => ({ value: s, label: s }))}
                                />
                                <Select
                                    label="Select a table or view"
                                    value={selectedTable}
                                    onChange={setSelectedTable}
                                    options={tables.map(t => ({ value: t.name, label: `${t.name} (${t.type})` }))}
                                    placeholder="Choose..."
                                />
                                {selectedTable && (
                                    <Button onClick={handleLoadTableData} loading={loadingTable}>
                                        Load Table Data
                                    </Button>
                                )}
                            </>
                        )}
                    </div>
                )}

                {/* Saved Datasets Tab */}
                {activeSourceTab === 'saved' && (
                    <div className="space-y-6">
                        {loadingDatasets ? (
                            <div className="text-center py-12">
                                <RefreshCw className="w-8 h-8 mx-auto text-slate-500 animate-spin" />
                                <p className="text-slate-400 mt-4">Loading saved datasets...</p>
                            </div>
                        ) : savedDatasets.length > 0 ? (
                            <div className="grid gap-3">
                                {savedDatasets.map(ds => (
                                    <div
                                        key={ds.id}
                                        className="flex items-center justify-between p-4 bg-slate-800/30 rounded-xl border border-slate-700/50 hover:border-blue-500/50 transition-all cursor-pointer"
                                    >
                                        <div className="flex items-center gap-3">
                                            <FolderOpen className="w-5 h-5 text-blue-400" />
                                            <span className="text-white font-medium">{ds.name}</span>
                                        </div>
                                        <Button size="sm">Load</Button>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <EmptyState
                                icon={<FolderOpen />}
                                title="No Saved Datasets"
                                description="Prepare and save a dataset to see it here"
                            />
                        )}
                    </div>
                )}
            </Card>
        </div>
    )
}
