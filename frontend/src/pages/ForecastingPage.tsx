/**
 * Forecasting Page - Generate and visualize predictions
 * Replicates functionality from 3_Forecasting.py
 */

import React, { useState, useCallback, useEffect } from 'react'
import Plot from 'react-plotly.js'
import {
    LineChart,
    Calendar,
    Sliders,
    Download,
    Info,
    Brain,
    Target,
    TrendingUp,
    BarChart2,
} from 'lucide-react'
import { Card, Button, Select, Tabs, Badge, Metric, Progress, EmptyState } from '../components/ui'
import { useAppStore, ModelInfo as StoreModelInfo } from '../store/appStore'
import { modelsApi, ModelInfo as ApiModelInfo } from '../api/client'

// Convert API model to store model
function toStoreModel(m: ApiModelInfo): StoreModelInfo {
    return {
        modelId: m.model_id,
        modelName: m.model_name,
        modelType: m.model_type,
        stations: m.stations,
        metrics: m.metrics,
        createdAt: m.created_at,
        input_chunk_length: m.input_chunk_length,
        output_chunk_length: m.output_chunk_length,
    }
}

export function ForecastingPage() {
    const { selectedModel, setSelectedModel } = useAppStore()

    // State
    const [models, setModels] = useState<StoreModelInfo[]>([])
    const [loadingModels, setLoadingModels] = useState(false)
    const [activeTab, setActiveTab] = useState<'predictions' | 'explainability'>('predictions')

    // Forecast controls
    const [windowPosition, setWindowPosition] = useState(0)
    const [forecastHorizon, setForecastHorizon] = useState(7)
    const [generating, setGenerating] = useState(false)

    // Forecast results
    const [predictions, setPredictions] = useState<{
        dates: string[]
        actual: number[]
        predicted: number[]
        metrics: Record<string, number>
    } | null>(null)

    // Load models on mount
    useEffect(() => {
        loadModels()
    }, [])

    const loadModels = async () => {
        setLoadingModels(true)
        try {
            const result = await modelsApi.list()
            const storeModels = result.models.map(toStoreModel)
            setModels(storeModels)
            if (storeModels.length > 0 && !selectedModel) {
                setSelectedModel(storeModels[0])
            }
        } catch (err) {
            console.error('Failed to load models:', err)
        } finally {
            setLoadingModels(false)
        }
    }

    // Generate forecast (mock for now - would call API)
    const generateForecast = useCallback(async () => {
        if (!selectedModel) return

        setGenerating(true)

        // Simulate forecast generation (in real app, this would call the API)
        await new Promise(resolve => setTimeout(resolve, 1500))

        // Generate mock data
        const startDate = new Date('2024-01-01')
        const dates: string[] = []
        const actual: number[] = []
        const predicted: number[] = []

        for (let i = 0; i < 60; i++) {
            const date = new Date(startDate)
            date.setDate(date.getDate() + i)
            dates.push(date.toISOString().split('T')[0])

            // Mock actual values
            const value = 10 + Math.sin(i / 10) * 2 + Math.random() * 0.5
            actual.push(value)

            // Mock predictions (slightly offset)
            predicted.push(value + (Math.random() - 0.5) * 0.8)
        }

        setPredictions({
            dates,
            actual,
            predicted,
            metrics: {
                MAE: 0.32,
                RMSE: 0.45,
                MAPE: 3.21,
                R2: 0.92,
            },
        })

        setGenerating(false)
    }, [selectedModel])

    // Export predictions
    const exportPredictions = () => {
        if (!predictions) return

        const csv = ['date,actual,predicted']
        predictions.dates.forEach((date, i) => {
            csv.push(`${date},${predictions.actual[i]},${predictions.predicted[i]}`)
        })

        const blob = new Blob([csv.join('\n')], { type: 'text/csv' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `forecast_${selectedModel?.modelId || 'model'}.csv`
        a.click()
        URL.revokeObjectURL(url)
    }

    if (loadingModels) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="text-center">
                    <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-slate-400">Loading models...</p>
                </div>
            </div>
        )
    }

    if (models.length === 0) {
        return (
            <div className="space-y-6">
                <div>
                    <h1 className="text-3xl font-bold text-white">Forecasting</h1>
                    <p className="text-slate-400 mt-1">Generate predictions using trained models</p>
                </div>
                <Card className="p-8">
                    <EmptyState
                        icon={<LineChart />}
                        title="No Trained Models"
                        description="Train a model first to generate forecasts"
                        action={
                            <Button onClick={() => useAppStore.getState().setActivePage('training')}>
                                Go to Training
                            </Button>
                        }
                    />
                </Card>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white">Forecasting</h1>
                    <p className="text-slate-400 mt-1">Generate and analyze predictions</p>
                </div>
                {predictions && (
                    <Button variant="secondary" onClick={exportPredictions}>
                        <Download className="w-4 h-4" />
                        Export CSV
                    </Button>
                )}
            </div>

            {/* Model Selection Sidebar */}
            <div className="grid grid-cols-4 gap-6">
                <Card className="p-6 space-y-6">
                    {/* Model Selection */}
                    <div>
                        <h3 className="text-sm font-medium text-slate-400 mb-3 flex items-center gap-2">
                            <Brain className="w-4 h-4" />
                            Model
                        </h3>
                        <Select
                            value={selectedModel?.modelId || ''}
                            onChange={(id) => setSelectedModel(models.find(m => m.modelId === id) || null)}
                            options={models.map(m => ({
                                value: m.modelId,
                                label: `${m.modelName} - ${m.stations[0] || 'all'}`,
                            }))}
                        />
                    </div>

                    {/* Model Info */}
                    {selectedModel && (
                        <div className="space-y-3 pt-3 border-t border-slate-700/50">
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Type</span>
                                <Badge variant={selectedModel.modelType === 'global' ? 'info' : 'default'}>
                                    {selectedModel.modelType}
                                </Badge>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Input Window</span>
                                <span className="text-white">{selectedModel.input_chunk_length}d</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-slate-400">Horizon</span>
                                <span className="text-white">{selectedModel.output_chunk_length}d</span>
                            </div>
                            {selectedModel.metrics && (
                                <>
                                    <div className="pt-2 border-t border-slate-700/50">
                                        <h4 className="text-xs text-slate-500 mb-2">Validation Metrics</h4>
                                        {Object.entries(selectedModel.metrics).slice(0, 3).map(([key, value]) => (
                                            <div key={key} className="flex justify-between text-sm">
                                                <span className="text-slate-400">{key}</span>
                                                <span className="text-white">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                                            </div>
                                        ))}
                                    </div>
                                </>
                            )}
                        </div>
                    )}

                    {/* Actions */}
                    <Button onClick={generateForecast} loading={generating} className="w-full">
                        <TrendingUp className="w-4 h-4" />
                        Generate Forecast
                    </Button>
                </Card>

                {/* Main Content */}
                <div className="col-span-3 space-y-6">
                    {/* Tabs */}
                    <Tabs
                        tabs={[
                            { id: 'predictions', label: 'Predictions', icon: <LineChart className="w-4 h-4" /> },
                            { id: 'explainability', label: 'Explainability', icon: <BarChart2 className="w-4 h-4" /> },
                        ]}
                        activeTab={activeTab}
                        onChange={(id) => setActiveTab(id as 'predictions' | 'explainability')}
                    />

                    {/* Predictions Tab */}
                    {activeTab === 'predictions' && (
                        predictions ? (
                            <div className="space-y-6">
                                {/* Window Slider */}
                                <Card className="p-4">
                                    <div className="flex items-center gap-4">
                                        <Sliders className="w-5 h-5 text-slate-400" />
                                        <span className="text-sm text-slate-400">Analysis Window</span>
                                        <input
                                            type="range"
                                            min={0}
                                            max={predictions.dates.length - forecastHorizon}
                                            value={windowPosition}
                                            onChange={(e) => setWindowPosition(parseInt(e.target.value))}
                                            className="flex-1"
                                        />
                                        <span className="text-sm text-white w-40 text-right">
                                            {predictions.dates[windowPosition]} → {predictions.dates[Math.min(windowPosition + forecastHorizon - 1, predictions.dates.length - 1)]}
                                        </span>
                                    </div>
                                </Card>

                                {/* Main Chart */}
                                <Card className="p-6">
                                    <Plot
                                        data={[
                                            {
                                                x: predictions.dates,
                                                y: predictions.actual,
                                                type: 'scatter',
                                                mode: 'lines',
                                                name: 'Actual',
                                                line: { color: '#3b82f6', width: 2 },
                                            },
                                            {
                                                x: predictions.dates,
                                                y: predictions.predicted,
                                                type: 'scatter',
                                                mode: 'lines',
                                                name: 'Predicted',
                                                line: { color: '#22c55e', width: 2, dash: 'dot' },
                                            },
                                            // Highlight window
                                            {
                                                x: [predictions.dates[windowPosition], predictions.dates[windowPosition]],
                                                y: [Math.min(...predictions.actual), Math.max(...predictions.actual)],
                                                type: 'scatter',
                                                mode: 'lines',
                                                name: 'Window Start',
                                                line: { color: '#f59e0b', width: 2, dash: 'dash' },
                                                showlegend: false,
                                            },
                                        ]}
                                        layout={{
                                            height: 400,
                                            margin: { t: 30, r: 30, b: 50, l: 60 },
                                            paper_bgcolor: 'transparent',
                                            plot_bgcolor: 'transparent',
                                            xaxis: { title: { text: 'Date' }, color: '#94a3b8', gridcolor: '#334155' },
                                            yaxis: { title: { text: 'Value' }, color: '#94a3b8', gridcolor: '#334155' },
                                            legend: { orientation: 'h', y: 1.12, font: { color: '#94a3b8' } },
                                            shapes: [{
                                                type: 'rect',
                                                xref: 'x',
                                                yref: 'paper',
                                                x0: predictions.dates[windowPosition],
                                                x1: predictions.dates[Math.min(windowPosition + forecastHorizon - 1, predictions.dates.length - 1)],
                                                y0: 0,
                                                y1: 1,
                                                fillcolor: '#f59e0b',
                                                opacity: 0.1,
                                                line: { width: 0 },
                                            }],
                                        }}
                                        config={{ displayModeBar: false }}
                                        style={{ width: '100%' }}
                                    />
                                </Card>

                                {/* Metrics */}
                                <div className="grid grid-cols-4 gap-4">
                                    {Object.entries(predictions.metrics).map(([key, value]) => (
                                        <Card key={key} className="p-4">
                                            <Metric label={key} value={value.toFixed(4)} />
                                        </Card>
                                    ))}
                                </div>
                            </div>
                        ) : (
                            <Card className="p-8">
                                <EmptyState
                                    icon={<Target />}
                                    title="No Predictions Yet"
                                    description="Select a model and click 'Generate Forecast' to create predictions"
                                />
                            </Card>
                        )
                    )}

                    {/* Explainability Tab */}
                    {activeTab === 'explainability' && (
                        <Card className="p-8">
                            <div className="text-center py-12">
                                <Brain className="w-16 h-16 mx-auto text-slate-500 mb-4" />
                                <h3 className="text-xl font-semibold text-white mb-2">Model Explainability</h3>
                                <p className="text-slate-400 max-w-md mx-auto mb-6">
                                    SHAP-based feature importance and temporal attribution analysis.
                                    Shows which features and time steps influenced predictions most.
                                </p>
                                <div className="grid grid-cols-2 gap-6 max-w-2xl mx-auto mt-8">
                                    <div className="bg-slate-800/30 rounded-xl p-6 text-left">
                                        <h4 className="font-medium text-white mb-2">Local Explanations</h4>
                                        <p className="text-sm text-slate-400 mb-4">
                                            Understand which features matter for a specific prediction window
                                        </p>
                                        <Button variant="secondary" size="sm" disabled>
                                            Coming Soon
                                        </Button>
                                    </div>
                                    <div className="bg-slate-800/30 rounded-xl p-6 text-left">
                                        <h4 className="font-medium text-white mb-2">Global Explanations</h4>
                                        <p className="text-sm text-slate-400 mb-4">
                                            Overall feature importance across all predictions
                                        </p>
                                        <Button variant="secondary" size="sm" disabled>
                                            Coming Soon
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        </Card>
                    )}
                </div>
            </div>
        </div>
    )
}
