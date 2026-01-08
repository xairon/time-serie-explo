/**
 * Models Page - View and manage trained models
 */

import React, { useState, useEffect, useCallback } from 'react'
import {
    Brain,
    Trash2,
    Download,
    Calendar,
    Target,
    RefreshCw,
    ChevronRight,
    BarChart2,
} from 'lucide-react'
import { Card, Button, Badge, Metric, EmptyState, Select } from '../components/ui'
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

export function ModelsPage() {
    const { setSelectedModel, setActivePage } = useAppStore()

    // State
    const [models, setModels] = useState<StoreModelInfo[]>([])
    const [loading, setLoading] = useState(true)
    const [selectedModelId, setSelectedModelId] = useState<string | null>(null)
    const [filter, setFilter] = useState<{ model_type?: string; model_name?: string }>({})

    // Load models
    useEffect(() => {
        loadModels()
    }, [])

    const loadModels = async () => {
        setLoading(true)
        try {
            const result = await modelsApi.list(filter)
            setModels(result.models.map(toStoreModel))
        } catch (err) {
            console.error('Failed to load models:', err)
        } finally {
            setLoading(false)
        }
    }

    // Delete model
    const handleDelete = async (modelId: string) => {
        if (!confirm('Are you sure you want to delete this model?')) return

        try {
            await modelsApi.delete(modelId)
            setModels(prev => prev.filter(m => m.modelId !== modelId))
        } catch (err) {
            console.error('Failed to delete model:', err)
        }
    }

    // Use for forecasting
    const handleUseForecast = (model: StoreModelInfo) => {
        setSelectedModel(model)
        setActivePage('forecasting')
    }

    // Group models by station
    const modelsByStation = models.reduce((acc, model) => {
        const station = model.stations[0] || 'Unknown'
        if (!acc[station]) acc[station] = []
        acc[station].push(model)
        return acc
    }, {} as Record<string, StoreModelInfo[]>)

    const selectedModel = models.find(m => m.modelId === selectedModelId)

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="text-center">
                    <RefreshCw className="w-8 h-8 mx-auto text-blue-500 animate-spin mb-4" />
                    <p className="text-slate-400">Loading models...</p>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white">Trained Models</h1>
                    <p className="text-slate-400 mt-1">Manage and deploy your forecasting models</p>
                </div>
                <div className="flex gap-3">
                    <Button variant="secondary" onClick={loadModels}>
                        <RefreshCw className="w-4 h-4" />
                        Refresh
                    </Button>
                    <Button onClick={() => setActivePage('training')}>
                        Train New Model
                    </Button>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
                <Card className="p-4">
                    <Metric label="Total Models" value={models.length} />
                </Card>
                <Card className="p-4">
                    <Metric label="Single Models" value={models.filter(m => m.modelType === 'single').length} />
                </Card>
                <Card className="p-4">
                    <Metric label="Global Models" value={models.filter(m => m.modelType === 'global').length} />
                </Card>
                <Card className="p-4">
                    <Metric label="Stations Covered" value={Object.keys(modelsByStation).length} />
                </Card>
            </div>

            {models.length === 0 ? (
                <Card className="p-8">
                    <EmptyState
                        icon={<Brain />}
                        title="No Trained Models"
                        description="Train a model to see it here"
                        action={
                            <Button onClick={() => setActivePage('training')}>
                                Go to Training
                            </Button>
                        }
                    />
                </Card>
            ) : (
                <div className="grid grid-cols-3 gap-6">
                    {/* Model List */}
                    <Card className="col-span-2 p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-medium text-white">All Models</h3>
                            <div className="flex gap-2">
                                <Select
                                    value={filter.model_type || ''}
                                    onChange={(v) => setFilter(prev => ({ ...prev, model_type: v || undefined }))}
                                    options={[
                                        { value: '', label: 'All Types' },
                                        { value: 'single', label: 'Single' },
                                        { value: 'global', label: 'Global' },
                                    ]}
                                />
                            </div>
                        </div>

                        <div className="space-y-2">
                            {models.map(model => (
                                <div
                                    key={model.modelId}
                                    onClick={() => setSelectedModelId(model.modelId)}
                                    className={`flex items-center justify-between p-4 rounded-xl border cursor-pointer transition-all ${selectedModelId === model.modelId
                                        ? 'bg-blue-500/10 border-blue-500/50'
                                        : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600'
                                        }`}
                                >
                                    <div className="flex items-center gap-4">
                                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                                            <Brain className="w-5 h-5 text-white" />
                                        </div>
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className="font-medium text-white">{model.modelName}</span>
                                                <Badge variant={model.modelType === 'global' ? 'info' : 'default'}>
                                                    {model.modelType}
                                                </Badge>
                                            </div>
                                            <div className="flex items-center gap-3 mt-1 text-sm text-slate-400">
                                                <span className="flex items-center gap-1">
                                                    <Target className="w-3 h-3" />
                                                    {model.stations.length} station{model.stations.length > 1 ? 's' : ''}
                                                </span>
                                                <span className="flex items-center gap-1">
                                                    <Calendar className="w-3 h-3" />
                                                    {new Date(model.createdAt).toLocaleDateString()}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        {model.metrics?.R2 && (
                                            <span className="text-sm text-green-400">R²: {model.metrics.R2.toFixed(2)}</span>
                                        )}
                                        <ChevronRight className="w-5 h-5 text-slate-500" />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>

                    {/* Model Details */}
                    <Card className="p-6">
                        {selectedModel ? (
                            <div className="space-y-6">
                                <div>
                                    <h3 className="text-lg font-medium text-white mb-1">{selectedModel.modelName}</h3>
                                    <p className="text-sm text-slate-400">ID: {selectedModel.modelId}</p>
                                </div>

                                <div className="space-y-3">
                                    <h4 className="text-sm font-medium text-slate-400">Configuration</h4>
                                    <div className="grid grid-cols-2 gap-2 text-sm">
                                        <div className="bg-slate-800/50 rounded-lg p-2">
                                            <span className="text-slate-400">Input</span>
                                            <p className="text-white font-medium">{selectedModel.input_chunk_length}d</p>
                                        </div>
                                        <div className="bg-slate-800/50 rounded-lg p-2">
                                            <span className="text-slate-400">Horizon</span>
                                            <p className="text-white font-medium">{selectedModel.output_chunk_length}d</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="space-y-3">
                                    <h4 className="text-sm font-medium text-slate-400">Stations</h4>
                                    <div className="flex flex-wrap gap-2">
                                        {selectedModel.stations.map(s => (
                                            <Badge key={s} variant="info">{s}</Badge>
                                        ))}
                                    </div>
                                </div>

                                {selectedModel.metrics && (
                                    <div className="space-y-3">
                                        <h4 className="text-sm font-medium text-slate-400">Metrics</h4>
                                        <div className="grid grid-cols-2 gap-2">
                                            {Object.entries(selectedModel.metrics).map(([key, value]) => (
                                                <div key={key} className="bg-slate-800/50 rounded-lg p-2 text-sm">
                                                    <span className="text-slate-400">{key}</span>
                                                    <p className="text-white font-medium">{typeof value === 'number' ? value.toFixed(4) : value}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                <div className="pt-4 border-t border-slate-700/50 space-y-2">
                                    <Button onClick={() => handleUseForecast(selectedModel)} className="w-full">
                                        <BarChart2 className="w-4 h-4" />
                                        Use for Forecasting
                                    </Button>
                                    <Button variant="danger" onClick={() => handleDelete(selectedModel.modelId)} className="w-full">
                                        <Trash2 className="w-4 h-4" />
                                        Delete Model
                                    </Button>
                                </div>
                            </div>
                        ) : (
                            <div className="text-center py-12 text-slate-400">
                                <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                <p>Select a model to view details</p>
                            </div>
                        )}
                    </Card>
                </div>
            )}
        </div>
    )
}
