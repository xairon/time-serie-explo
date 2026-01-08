/**
 * Training Page - Model training with real-time progress
 * Replicates functionality from 2_Train_Models.py
 */

import React, { useState, useCallback, useEffect } from 'react'
import Plot from 'react-plotly.js'
import {
    Play,
    Pause,
    Square,
    Settings,
    Zap,
    Clock,
    TrendingUp,
    CheckCircle,
    XCircle,
    Loader2,
} from 'lucide-react'
import { Card, Button, Input, Select, Tabs, Badge, Metric, Progress, EmptyState } from '../components/ui'
import { useAppStore } from '../store/appStore'
import { trainingApi, connectTrainingWebSocket, TrainingProgressEvent } from '../api/client'

// Model configurations
const MODELS = [
    { id: 'TFT', name: 'Temporal Fusion Transformer', description: 'State-of-the-art for complex time series', recommended: true },
    { id: 'NHiTS', name: 'N-HiTS', description: 'Fast and accurate long-horizon forecasting', recommended: true },
    { id: 'TiDE', name: 'TiDE', description: 'Time-series Dense Encoder (Google)', recommended: true },
    { id: 'DLinear', name: 'DLinear', description: 'Simple linear model (often beats Transformers)', recommended: false },
    { id: 'NLinear', name: 'NLinear', description: 'Normalized linear model', recommended: false },
    { id: 'NBEATS', name: 'N-BEATS', description: 'Interpretable neural network', recommended: false },
    { id: 'RNN', name: 'LSTM/GRU', description: 'Classic recurrent neural network', recommended: false },
    { id: 'BlockRNN', name: 'Block RNN', description: 'Fixed-length encoding RNN', recommended: false },
    { id: 'TCN', name: 'TCN', description: 'Temporal Convolutional Network', recommended: false },
    { id: 'Transformer', name: 'Transformer', description: 'Attention-based architecture', recommended: false },
    { id: 'XGBoost', name: 'XGBoost', description: 'Gradient Boosting (Global)', recommended: false },
    { id: 'LightGBM', name: 'LightGBM', description: 'Light Gradient Boosting', recommended: false },
]

export function TrainingPage() {
    const { currentDataset, activeJobs, addJob, updateJob, removeJob } = useAppStore()

    // Training configuration
    const [config, setConfig] = useState({
        modelName: 'TFT',
        inputChunkLength: 30,
        outputChunkLength: 7,
        trainRatio: 70,
        valRatio: 15,
        epochs: 100,
        batchSize: 32,
        learningRate: 0.001,
        useCovariates: true,
        earlyStoppingPatience: 5,
        optimizeHyperparameters: false,
    })

    // UI state
    const [activeTab, setActiveTab] = useState<'config' | 'progress' | 'history'>('config')
    const [starting, setStarting] = useState(false)
    const [selectedJob, setSelectedJob] = useState<string | null>(null)

    // Live metrics from WebSocket
    const [liveMetrics, setLiveMetrics] = useState<{
        epochs: number[]
        trainLoss: number[]
        valLoss: number[]
    }>({ epochs: [], trainLoss: [], valLoss: [] })

    // WebSocket connection for live updates
    useEffect(() => {
        if (!selectedJob) return

        const ws = connectTrainingWebSocket(
            selectedJob,
            (event: TrainingProgressEvent) => {
                if (event.event_type === 'progress' || event.event_type === 'metric') {
                    updateJob(selectedJob, {
                        progress: event.progress || 0,
                        currentEpoch: event.epoch,
                        trainLoss: event.train_loss,
                        valLoss: event.val_loss,
                    })

                    if (event.epoch && event.train_loss) {
                        setLiveMetrics(prev => ({
                            epochs: [...prev.epochs, event.epoch!],
                            trainLoss: [...prev.trainLoss, event.train_loss!],
                            valLoss: event.val_loss ? [...prev.valLoss, event.val_loss] : prev.valLoss,
                        }))
                    }
                } else if (event.event_type === 'completed') {
                    updateJob(selectedJob, {
                        status: 'completed',
                        progress: 1,
                        modelId: event.model_id,
                    })
                } else if (event.event_type === 'error') {
                    updateJob(selectedJob, {
                        status: 'failed',
                        error: event.error,
                    })
                }
            },
            (error) => console.error('WebSocket error:', error),
            () => console.log('WebSocket closed')
        )

        return () => ws.close()
    }, [selectedJob, updateJob])

    // Start training
    const handleStartTraining = useCallback(async () => {
        if (!currentDataset) return

        setStarting(true)
        try {
            const result = await trainingApi.start({
                dataset_id: currentDataset.id,
                stations: currentDataset.stations || ['default'],
                training_strategy: 'independent',
                config: {
                    model_name: config.modelName,
                    train_ratio: config.trainRatio / 100,
                    val_ratio: config.valRatio / 100,
                    input_chunk_length: config.inputChunkLength,
                    output_chunk_length: config.outputChunkLength,
                    epochs: config.epochs,
                    batch_size: config.batchSize,
                    learning_rate: config.learningRate,
                    early_stopping_patience: config.earlyStoppingPatience,
                    optimize_hyperparameters: config.optimizeHyperparameters,
                },
                use_covariates: config.useCovariates,
                save_model: true,
            })

            addJob({
                jobId: result.job_id,
                status: 'pending',
                progress: 0,
                totalEpochs: config.epochs,
            })

            setSelectedJob(result.job_id)
            setActiveTab('progress')
            setLiveMetrics({ epochs: [], trainLoss: [], valLoss: [] })

        } catch (err) {
            console.error('Failed to start training:', err)
        } finally {
            setStarting(false)
        }
    }, [currentDataset, config, addJob])

    // Cancel training
    const handleCancelTraining = useCallback(async (jobId: string) => {
        try {
            await trainingApi.cancel(jobId)
            updateJob(jobId, { status: 'cancelled' })
        } catch (err) {
            console.error('Failed to cancel:', err)
        }
    }, [updateJob])

    // Get current job info
    const currentJob = selectedJob ? activeJobs.find(j => j.jobId === selectedJob) : null

    if (!currentDataset) {
        return (
            <div className="space-y-6">
                <div>
                    <h1 className="text-3xl font-bold text-white">Train Models</h1>
                    <p className="text-slate-400 mt-1">Configure and train forecasting models</p>
                </div>
                <Card className="p-8">
                    <EmptyState
                        icon={<Settings />}
                        title="No Dataset Configured"
                        description="Please prepare a dataset first before training a model"
                        action={
                            <Button onClick={() => useAppStore.getState().setActivePage('datasets')}>
                                Go to Datasets
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
                    <h1 className="text-3xl font-bold text-white">Train Models</h1>
                    <p className="text-slate-400 mt-1">
                        Dataset: <span className="text-blue-400">{currentDataset.name}</span> •
                        Target: <span className="text-cyan-400">{currentDataset.targetColumn}</span>
                    </p>
                </div>
                {activeJobs.filter(j => j.status === 'running').length > 0 && (
                    <Badge variant="success">
                        <Loader2 className="w-3 h-3 animate-spin mr-1" />
                        Training in Progress
                    </Badge>
                )}
            </div>

            {/* Tabs */}
            <Tabs
                tabs={[
                    { id: 'config', label: 'Configuration', icon: <Settings className="w-4 h-4" /> },
                    { id: 'progress', label: 'Progress', icon: <TrendingUp className="w-4 h-4" /> },
                    { id: 'history', label: 'History', icon: <Clock className="w-4 h-4" /> },
                ]}
                activeTab={activeTab}
                onChange={(id) => setActiveTab(id as 'config' | 'progress' | 'history')}
            />

            {/* Configuration Tab */}
            {activeTab === 'config' && (
                <div className="grid grid-cols-3 gap-6">
                    {/* Model Selection */}
                    <Card className="p-6 col-span-2">
                        <h3 className="text-lg font-medium text-white mb-4">Select Model Architecture</h3>
                        <div className="grid grid-cols-2 gap-3 max-h-[400px] overflow-y-auto pr-2">
                            {MODELS.map(model => (
                                <button
                                    key={model.id}
                                    onClick={() => setConfig(prev => ({ ...prev, modelName: model.id }))}
                                    className={`p-4 rounded-xl border text-left transition-all ${config.modelName === model.id
                                        ? 'bg-blue-500/10 border-blue-500/50 ring-2 ring-blue-500/30'
                                        : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600'
                                        }`}
                                >
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="font-medium text-white">{model.name}</span>
                                        {model.recommended && <Badge variant="success">⭐</Badge>}
                                    </div>
                                    <p className="text-sm text-slate-400">{model.description}</p>
                                </button>
                            ))}
                        </div>
                    </Card>

                    {/* Quick Stats */}
                    <Card className="p-6">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-lg font-medium text-white">Dataset Summary</h3>
                            <Badge variant="info">Ready</Badge>
                        </div>
                        <div className="space-y-4">
                            <Metric label="Target" value={currentDataset.targetColumn} />
                            <Metric label="Covariates" value={currentDataset.covariateColumns.length} />
                            <Metric label="Normalization" value={currentDataset.preprocessing.normalization} />
                            <Metric label="Stations" value={currentDataset.stations?.length || 1} />
                        </div>
                    </Card>

                    {/* Hyperparameters */}
                    <Card className="p-6 col-span-2">
                        <h3 className="text-lg font-medium text-white mb-4">Hyperparameters</h3>
                        <div className="grid grid-cols-4 gap-4">
                            <Input
                                label="Input Window"
                                type="number"
                                value={config.inputChunkLength}
                                onChange={(v) => setConfig(prev => ({ ...prev, inputChunkLength: parseInt(v) || 30 }))}
                            />
                            <Input
                                label="Forecast Horizon"
                                type="number"
                                value={config.outputChunkLength}
                                onChange={(v) => setConfig(prev => ({ ...prev, outputChunkLength: parseInt(v) || 7 }))}
                            />
                            <Input
                                label="Epochs"
                                type="number"
                                value={config.epochs}
                                onChange={(v) => setConfig(prev => ({ ...prev, epochs: parseInt(v) || 100 }))}
                            />
                            <Input
                                label="Batch Size"
                                type="number"
                                value={config.batchSize}
                                onChange={(v) => setConfig(prev => ({ ...prev, batchSize: parseInt(v) || 32 }))}
                            />
                        </div>

                        <div className="grid grid-cols-2 gap-8 mt-6">
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-1.5">
                                        Train / Val / Test Split
                                    </label>
                                    <div className="flex items-center gap-2">
                                        <div className="relative flex-1">
                                            <span className="absolute left-2 top-2 text-xs text-slate-500">Train</span>
                                            <Input
                                                type="number"
                                                value={config.trainRatio}
                                                onChange={(v) => setConfig(prev => ({ ...prev, trainRatio: parseInt(v) || 70 }))}
                                                className="pt-5"
                                            />
                                        </div>
                                        <div className="relative flex-1">
                                            <span className="absolute left-2 top-2 text-xs text-slate-500">Val</span>
                                            <Input
                                                type="number"
                                                value={config.valRatio}
                                                onChange={(v) => setConfig(prev => ({ ...prev, valRatio: parseInt(v) || 15 }))}
                                                className="pt-5"
                                            />
                                        </div>
                                        <div className="flex-1 text-center py-2 bg-slate-800/50 rounded border border-slate-700">
                                            <div className="text-xs text-slate-500 uppercase">Test</div>
                                            <div className="text-slate-300 font-mono">
                                                {100 - config.trainRatio - config.valRatio}%
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <Input
                                    label="Learning Rate"
                                    type="number"
                                    value={config.learningRate}
                                    onChange={(v) => setConfig(prev => ({ ...prev, learningRate: parseFloat(v) || 0.001 }))}
                                />
                            </div>

                            <div className="space-y-4">
                                <label className="block text-sm font-medium text-slate-300">Advanced Options</label>

                                <div className="space-y-3 p-4 bg-slate-800/30 rounded-xl">
                                    <label className="flex items-center justify-between cursor-pointer">
                                        <span className="text-slate-300">Use Covariates</span>
                                        <input
                                            type="checkbox"
                                            checked={config.useCovariates}
                                            onChange={(e) => setConfig(prev => ({ ...prev, useCovariates: e.target.checked }))}
                                            className="w-4 h-4 rounded bg-slate-800 border-slate-600"
                                        />
                                    </label>

                                    <div className="flex items-center justify-between">
                                        <span className="text-slate-300 pointer-events-none opacity-50">Auto-Optimization (Optuna)</span>
                                        <input
                                            type="checkbox"
                                            checked={config.optimizeHyperparameters}
                                            disabled={true}
                                            onChange={(e) => setConfig(prev => ({ ...prev, optimizeHyperparameters: e.target.checked }))}
                                            className="w-4 h-4 rounded bg-slate-800 border-slate-600 opacity-50 cursor-not-allowed"
                                        />
                                    </div>
                                    <p className="text-xs text-slate-500 mt-0">Coming soon</p>

                                    <div className="pt-2 border-t border-slate-700/50">
                                        <div className="flex justify-between items-center mb-2">
                                            <span className="text-slate-300">Early Stopping Patience</span>
                                            <span className="text-xs text-slate-500">0 to disable</span>
                                        </div>
                                        <Input
                                            type="number"
                                            value={config.earlyStoppingPatience}
                                            onChange={(v) => setConfig(prev => ({ ...prev, earlyStoppingPatience: parseInt(v) || 0 }))}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </Card>

                    {/* Training Actions */}
                    <Card className="p-6">
                        <h3 className="text-lg font-medium text-white mb-4">Start Training</h3>
                        <div className="space-y-4">
                            <div className="text-sm text-slate-400 space-y-2">
                                <div className="flex justify-between">
                                    <span>Model:</span>
                                    <span className="text-white font-medium">{config.modelName}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Epochs:</span>
                                    <span className="text-white font-medium">{config.epochs}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Device:</span>
                                    <span className="text-green-400 font-medium flex items-center gap-1">
                                        <Zap className="w-3 h-3" /> Auto (GPU/CPU)
                                    </span>
                                </div>
                            </div>
                            <Button onClick={handleStartTraining} loading={starting} className="w-full">
                                <Play className="w-4 h-4" />
                                Start Training
                            </Button>
                        </div>
                    </Card>
                </div>
            )}

            {/* Progress Tab */}
            {activeTab === 'progress' && (
                <div className="space-y-6">
                    {currentJob ? (
                        <>
                            {/* Status bar */}
                            <Card className="p-6">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        {currentJob.status === 'running' && <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />}
                                        {currentJob.status === 'completed' && <CheckCircle className="w-5 h-5 text-green-400" />}
                                        {currentJob.status === 'failed' && <XCircle className="w-5 h-5 text-red-400" />}
                                        <span className="text-white font-medium">
                                            {currentJob.status === 'running' && `Training Epoch ${currentJob.currentEpoch || 0}/${currentJob.totalEpochs}`}
                                            {currentJob.status === 'completed' && 'Training Complete!'}
                                            {currentJob.status === 'failed' && 'Training Failed'}
                                            {currentJob.status === 'pending' && 'Preparing...'}
                                        </span>
                                    </div>
                                    {currentJob.status === 'running' && (
                                        <Button variant="danger" size="sm" onClick={() => handleCancelTraining(currentJob.jobId)}>
                                            <Square className="w-4 h-4" />
                                            Cancel
                                        </Button>
                                    )}
                                </div>
                                <Progress value={currentJob.progress * 100} label="Progress" />
                            </Card>

                            {/* Metrics */}
                            <div className="grid grid-cols-4 gap-4">
                                <Card className="p-4">
                                    <Metric label="Current Epoch" value={currentJob.currentEpoch || '-'} />
                                </Card>
                                <Card className="p-4">
                                    <Metric label="Train Loss" value={currentJob.trainLoss?.toFixed(6) || '-'} />
                                </Card>
                                <Card className="p-4">
                                    <Metric label="Val Loss" value={currentJob.valLoss?.toFixed(6) || '-'} />
                                </Card>
                                <Card className="p-4">
                                    <Metric label="Progress" value={`${(currentJob.progress * 100).toFixed(0)}%`} />
                                </Card>
                            </div>

                            {/* Loss chart */}
                            <Card className="p-6">
                                <h3 className="text-lg font-medium text-white mb-4">Training Curves</h3>
                                {liveMetrics.epochs.length > 0 ? (
                                    <Plot
                                        data={[
                                            {
                                                x: liveMetrics.epochs,
                                                y: liveMetrics.trainLoss,
                                                type: 'scatter',
                                                mode: 'lines',
                                                name: 'Train Loss',
                                                line: { color: '#3b82f6', width: 2 },
                                            },
                                            {
                                                x: liveMetrics.epochs,
                                                y: liveMetrics.valLoss,
                                                type: 'scatter',
                                                mode: 'lines',
                                                name: 'Val Loss',
                                                line: { color: '#22c55e', width: 2 },
                                            },
                                        ]}
                                        layout={{
                                            height: 300,
                                            margin: { t: 20, r: 20, b: 50, l: 60 },
                                            paper_bgcolor: 'transparent',
                                            plot_bgcolor: 'transparent',
                                            xaxis: { title: { text: 'Epoch' }, color: '#94a3b8', gridcolor: '#334155' },
                                            yaxis: { title: { text: 'Loss' }, color: '#94a3b8', gridcolor: '#334155' },
                                            legend: { orientation: 'h', y: 1.1, font: { color: '#94a3b8' } },
                                        }}
                                        config={{ displayModeBar: false }}
                                        style={{ width: '100%' }}
                                    />
                                ) : (
                                    <div className="h-64 flex items-center justify-center text-slate-500">
                                        Waiting for training data...
                                    </div>
                                )}
                            </Card>

                            {/* Completion actions */}
                            {currentJob.status === 'completed' && (
                                <Card className="p-6 bg-green-500/10 border-green-500/30">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <CheckCircle className="w-8 h-8 text-green-400" />
                                            <div>
                                                <h3 className="text-lg font-medium text-white">Training Complete!</h3>
                                                <p className="text-sm text-slate-400">Model ID: {currentJob.modelId}</p>
                                            </div>
                                        </div>
                                        <div className="flex gap-3">
                                            <Button variant="secondary" onClick={() => setActiveTab('config')}>
                                                Train Another
                                            </Button>
                                            <Button onClick={() => useAppStore.getState().setActivePage('forecasting')}>
                                                <Zap className="w-4 h-4" />
                                                Make Predictions
                                            </Button>
                                        </div>
                                    </div>
                                </Card>
                            )}
                        </>
                    ) : (
                        <EmptyState
                            icon={<TrendingUp />}
                            title="No Active Training"
                            description="Start a training job to see real-time progress"
                            action={
                                <Button onClick={() => setActiveTab('config')}>
                                    Configure Training
                                </Button>
                            }
                        />
                    )}
                </div>
            )}

            {/* History Tab */}
            {activeTab === 'history' && (
                <Card className="p-6">
                    <h3 className="text-lg font-medium text-white mb-4">Training History</h3>
                    {activeJobs.length > 0 ? (
                        <div className="space-y-3">
                            {activeJobs.map(job => (
                                <div
                                    key={job.jobId}
                                    className={`flex items-center justify-between p-4 rounded-xl border cursor-pointer transition-all ${selectedJob === job.jobId
                                        ? 'bg-blue-500/10 border-blue-500/50'
                                        : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600'
                                        }`}
                                    onClick={() => {
                                        setSelectedJob(job.jobId)
                                        setActiveTab('progress')
                                    }}
                                >
                                    <div className="flex items-center gap-3">
                                        {job.status === 'running' && <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />}
                                        {job.status === 'completed' && <CheckCircle className="w-5 h-5 text-green-400" />}
                                        {job.status === 'failed' && <XCircle className="w-5 h-5 text-red-400" />}
                                        {job.status === 'cancelled' && <Square className="w-5 h-5 text-slate-400" />}
                                        {job.status === 'pending' && <Clock className="w-5 h-5 text-yellow-400" />}
                                        <div>
                                            <span className="text-white font-medium">{job.jobId.slice(0, 8)}...</span>
                                            <div className="flex gap-2 mt-1">
                                                <Badge variant={
                                                    job.status === 'completed' ? 'success' :
                                                        job.status === 'running' ? 'info' :
                                                            job.status === 'failed' ? 'error' : 'default'
                                                }>
                                                    {job.status}
                                                </Badge>
                                                {job.modelId && <Badge variant="info">{job.modelId.slice(0, 12)}</Badge>}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <span className="text-slate-400 text-sm">
                                            {job.currentEpoch || 0}/{job.totalEpochs || '-'} epochs
                                        </span>
                                        <Progress value={job.progress * 100} showValue={false} size="sm" />
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <EmptyState
                            icon={<Clock />}
                            title="No Training History"
                            description="Your completed and running training jobs will appear here"
                        />
                    )}
                </Card>
            )}
        </div>
    )
}
