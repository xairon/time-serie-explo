import type { ReactNode } from 'react'
import { useState } from 'react'
import { ChevronDown, ChevronRight, Play, Square, RotateCcw, AlertTriangle } from 'lucide-react'
import { useDatasets } from '@/hooks/useDatasets'
import { usePumpingDetection, usePumpingResults } from '@/hooks/usePumpingDetection'
import { AnnotatedChroniquePlot } from '@/components/pumping/AnnotatedChroniquePlot'
import { PastasPanel } from '@/components/pumping/PastasPanel'
import { XAIDriftPanel } from '@/components/pumping/XAIDriftPanel'
import { EmbeddingPanel } from '@/components/pumping/EmbeddingPanel'
import { VerdictPanel } from '@/components/pumping/VerdictPanel'

type DiagTab = 'pastas' | 'xai' | 'embeddings'

// --- Config panel ----------------------------------------------------------

interface AnalysisConfig {
  n_jobs: number
  window_days: number
  min_amplitude: number
  enable_xai: boolean
}

interface ConfigPanelProps {
  config: AnalysisConfig
  onChange: (c: AnalysisConfig) => void
}

function ConfigPanel({ config, onChange }: ConfigPanelProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
      <label className="flex flex-col gap-1">
        <span className="text-xs text-text-secondary">Fenêtre (jours)</span>
        <input
          type="number"
          min={7}
          max={365}
          value={config.window_days}
          onChange={e => onChange({ ...config, window_days: Number(e.target.value) })}
          className="bg-bg-primary border border-white/10 rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:border-accent-cyan/40"
        />
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-xs text-text-secondary">Amplitude min (m)</span>
        <input
          type="number"
          min={0.01}
          step={0.01}
          value={config.min_amplitude}
          onChange={e => onChange({ ...config, min_amplitude: Number(e.target.value) })}
          className="bg-bg-primary border border-white/10 rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:border-accent-cyan/40"
        />
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-xs text-text-secondary">Workers</span>
        <input
          type="number"
          min={1}
          max={16}
          value={config.n_jobs}
          onChange={e => onChange({ ...config, n_jobs: Number(e.target.value) })}
          className="bg-bg-primary border border-white/10 rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:border-accent-cyan/40"
        />
      </label>
      <label className="flex items-center gap-2 pt-4 cursor-pointer">
        <input
          type="checkbox"
          checked={config.enable_xai}
          onChange={e => onChange({ ...config, enable_xai: e.target.checked })}
          className="w-4 h-4 rounded border-white/20 accent-cyan-400"
        />
        <span className="text-sm text-text-secondary">Activer XAI (couche 2)</span>
      </label>
    </div>
  )
}

// --- Progress bar ----------------------------------------------------------

interface ProgressBarProps {
  stages: { stage: string; pct: number; message: string }[]
  currentStage: { stage: string; pct: number; message: string } | null
}

function ProgressBar({ stages, currentStage }: ProgressBarProps) {
  const pct = currentStage?.pct ?? (stages.length > 0 ? stages[stages.length - 1].pct : 0)
  const message = currentStage?.message ?? ''

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs text-text-secondary">
        <span>{message || 'Analyse en cours…'}</span>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-bg-primary rounded-full overflow-hidden">
        <div
          className="h-full bg-accent-cyan rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// --- Tab button ------------------------------------------------------------

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm rounded-lg transition-colors ${
        active
          ? 'bg-accent-cyan/10 text-accent-cyan'
          : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
      }`}
    >
      {children}
    </button>
  )
}

// --- Main page -------------------------------------------------------------

export default function PumpingDetectionPage() {
  const [datasetId, setDatasetId] = useState('')
  const [configOpen, setConfigOpen] = useState(false)
  const [diagTab, setDiagTab] = useState<DiagTab>('pastas')
  const [config, setConfig] = useState<AnalysisConfig>({
    n_jobs: 4,
    window_days: 30,
    min_amplitude: 0.05,
    enable_xai: true,
  })

  const { data: datasets, isLoading: datasetsLoading } = useDatasets()
  const detection = usePumpingDetection()
  const { data: results } = usePumpingResults(
    detection.status === 'done' ? (detection.taskId ?? null) : null,
  )

  const handleAnalyze = () => {
    if (!datasetId) return
    detection.analyze({
      dataset_id: datasetId,
      config: {
        n_jobs: config.n_jobs,
        window_days: config.window_days,
        min_amplitude: config.min_amplitude,
        enable_xai: config.enable_xai,
      },
    })
  }

  // Derive typed data from partial/final results
  const partialResults = detection.partialResults
  const finalResults = results as Record<string, unknown> | undefined

  const pastasData = (finalResults?.pastas ?? partialResults?.pastas) as Record<string, unknown> | undefined
  const xaiData = (finalResults?.xai ?? partialResults?.xai) as Record<string, unknown> | undefined
  const fusionData = (finalResults?.fusion ?? partialResults?.fusion) as Record<string, unknown> | undefined

  const chronique = (finalResults?.chronique ?? partialResults?.chronique) as
    | { time: string; value: number }[]
    | undefined

  const suspectWindows = (fusionData?.suspect_windows as
    | { start: string; end: string; confidence: number; label?: string }[]
    | undefined) ?? []

  const xaiIsPending = config.enable_xai && detection.isAnalyzing && !xaiData
  const xaiTyped = xaiData as {
    windows?: string[]
    features?: string[]
    attributions?: number[][]
    jsDivergences?: { window: string; js_div: number }[]
  } | undefined

  return (
    <div className="flex flex-col h-full overflow-y-auto bg-bg-primary">
      <div className="max-w-screen-xl mx-auto w-full px-4 py-6 space-y-5">

        {/* Header */}
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-xl font-bold text-text-primary">Détection de Pompage</h1>
            <p className="text-sm text-text-secondary mt-0.5">
              Analyse multi-couches des artefacts de pompage sur chronique piézométrique
            </p>
          </div>
        </div>

        {/* Dataset + config card */}
        <div className="bg-bg-card border border-white/5 rounded-xl p-4 space-y-4">
          <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-end">
            <div className="flex-1 flex flex-col gap-1">
              <label htmlFor="dataset-select" className="text-xs text-text-secondary">
                Jeu de données
              </label>
              <select
                id="dataset-select"
                value={datasetId}
                onChange={e => setDatasetId(e.target.value)}
                disabled={datasetsLoading || detection.isAnalyzing}
                className="bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-cyan/40 disabled:opacity-50"
              >
                <option value="">-- Sélectionner --</option>
                {(datasets ?? []).map(d => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex gap-2">
              {detection.isAnalyzing ? (
                <button
                  onClick={detection.cancel}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/15 border border-red-500/30 text-red-400 text-sm hover:bg-red-500/25 transition-colors"
                >
                  <Square className="w-4 h-4" />
                  Annuler
                </button>
              ) : (
                <>
                  {(detection.status === 'done' || detection.status === 'error' || detection.status === 'cancelled') && (
                    <button
                      onClick={detection.reset}
                      className="flex items-center gap-2 px-3 py-2 rounded-lg bg-bg-hover border border-white/5 text-text-secondary text-sm hover:text-text-primary transition-colors"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                  )}
                  <button
                    onClick={handleAnalyze}
                    disabled={!datasetId}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent-cyan/10 border border-accent-cyan/20 text-accent-cyan text-sm hover:bg-accent-cyan/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    <Play className="w-4 h-4" />
                    Analyser
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Config toggle */}
          <div>
            <button
              onClick={() => setConfigOpen(v => !v)}
              className="flex items-center gap-1.5 text-xs text-text-secondary hover:text-text-primary transition-colors"
            >
              {configOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
              Configuration avancée
            </button>
            {configOpen && (
              <ConfigPanel config={config} onChange={setConfig} />
            )}
          </div>

          {/* Progress */}
          {detection.isAnalyzing && (
            <ProgressBar stages={detection.stages} currentStage={detection.currentStage} />
          )}

          {/* Error */}
          {detection.status === 'error' && detection.error && (
            <div className="flex items-center gap-2 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              {detection.error}
            </div>
          )}
        </div>

        {/* Chronique plot */}
        {(chronique?.length || suspectWindows.length > 0) ? (
          <div className="bg-bg-card border border-white/5 rounded-xl p-4">
            <AnnotatedChroniquePlot
              data={chronique ?? []}
              suspectWindows={suspectWindows}
            />
          </div>
        ) : null}

        {/* Diagnostic panels */}
        {(detection.status !== 'idle' || Object.keys(partialResults).length > 0) && (
          <div className="bg-bg-card border border-white/5 rounded-xl overflow-hidden">
            <div className="flex border-b border-white/5 px-4 pt-3 gap-1">
              <TabButton active={diagTab === 'pastas'} onClick={() => setDiagTab('pastas')}>
                Pastas
              </TabButton>
              <TabButton active={diagTab === 'xai'} onClick={() => setDiagTab('xai')}>
                XAI Drift
              </TabButton>
              <TabButton active={diagTab === 'embeddings'} onClick={() => setDiagTab('embeddings')}>
                Embeddings
              </TabButton>
            </div>

            <div className="p-4">
              {diagTab === 'pastas' && (
                <PastasPanel
                  metrics={
                    pastasData
                      ? {
                          evp: pastasData.evp as number | undefined,
                          rmse: pastasData.rmse as number | undefined,
                          nse: pastasData.nse as number | undefined,
                        }
                      : undefined
                  }
                  residuals={(pastasData?.residuals as { time: string; value: number }[] | undefined)}
                  acf={(pastasData?.acf as number[] | undefined)}
                  pacf={(pastasData?.pacf as number[] | undefined)}
                  changepointDates={(pastasData?.changepoints as string[] | undefined)}
                />
              )}
              {diagTab === 'xai' && (
                <XAIDriftPanel
                  data={
                    xaiTyped?.windows && xaiTyped.features && xaiTyped.attributions
                      ? {
                          windows: xaiTyped.windows,
                          features: xaiTyped.features,
                          attributions: xaiTyped.attributions,
                          jsDivergences: xaiTyped.jsDivergences,
                        }
                      : undefined
                  }
                  isPending={xaiIsPending}
                />
              )}
              {diagTab === 'embeddings' && <EmbeddingPanel />}
            </div>
          </div>
        )}

        {/* Verdict panel */}
        {(detection.status === 'done' || fusionData) && (
          <div className="bg-bg-card border border-white/5 rounded-xl p-4 space-y-3">
            <h2 className="text-sm font-semibold text-text-primary">Verdict de fusion</h2>
            <VerdictPanel
              globalScore={fusionData?.global_score as number | undefined}
              suspectWindows={suspectWindows}
              layerConcordance={
                fusionData?.layer_concordance as
                  | { layer: string; score: number; n_windows: number }[]
                  | undefined
              }
              verdictLabel={fusionData?.verdict_label as string | undefined}
            />
          </div>
        )}
      </div>
    </div>
  )
}
