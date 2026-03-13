import { useState, useMemo, useCallback, useEffect } from 'react'
import { ChevronDown, ChevronRight, Download } from 'lucide-react'
import { ModelSelector } from '@/components/forecasting/ModelSelector'
import { ForecastView } from '@/components/forecasting/ForecastView'
import { TestSetOverview } from '@/components/charts/TestSetOverview'
import { ExplainabilityPanel } from '@/components/forecasting/ExplainabilityPanel'
import { useForecastSingle } from '@/hooks/useForecasting'
import { useModelDetail, useModelTestInfo } from '@/hooks/useModels'

/** Tooltip descriptions for each metric (matches Streamlit) */
const METRIC_TOOLTIPS: Record<string, string> = {
  MAE: 'Mean Absolute Error — average error in variable units',
  RMSE: 'Root Mean Squared Error — penalizes large errors more',
  sMAPE: 'Symmetric Mean Absolute Percentage Error (%)',
  WAPE: 'Weighted Absolute Percentage Error (%) — plus stable que MAPE',
  NRMSE: "Normalized RMSE — % de l'amplitude (max-min)",
  NSE: 'Nash-Sutcliffe Efficiency — 1=parfait, <0 pire que la moyenne',
  KGE: 'Kling-Gupta Efficiency — combine correlation, biais, variabilite',
  Dir_Acc: 'Directional Accuracy — % de directions correctes',
}

/** Display order for metrics (matches Streamlit) */
const METRIC_ORDER = ['MAE', 'RMSE', 'NRMSE', 'sMAPE', 'WAPE', 'NSE', 'KGE', 'Dir_Acc']

/** Color coding for metric values */
function metricColor(key: string, value: number): string {
  const lowerBetter = ['MAE', 'RMSE', 'sMAPE', 'WAPE', 'NRMSE']
  const higherBetter = ['NSE', 'KGE', 'Dir_Acc']
  if (lowerBetter.includes(key)) return value < 0.1 ? 'text-accent-green' : value > 1 ? 'text-accent-red' : 'text-text-primary'
  if (higherBetter.includes(key)) return value > 0.7 ? 'text-accent-green' : value < 0 ? 'text-accent-red' : 'text-text-primary'
  return 'text-text-primary'
}

/** Format metric value with suffix */
function formatMetric(key: string, value: number): string {
  const pctMetrics = ['sMAPE', 'WAPE', 'NRMSE', 'Dir_Acc']
  return `${value.toFixed(4)}${pctMetrics.includes(key) ? '%' : ''}`
}

export default function ForecastingPage() {
  const [modelId, setModelId] = useState('')
  const [sliderIdx, setSliderIdx] = useState<number | null>(null)
  const [hyperparamsOpen, setHyperparamsOpen] = useState(false)

  const { data: modelDetail } = useModelDetail(modelId || null)
  const { data: testInfo } = useModelTestInfo(modelId || null)
  const forecastMutation = useForecastSingle()

  // Reset slider when model changes
  useEffect(() => {
    if (testInfo) {
      setSliderIdx(testInfo.valid_start_idx)
    } else {
      setSliderIdx(null)
    }
    forecastMutation.reset()
  }, [modelId, testInfo]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-trigger prediction when slider changes (debounced to avoid spam)
  useEffect(() => {
    if (!modelId || !testInfo || sliderIdx === null) return
    const startDate = testInfo.test_dates[sliderIdx]
    if (!startDate) return

    const timer = setTimeout(() => {
      forecastMutation.mutate({
        model_id: modelId,
        start_date: startDate,
      })
    }, 300)
    return () => clearTimeout(timer)
  }, [modelId, sliderIdx]) // eslint-disable-line react-hooks/exhaustive-deps

  const inputChunkLength = testInfo?.input_chunk_length
  const outputChunkLength = testInfo?.output_chunk_length

  // Use one-step metrics/predictions (exact, as in Streamlit)
  const result = forecastMutation.data ?? null
  const displayResult = useMemo(() => {
    if (!result) return null
    // If one-step predictions available, use those for display
    if (result.predictions_onestep) {
      return {
        ...result,
        predictions: result.predictions_onestep,
        metrics: result.metrics_onestep ?? result.metrics,
      }
    }
    return result
  }, [result])

  const isPending = forecastMutation.isPending
  const isError = forecastMutation.isError
  const error = forecastMutation.error

  // Compute window dates for display
  const windowInfo = useMemo(() => {
    if (!testInfo || sliderIdx === null) return null
    const startDate = testInfo.test_dates[sliderIdx]
    const endIdx = Math.min(sliderIdx + testInfo.output_chunk_length - 1, testInfo.test_length - 1)
    const endDate = testInfo.test_dates[endIdx]
    const contextStartIdx = Math.max(0, sliderIdx - testInfo.input_chunk_length)
    const contextStartDate = testInfo.test_dates[contextStartIdx]
    return {
      contextStartDate,
      startDate,
      endDate,
      horizon: testInfo.output_chunk_length,
      inputChunk: testInfo.input_chunk_length,
    }
  }, [testInfo, sliderIdx])

  // Relative scale info from actuals
  const relativeInfo = useMemo(() => {
    if (!displayResult) return null
    const actuals = displayResult.actuals.filter((v): v is number => v !== null)
    if (actuals.length < 4) return null
    const sorted = [...actuals].sort((a, b) => a - b)
    const q25 = sorted[Math.floor(sorted.length * 0.25)]
    const q75 = sorted[Math.floor(sorted.length * 0.75)]
    const iqr = q75 - q25
    const scale = iqr > 0 ? iqr : undefined
    const mae = displayResult.metrics['MAE']
    const rmse = displayResult.metrics['RMSE']
    if (!scale || mae == null) return null
    return {
      relMAE: (mae / scale) * 100,
      relRMSE: rmse != null ? (rmse / scale) * 100 : null,
      scaleLabel: 'IQR',
      scaleValue: iqr,
    }
  }, [displayResult])

  // CSV export
  const handleDownloadCSV = useCallback(() => {
    if (!displayResult) return
    const lines = ['date,ground_truth,prediction']
    for (let i = 0; i < displayResult.dates.length; i++) {
      const date = displayResult.dates[i]
      const actual = displayResult.actuals[i] != null ? String(displayResult.actuals[i]) : ''
      const predicted = displayResult.predictions[i] != null ? String(displayResult.predictions[i]) : ''
      lines.push(`${date},${actual},${predicted}`)
    }
    const csv = lines.join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    const dateStr = windowInfo?.startDate ? new Date(windowInfo.startDate).toISOString().slice(0, 10) : 'unknown'
    a.href = url
    a.download = `prediction_${dateStr}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [displayResult, windowInfo])

  // Filtered hyperparams for collapsible section
  const displayHyperparams = useMemo(() => {
    if (!modelDetail?.hyperparams) return null
    const skipKeys = new Set([
      'train_size', 'val_size', 'test_size', 'n_train', 'n_val', 'n_test',
      'test_start_date', 'test_end_date',
    ])
    const entries = Object.entries(modelDetail.hyperparams).filter(
      ([key]) => !skipKeys.has(key),
    )
    return entries.length > 0 ? entries : null
  }, [modelDetail])

  // Dataset splits
  const datasetSplits = useMemo(() => {
    const hp = modelDetail?.hyperparams
    if (!hp) return null
    const trainSize = hp['train_size'] ?? hp['n_train']
    const valSize = hp['val_size'] ?? hp['n_val']
    const testSize = hp['test_size'] ?? hp['n_test']
    if (trainSize != null || valSize != null || testSize != null) {
      return { train: trainSize as number | undefined, val: valSize as number | undefined, test: testSize as number | undefined }
    }
    return null
  }, [modelDetail])

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-1">Prevision</h1>
        <p className="text-sm text-text-secondary">
          Prediction piezometrique avec fenetre glissante sur le jeu de test
        </p>
      </div>

      {/* Model selection + info panel */}
      <div className="bg-bg-card rounded-xl border border-white/5 p-5">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Left: Model selector */}
          <div>
            <ModelSelector value={modelId} onChange={setModelId} />
          </div>

          {/* Center: Dataset info */}
          {modelDetail && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                Dataset
              </h4>
              {datasetSplits && (
                <div className="space-y-1">
                  {datasetSplits.train != null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Train</span>
                      <span className="text-text-primary">{datasetSplits.train} pts</span>
                    </div>
                  )}
                  {datasetSplits.val != null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Validation</span>
                      <span className="text-text-primary">{datasetSplits.val} pts</span>
                    </div>
                  )}
                  {datasetSplits.test != null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Test</span>
                      <span className="text-text-primary">{datasetSplits.test} pts</span>
                    </div>
                  )}
                </div>
              )}
              {testInfo && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-text-secondary">Test range</span>
                    <span className="text-text-primary text-[10px]">
                      {new Date(testInfo.test_dates[0]).toLocaleDateString('fr-FR')} — {new Date(testInfo.test_dates[testInfo.test_length - 1]).toLocaleDateString('fr-FR')}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Right: Model architecture */}
          {modelDetail && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                Modele
              </h4>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-text-secondary">Type</span>
                  <span className="text-text-primary">{modelDetail.model_type}</span>
                </div>
                {inputChunkLength != null && (
                  <div className="flex justify-between text-xs">
                    <span className="text-text-secondary">Input</span>
                    <span className="text-text-primary">{inputChunkLength} jours</span>
                  </div>
                )}
                {outputChunkLength != null && (
                  <div className="flex justify-between text-xs">
                    <span className="text-text-secondary">Horizon</span>
                    <span className="text-text-primary">{outputChunkLength} jours</span>
                  </div>
                )}
                {modelDetail.preprocessing_config?.normalization != null && (
                  <div className="flex justify-between text-xs">
                    <span className="text-text-secondary">Scaler</span>
                    <span className="text-text-primary">{String(modelDetail.preprocessing_config.normalization as string)}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Collapsible hyperparams */}
        {displayHyperparams && (
          <div className="mt-4 pt-3 border-t border-white/5">
            <button
              onClick={() => setHyperparamsOpen(!hyperparamsOpen)}
              className="flex items-center gap-1.5 text-xs text-text-secondary hover:text-text-primary transition-colors"
            >
              {hyperparamsOpen ? (
                <ChevronDown className="w-3.5 h-3.5" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5" />
              )}
              Hyperparametres ({displayHyperparams.length})
            </button>
            {hyperparamsOpen && (
              <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-1 bg-white/[0.02] rounded-lg p-3">
                {displayHyperparams.map(([key, val]) => (
                  <div key={key} className="flex justify-between text-xs py-0.5">
                    <span className="text-text-secondary truncate mr-2">{key}</span>
                    <span className="text-text-primary font-mono text-[11px] shrink-0">
                      {typeof val === 'boolean'
                        ? val ? 'true' : 'false'
                        : typeof val === 'object'
                          ? JSON.stringify(val)
                          : String(val ?? '')}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Sliding window slider */}
      {testInfo && sliderIdx !== null && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-text-primary">
              Fenetre glissante sur le jeu de test ({testInfo.output_chunk_length}j de prediction)
            </h3>
            {windowInfo && (
              <p className="text-xs text-text-secondary">
                {new Date(windowInfo.startDate).toLocaleDateString('fr-FR')} → {new Date(windowInfo.endDate).toLocaleDateString('fr-FR')} ({windowInfo.horizon}j)
              </p>
            )}
          </div>

          <div className="flex items-center gap-4">
            <span className="text-[10px] text-text-secondary whitespace-nowrap">
              {new Date(testInfo.test_dates[testInfo.valid_start_idx]).toLocaleDateString('fr-FR')}
            </span>
            <input
              type="range"
              min={testInfo.valid_start_idx}
              max={testInfo.valid_end_idx}
              value={sliderIdx}
              onChange={(e) => setSliderIdx(Number(e.target.value))}
              className="flex-1 accent-accent-cyan h-2"
            />
            <span className="text-[10px] text-text-secondary whitespace-nowrap">
              {new Date(testInfo.test_dates[testInfo.valid_end_idx]).toLocaleDateString('fr-FR')}
            </span>
          </div>

          <p className="text-[10px] text-text-secondary">
            Input : {testInfo.input_chunk_length} jours de contexte | Prediction : {testInfo.output_chunk_length} jours
          </p>
        </div>
      )}

      {/* Full test set overview with sliding window */}
      {testInfo && testInfo.test_values && sliderIdx !== null && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-5">
          <h3 className="text-sm font-semibold text-text-primary mb-2">
            Test set overview
          </h3>
          <TestSetOverview
            testDates={testInfo.test_dates}
            testValues={testInfo.test_values}
            sliderIdx={sliderIdx}
            inputChunkLength={testInfo.input_chunk_length}
            outputChunkLength={testInfo.output_chunk_length}
            windowResult={displayResult}
            className="h-[250px]"
          />
        </div>
      )}

      {isError && (
        <div className="bg-accent-red/10 border border-accent-red/20 rounded-xl p-4">
          <p className="text-sm text-accent-red">
            Error: {(error as Error).message}
          </p>
        </div>
      )}

      {/* Window detail: chart + metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart (2/3 width) */}
        <div className="lg:col-span-2">
          <ForecastView
            result={displayResult}
            isLoading={isPending}
            inputChunkLength={inputChunkLength}
            className="min-h-[400px]"
          />
        </div>

        {/* Metrics panel (1/3 width) */}
        <div className="space-y-4">
          {displayResult ? (
            <>
              {/* Metrics grid - ordered like Streamlit */}
              <div className="bg-bg-card rounded-xl border border-white/5 p-4 space-y-3">
                <h4 className="text-sm font-semibold text-text-primary">Metriques de la fenetre</h4>
                <div className="grid grid-cols-2 gap-2">
                  {METRIC_ORDER.filter((key) => displayResult.metrics[key] != null).map((key) => {
                    const val = displayResult.metrics[key]
                    return (
                      <div
                        key={key}
                        className="bg-bg-hover rounded-lg p-3 text-center group relative"
                        title={METRIC_TOOLTIPS[key]}
                      >
                        <p className="text-[10px] text-text-secondary uppercase mb-1">{key}</p>
                        <p className={`text-base font-bold ${metricColor(key, val)}`}>
                          {formatMetric(key, val)}
                        </p>
                      </div>
                    )
                  })}
                </div>

                {/* Relative scale info */}
                {relativeInfo && (
                  <div className="bg-bg-hover rounded-lg p-3">
                    <p className="text-[10px] text-text-secondary">
                      MAE ≈ <span className="text-text-primary font-medium">{relativeInfo.relMAE.toFixed(1)}%</span>
                      {relativeInfo.relRMSE != null && (
                        <> et RMSE ≈ <span className="text-text-primary font-medium">{relativeInfo.relRMSE.toFixed(1)}%</span></>
                      )}
                      {' '}de l'echelle ({relativeInfo.scaleLabel} = {relativeInfo.scaleValue.toFixed(4)})
                    </p>
                  </div>
                )}
              </div>

              {/* Window info */}
              {windowInfo && (
                <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                  <h4 className="text-xs font-semibold text-text-secondary mb-3 uppercase tracking-wide">
                    Fenetre de prevision
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Debut</span>
                      <span className="text-text-primary">{new Date(windowInfo.startDate).toLocaleDateString('fr-FR')}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">End</span>
                      <span className="text-text-primary">{new Date(windowInfo.endDate).toLocaleDateString('fr-FR')}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">Points</span>
                      <span className="text-text-primary">{displayResult.dates.length}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* CSV export */}
              <button
                onClick={handleDownloadCSV}
                className="w-full flex items-center justify-center gap-1.5 text-xs text-accent-cyan hover:text-accent-cyan/80 transition-colors px-3 py-2 rounded-lg border border-accent-cyan/20 hover:border-accent-cyan/40"
              >
                <Download className="w-3.5 h-3.5" />
                Export CSV
              </button>
            </>
          ) : (
            <div className="bg-bg-card rounded-xl border border-white/5 p-6 flex items-center justify-center min-h-[200px]">
              <p className="text-xs text-text-secondary text-center">
                {modelId
                  ? 'Deplacez le slider pour generer une prevision.'
                  : 'Selectionnez un modele pour commencer.'}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Explainability */}
      {modelId && displayResult && (
        <ExplainabilityPanel modelId={modelId} />
      )}
    </div>
  )
}
