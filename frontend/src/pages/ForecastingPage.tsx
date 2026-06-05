import { useState, useMemo, useCallback, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ChevronDown, ChevronRight, Download } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { ModelSelector } from '@/components/forecasting/ModelSelector'
import { ForecastView } from '@/components/forecasting/ForecastView'
import { TestSetOverview } from '@/components/charts/TestSetOverview'
import { ExplainabilityPanel } from '@/components/forecasting/ExplainabilityPanel'
import { useForecastSingle } from '@/hooks/useForecasting'
import { useModelDetail, useModelTestInfo } from '@/hooks/useModels'

/** Display order for metrics (matches Streamlit) */
const METRIC_ORDER = ['MAE', 'RMSE', 'bias', 'sMAPE', 'NRMSE', 'WAPE', 'NSE', 'KGE', 'Dir_Acc']

// Metrics expressed in the target's physical unit (m NGF for piezometry)
const PHYSICAL_METRICS = ['MAE', 'RMSE', 'bias']

/** Color coding for metric values */
function metricColor(key: string, value: number): string {
  const higherBetter = ['NSE', 'KGE', 'Dir_Acc']
  // Physical-unit errors have no universal good/bad threshold → judge in context (see relMAE line)
  if (PHYSICAL_METRICS.includes(key)) return 'text-text-primary'
  if (['sMAPE', 'WAPE', 'NRMSE'].includes(key)) return value < 10 ? 'text-accent-green' : value > 50 ? 'text-accent-red' : 'text-text-primary'
  if (higherBetter.includes(key)) return value > 0.7 ? 'text-accent-green' : value < 0 ? 'text-accent-red' : 'text-text-primary'
  return 'text-text-primary'
}

/** Format metric value with suffix */
function formatMetric(key: string, value: number): string {
  const pctMetrics = ['sMAPE', 'WAPE', 'NRMSE', 'Dir_Acc']
  if (key === 'bias') return `${value >= 0 ? '+' : ''}${value.toFixed(3)} m`
  if (PHYSICAL_METRICS.includes(key)) return `${value.toFixed(3)} m`
  return `${value.toFixed(4)}${pctMetrics.includes(key) ? '%' : ''}`
}

export default function ForecastingPage() {
  const { t } = useTranslation()
  const METRIC_TOOLTIPS: Record<string, string> = {
    MAE: t('mainPages.forecasting.metricTooltipMAE'),
    RMSE: t('mainPages.forecasting.metricTooltipRMSE'),
    bias: t('mainPages.forecasting.metricTooltipBias'),
    sMAPE: t('mainPages.forecasting.metricTooltipSMAPE'),
    WAPE: t('mainPages.forecasting.metricTooltipWAPE'),
    NRMSE: t('mainPages.forecasting.metricTooltipNRMSE'),
    NSE: t('mainPages.forecasting.metricTooltipNSE'),
    KGE: t('mainPages.forecasting.metricTooltipKGE'),
    Dir_Acc: t('mainPages.forecasting.metricTooltipDirAcc'),
  }
  const [modelId, setModelId] = useState('')
  const [aiTab, setAiTab] = useState<'forecast' | 'analysis'>('forecast')
  const [searchParams] = useSearchParams()
  useEffect(() => { const m = searchParams.get('model'); if (m) setModelId(m) }, [searchParams])
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
        <h1 className="text-2xl font-bold text-text-primary">{t('mainPages.forecasting.title')}</h1>
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
                {t('mainPages.forecasting.dataset')}
              </h4>
              {datasetSplits && (
                <div className="space-y-1">
                  {datasetSplits.train != null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">{t('mainPages.forecasting.trainingSplit')}</span>
                      <span className="text-text-primary">{datasetSplits.train} {t('mainPages.forecasting.pointsUnit')}</span>
                    </div>
                  )}
                  {datasetSplits.val != null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">{t('mainPages.forecasting.validation')}</span>
                      <span className="text-text-primary">{datasetSplits.val} {t('mainPages.forecasting.pointsUnit')}</span>
                    </div>
                  )}
                  {datasetSplits.test != null && (
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">{t('mainPages.forecasting.test')}</span>
                      <span className="text-text-primary">{datasetSplits.test} {t('mainPages.forecasting.pointsUnit')}</span>
                    </div>
                  )}
                </div>
              )}
              {testInfo && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-text-secondary">{t('mainPages.forecasting.testRange')}</span>
                    <span className="text-text-primary text-[10px]">
                      {new Date(testInfo.test_dates[0]).toLocaleDateString('en-GB')} — {new Date(testInfo.test_dates[testInfo.test_length - 1]).toLocaleDateString('en-GB')}
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
                {t('mainPages.forecasting.model')}
              </h4>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-text-secondary">{t('mainPages.forecasting.type')}</span>
                  <span className="text-text-primary">{modelDetail.model_type}</span>
                </div>
                {inputChunkLength != null && (
                  <div className="flex justify-between text-xs">
                    <span className="text-text-secondary">{t('mainPages.forecasting.input')}</span>
                    <span className="text-text-primary">{inputChunkLength} {t('mainPages.forecasting.daysUnit')}</span>
                  </div>
                )}
                {outputChunkLength != null && (
                  <div className="flex justify-between text-xs">
                    <span className="text-text-secondary">{t('mainPages.forecasting.horizon')}</span>
                    <span className="text-text-primary">{outputChunkLength} {t('mainPages.forecasting.daysUnit')}</span>
                  </div>
                )}
                {modelDetail.preprocessing_config?.normalization != null && (
                  <div className="flex justify-between text-xs">
                    <span className="text-text-secondary">{t('mainPages.forecasting.scaler')}</span>
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
              {t('mainPages.forecasting.hyperparameters')} ({displayHyperparams.length})
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

      {/* Tabs: forecast vs advanced analysis */}
      <div className="flex items-center gap-1 border-b border-white/5">
        {(['forecast', 'analysis'] as const).map((key) => (
          <button
            key={key}
            onClick={() => setAiTab(key)}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${aiTab === key ? 'border-purple-400 text-text-primary' : 'border-transparent text-text-muted hover:text-text-secondary'}`}
          >
            {key === 'forecast' ? t('mainPages.forecasting.tabForecast', 'Prévision') : t('mainPages.forecasting.tabAnalysis', 'Analyse avancée')}
          </button>
        ))}
      </div>

      {aiTab === 'forecast' && (
      <>
      {/* Sliding window slider */}
      {testInfo && sliderIdx !== null && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-text-primary">
              {t('mainPages.forecasting.slidingWindowTitle', { horizon: testInfo.output_chunk_length })}
            </h3>
            {windowInfo && (
              <p className="text-xs text-text-secondary">
                {new Date(windowInfo.startDate).toLocaleDateString('en-GB')} → {new Date(windowInfo.endDate).toLocaleDateString('en-GB')} ({windowInfo.horizon}j)
              </p>
            )}
          </div>

          <div className="flex items-center gap-4">
            <span className="text-[10px] text-text-secondary whitespace-nowrap">
              {new Date(testInfo.test_dates[testInfo.valid_start_idx]).toLocaleDateString('en-GB')}
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
              {new Date(testInfo.test_dates[testInfo.valid_end_idx]).toLocaleDateString('en-GB')}
            </span>
          </div>

          <p className="text-[10px] text-text-secondary">
            {t('mainPages.forecasting.windowContext', { input: testInfo.input_chunk_length, output: testInfo.output_chunk_length })}
          </p>
        </div>
      )}

      {/* Full test set overview with sliding window */}
      {testInfo && testInfo.test_values && sliderIdx !== null && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-5">
          <h3 className="text-sm font-semibold text-text-primary mb-2">
            {t('mainPages.forecasting.testSetOverview')}
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
            {t('mainPages.forecasting.errorPrefix')} {(error as Error).message}
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
                <h4 className="text-sm font-semibold text-text-primary">{t('mainPages.forecasting.windowMetrics')}</h4>
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
                        <> {t('mainPages.forecasting.relativeScaleInfix')} RMSE ≈ <span className="text-text-primary font-medium">{relativeInfo.relRMSE.toFixed(1)}%</span></>
                      )}
                      {' '}{t('mainPages.forecasting.relativeScaleSuffix')} ({relativeInfo.scaleLabel} = {relativeInfo.scaleValue.toFixed(4)})
                    </p>
                  </div>
                )}
              </div>

              {/* Window info */}
              {windowInfo && (
                <div className="bg-bg-card rounded-xl border border-white/5 p-4">
                  <h4 className="text-xs font-semibold text-text-secondary mb-3 uppercase tracking-wide">
                    {t('mainPages.forecasting.forecastWindow')}
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">{t('mainPages.forecasting.start')}</span>
                      <span className="text-text-primary">{new Date(windowInfo.startDate).toLocaleDateString('en-GB')}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">{t('mainPages.forecasting.end')}</span>
                      <span className="text-text-primary">{new Date(windowInfo.endDate).toLocaleDateString('en-GB')}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-text-secondary">{t('mainPages.forecasting.points')}</span>
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
                {t('mainPages.forecasting.exportCsv')}
              </button>
            </>
          ) : (
            <div className="bg-bg-card rounded-xl border border-white/5 p-6 flex items-center justify-center min-h-[200px]">
              <p className="text-xs text-text-secondary text-center">
                {modelId
                  ? t('mainPages.forecasting.moveSlider')
                  : t('mainPages.forecasting.selectModelToStart')}
              </p>
            </div>
          )}
        </div>
      </div>

      </>
      )}

      {/* Advanced analysis tab */}
      {aiTab === 'analysis' && (
        modelId && displayResult ? (
          <ExplainabilityPanel modelId={modelId} />
        ) : (
          <p className="text-sm text-text-secondary italic">{t('mainPages.forecasting.analysisHint', 'Sélectionnez un modèle et lancez une prévision pour voir l’analyse.')}</p>
        )
      )}
    </div>
  )
}
