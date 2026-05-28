import Plot from 'react-plotly.js'
import { useTranslation } from 'react-i18next'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface Props { data: any }

export function CrossCorrelationPanel({ data }: Props) {
  const { t } = useTranslation()
  if (!data?.lags_months?.length) return <p className="text-xs text-text-muted">{t('pastas.crossCorr.notEnough')}</p>

  const shapes: any[] = []
  const annotations: any[] = []

  if (data.max_lag_months != null) {
    shapes.push({ type: 'line', x0: data.max_lag_months, x1: data.max_lag_months, y0: 0, y1: 1, yref: 'paper', line: { color: '#22d3ee', dash: 'dash', width: 1.5 } })
    annotations.push({ x: data.max_lag_months, y: 1.05, yref: 'paper', text: t('pastas.crossCorr.peakAnnot', { lag: data.max_lag_months, r: data.max_correlation }), showarrow: false, font: { size: 10, color: '#22d3ee' } })
  }
  if (data.t95_months != null) {
    shapes.push({ type: 'line', x0: data.t95_months, x1: data.t95_months, y0: 0, y1: 1, yref: 'paper', line: { color: '#f97316', dash: 'dot', width: 1.5 } })
    annotations.push({ x: data.t95_months, y: 0.95, yref: 'paper', text: t('pastas.crossCorr.t95Annot', { lag: data.t95_months }), showarrow: false, font: { size: 10, color: '#f97316' } })
  }

  return (
    <div>
      <Plot
        data={[{
          x: data.lags_months, y: data.correlation, type: 'bar' as const,
          marker: { color: data.lags_months.map((l: number) => l === data.max_lag_months ? '#22d3ee' : 'rgba(156,163,175,0.4)') },
        }]}
        layout={{ ...darkLayout, margin: { t: 20, r: 20, b: 40, l: 55 }, height: 220,
          xaxis: { ...darkLayout.xaxis, title: { text: t('pastas.crossCorr.lagMonths') } },
          yaxis: { ...darkLayout.yaxis, title: { text: t('pastas.crossCorr.correlation') } },
          shapes, annotations,
        }}
        config={plotlyConfig} style={{ width: '100%' }}
      />
      {data.t95_months != null && data.max_lag_months != null && (
        <p className="text-[10px] text-text-muted mt-1">
          {t('pastas.crossCorr.empiricalPeak', { lag: data.max_lag_months, t95: data.t95_months })}
          {Math.abs(data.max_lag_months - data.t95_months) > 2 && <span className="text-orange-400"> {t('pastas.crossCorr.gapWarn')}</span>}
        </p>
      )}
    </div>
  )
}
