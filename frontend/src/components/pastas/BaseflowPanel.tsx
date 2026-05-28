import Plot from 'react-plotly.js'
import { useTranslation } from 'react-i18next'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface Props { data: any }

export function BaseflowPanel({ data }: Props) {
  const { t } = useTranslation()
  if (!data?.index?.length) return <p className="text-xs text-text-muted">{t('pastas.baseflow.notEnough')}</p>

  const bfiLabel = data.bfi > 0.7 ? t('pastas.baseflow.labelBase') : data.bfi > 0.4 ? t('pastas.baseflow.labelMixed') : t('pastas.baseflow.labelRunoff')

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <div className="bg-bg-primary rounded-lg border border-white/5 px-3 py-2">
          <div className="text-[10px] text-text-muted">{t('pastas.baseflow.baseflowIndex')}</div>
          <div className="text-lg font-semibold text-accent-cyan font-mono">{data.bfi}</div>
        </div>
        <span className="text-xs text-text-secondary">{bfiLabel}</span>
      </div>
      <Plot
        data={[
          { x: data.index, y: data.baseflow, type: 'scatter' as const, mode: 'lines' as const, fill: 'tozeroy' as const,
            name: t('pastas.baseflow.baseflow'), line: { width: 0 }, fillcolor: 'rgba(59,130,246,0.3)' },
          { x: data.index, y: data.observed, type: 'scatter' as const, mode: 'lines' as const, fill: 'tonexty' as const,
            name: t('pastas.baseflow.runoff'), line: { width: 0 }, fillcolor: 'rgba(34,211,238,0.3)' },
          { x: data.index, y: data.observed, type: 'scatter' as const, mode: 'lines' as const,
            name: t('pastas.baseflow.observed'), line: { color: '#9ca3af', width: 1 } },
        ]}
        layout={{ ...darkLayout, margin: { t: 10, r: 20, b: 40, l: 55 }, height: 250,
          yaxis: { ...darkLayout.yaxis, title: { text: 'm NGF' } } }}
        config={plotlyConfig} style={{ width: '100%' }}
      />
    </div>
  )
}
