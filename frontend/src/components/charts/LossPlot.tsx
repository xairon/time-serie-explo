import Plot from 'react-plotly.js'
import { useTranslation } from 'react-i18next'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Layout } from 'plotly.js-dist-min'

interface LossPlotProps {
  trainLoss: number[]
  valLoss: number[]
  className?: string
}

export function LossPlot({ trainLoss, valLoss, className = '' }: LossPlotProps) {
  const { t } = useTranslation()
  const epochs = trainLoss.map((_, i) => i + 1)

  // First-epoch loss spikes (MLP-mixers can start at ~1e3 before any update)
  // otherwise dominate the y-axis and squash the entire convergence region into
  // a flat line at the bottom — making train/val look identical and frozen.
  // Scale the axis to the values AFTER epoch 1; the initial spike clips off top.
  const tail = [...trainLoss.slice(1), ...valLoss.slice(1)].filter((v) => Number.isFinite(v))
  const yMax = tail.length ? Math.max(...tail) : Math.max(...trainLoss, ...valLoss, 1)
  const yRange: [number, number] | null = tail.length ? [0, yMax * 1.15] : null

  const layout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: 'Epoch' } },
    yaxis: {
      ...darkLayout.yaxis,
      title: { text: 'Loss' },
      ...(yRange ? { range: yRange, autorange: false as const } : {}),
    },
  }

  return (
    <div className={className}>
      <Plot
        data={[
          {
            x: epochs,
            y: trainLoss,
            type: 'scatter',
            mode: 'lines',
            name: t('sharedComponents.charts.training'),
            line: { color: '#06b6d4', width: 2 },
          },
          {
            x: epochs.slice(0, valLoss.length),
            y: valLoss,
            type: 'scatter',
            mode: 'lines',
            name: t('sharedComponents.charts.validation'),
            line: { color: '#f59e0b', width: 2 },
          },
        ]}
        layout={layout}
        config={plotlyConfig}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
