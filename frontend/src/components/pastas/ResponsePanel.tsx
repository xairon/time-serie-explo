import Plot from 'react-plotly.js'
import type { TimeSeriesData, FitParameter } from '@/lib/types'
import type Plotly from 'plotly.js-dist-min'

interface Props {
  stepResponse: TimeSeriesData
  blockResponse: TimeSeriesData
  parameters: FitParameter[]
  responseType: string
}

export function ResponsePanel({ stepResponse, blockResponse, parameters, responseType }: Props) {
  const hasStep = stepResponse?.values?.length > 0
  const hasBlock = blockResponse?.values?.length > 0

  if (!hasStep && !hasBlock) return null

  const responseParams = parameters.filter((p) => p.name.startsWith('recharge_'))

  let t50: number | null = null
  let t95: number | null = null
  if (hasStep) {
    const vals = stepResponse.values
    const finalVal = vals[vals.length - 1]
    if (finalVal !== 0) {
      const i50 = vals.findIndex((v) => Math.abs(v) >= Math.abs(finalVal) * 0.5)
      const i95 = vals.findIndex((v) => Math.abs(v) >= Math.abs(finalVal) * 0.95)
      t50 = i50 === -1 ? null : i50
      t95 = i95 === -1 ? null : i95
    }
  }

  const chartBase = {
    paper_bgcolor: 'transparent' as const,
    plot_bgcolor: 'transparent' as const,
    font: { color: '#9ca3af', size: 9 },
    margin: { t: 25, r: 10, b: 30, l: 40 },
    height: 200,
    showlegend: false,
  }

  const title = responseType ? `Response Function — ${responseType}` : 'Response Function'

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
        {title}
      </div>

      <div className="flex flex-wrap gap-2">
        {responseParams.map((p) => (
          <div
            key={p.name}
            className="bg-bg-primary rounded px-2 py-1 text-xs border border-white/5"
          >
            <span className="text-text-muted">{p.name.replace('recharge_', '')}</span>
            <span className="ml-1 text-text-primary font-mono">{p.optimal.toFixed(4)}</span>
            {p.stderr != null && (
              <span className="text-text-muted"> ± {p.stderr.toFixed(4)}</span>
            )}
          </div>
        ))}
        {t50 != null && (
          <div className="bg-bg-primary rounded px-2 py-1 text-xs border border-accent-cyan/20">
            <span className="text-text-muted">t₅₀</span>
            <span className="ml-1 font-mono text-accent-cyan">{t50} j</span>
          </div>
        )}
        {t95 != null && (
          <div className="bg-bg-primary rounded px-2 py-1 text-xs border border-accent-cyan/20">
            <span className="text-text-muted">t₉₅</span>
            <span className="ml-1 font-mono text-accent-cyan">{t95} j</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        {hasStep && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <Plot
              data={[
                {
                  y: stepResponse.values,
                  type: 'scatter',
                  mode: 'lines',
                  line: { color: '#34d399', width: 2 },
                },
              ]}
              layout={{
                ...chartBase,
                title: { text: 'Step Response', font: { size: 11 } },
                xaxis: { title: { text: 'Days' }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: { text: 'm' }, gridcolor: 'rgba(255,255,255,0.05)' },
                shapes: (() => {
                  const shapes: Partial<Plotly.Shape>[] = []
                  const vals = stepResponse.values
                  const finalVal = vals[vals.length - 1]
                  if (finalVal !== 0) {
                    shapes.push({
                      type: 'line',
                      x0: 0,
                      x1: vals.length,
                      y0: finalVal * 0.5,
                      y1: finalVal * 0.5,
                      line: { color: 'rgba(255,255,255,0.2)', dash: 'dot', width: 1 },
                    })
                    shapes.push({
                      type: 'line',
                      x0: 0,
                      x1: vals.length,
                      y0: finalVal * 0.95,
                      y1: finalVal * 0.95,
                      line: { color: 'rgba(255,255,255,0.2)', dash: 'dot', width: 1 },
                    })
                  }
                  return shapes
                })(),
              }}
              useResizeHandler
              className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        )}
        {hasBlock && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <Plot
              data={[
                {
                  y: blockResponse.values,
                  type: 'scatter',
                  mode: 'lines',
                  line: { color: '#f97316', width: 2 },
                },
              ]}
              layout={{
                ...chartBase,
                title: { text: 'Block Response', font: { size: 11 } },
                xaxis: { title: { text: 'Days' }, gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { title: { text: 'm/d' }, gridcolor: 'rgba(255,255,255,0.05)' },
              }}
              useResizeHandler
              className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        )}
      </div>
    </div>
  )
}
