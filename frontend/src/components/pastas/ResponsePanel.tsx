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
    const idx = stepResponse.index
    const finalVal = vals[vals.length - 1]
    if (finalVal !== 0) {
      const i50 = vals.findIndex((v) => Math.abs(v) >= Math.abs(finalVal) * 0.5)
      const i95 = vals.findIndex((v) => Math.abs(v) >= Math.abs(finalVal) * 0.95)
      t50 = i50 >= 0 && idx[i50] ? Math.round(parseFloat(idx[i50])) || i50 : null
      t95 = i95 >= 0 && idx[i95] ? Math.round(parseFloat(idx[i95])) || i95 : null
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
        {responseParams.map((p) => {
          const paramName = p.name.replace('recharge_', '')
          const paramTooltips: Record<string, string> = {
            A: 'Gain - steady-state response amplitude. The larger A, the more sensitive the aquifer is to input (recharge).',
            n: 'Shape - controls the signal rise. n > 1: delayed response (inertia). n close to 1: fast response.',
            a: 'Time scale - controls the response duration. The larger a, the longer the aquifer takes to react.',
            f: 'Fraction (FlexModel) - split between fast and slow recharge.',
          }
          return (
            <div
              key={p.name}
              className="bg-bg-primary rounded px-2 py-1 text-xs border border-white/5"
              title={paramTooltips[paramName] ?? ''}
            >
              <span className="text-text-muted">{paramName}</span>
              <span className="ml-1 text-text-primary font-mono">{p.optimal.toFixed(4)}</span>
              {p.stderr != null && (
                <span className="text-text-muted"> ± {p.stderr.toFixed(4)}</span>
              )}
            </div>
          )
        })}
        {t50 != null && (
          <div className="bg-bg-primary rounded px-2 py-1 text-xs border border-accent-cyan/20" title="Half-response time — delay to reach 50% of the final effect. Indicates the initial reactivity of the aquifer.">
            <span className="text-text-muted">t₅₀</span>
            <span className="ml-1 font-mono text-accent-cyan">{t50} d</span>
          </div>
        )}
        {t95 != null && (
          <div className="bg-bg-primary rounded px-2 py-1 text-xs border border-accent-cyan/20" title="95% response time — delay to reach 95% of the final effect. Represents the effective memory of the aquifer. Short (< 100 d) = alluvial. Long (> 500 d) = deep sedimentary or confined.">
            <span className="text-text-muted">t₉₅</span>
            <span className="ml-1 font-mono text-accent-cyan">{t95} d</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        {hasStep && (
          <div className="bg-bg-card rounded-lg border border-white/5 p-2">
            <p className="text-[9px] text-text-muted px-1 mb-0.5">If recharge increases by 1 mm/d and stays constant, how the water level rises over time. The curve eventually stabilizes at the gain A.</p>
            <Plot
              data={[
                {
                  x: stepResponse.index.map(Number),
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
            <p className="text-[9px] text-text-muted px-1 mb-0.5">Effect of a single-day recharge pulse. The peak shows the maximum reactivity, the decline shows the drainage speed.</p>
            <Plot
              data={[
                {
                  x: blockResponse.index.map(Number),
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
                yaxis: { title: { text: 'm/(mm/d)' }, gridcolor: 'rgba(255,255,255,0.05)' },
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
