import Plot from 'react-plotly.js'
import { useTranslation } from 'react-i18next'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface Props { data: any }

export function RecessionPanel({ data }: Props) {
  const { t } = useTranslation()
  if (!data?.segments?.length) return <p className="text-xs text-text-muted">{t('pastas.recession.noSegments')}</p>

  const layout = { ...darkLayout, margin: { t: 20, r: 20, b: 40, l: 55 }, height: 220 }

  return (
    <div className="space-y-3">
      <div className="flex gap-3">
        {[
          { label: t('pastas.recession.tMedian'), value: data.T_median_days != null ? `${data.T_median_days} j` : '—' },
          { label: t('pastas.recession.segments'), value: data.n_segments },
          { label: t('pastas.recession.tStd'), value: data.T_std_days != null ? t('pastas.recession.tStdValue', { value: data.T_std_days }) : '—' },
        ].map(kpi => (
          <div key={kpi.label} className="bg-bg-primary rounded-lg border border-white/5 px-3 py-2">
            <div className="text-[10px] text-text-muted">{kpi.label}</div>
            <div className="text-sm font-semibold text-text-primary font-mono">{kpi.value}</div>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3">
        {data.mrc && (
          <Plot
            data={[
              ...data.segments.map((s: any, i: number) => ({
                x: Array.from({ length: s.duration_days }, (_, j) => j),
                y: Array.from({ length: s.duration_days }, (_, j) => Math.exp(-j / s.T_days)),
                type: 'scatter' as const, mode: 'lines' as const,
                line: { color: 'rgba(156,163,175,0.3)', width: 1 }, showlegend: i === 0, name: t('pastas.recession.segments'),
              })),
              { x: data.mrc.normalized_time, y: data.mrc.normalized_level, type: 'scatter' as const, mode: 'lines' as const,
                name: 'MRC', line: { color: '#22d3ee', width: 2.5 } },
            ]}
            layout={{ ...layout, xaxis: { ...layout.xaxis, title: { text: t('pastas.response.days') } }, yaxis: { ...layout.yaxis, title: { text: t('pastas.recession.normalizedLevel') } } }}
            config={plotlyConfig} style={{ width: '100%' }}
          />
        )}
        <Plot
          data={[{
            x: data.segments.map((s: any) => s.T_days), type: 'histogram' as const,
            marker: { color: 'rgba(34,211,238,0.5)', line: { color: '#22d3ee', width: 1 } },
            name: t('pastas.recession.distributionT'),
          }]}
          layout={{ ...layout, xaxis: { ...layout.xaxis, title: { text: t('pastas.recession.tBin') } }, yaxis: { ...layout.yaxis, title: { text: t('pastas.recession.count') } },
            shapes: data.T_median_days ? [{ type: 'line', x0: data.T_median_days, x1: data.T_median_days, y0: 0, y1: 1, yref: 'paper', line: { color: '#ef4444', dash: 'dash', width: 1.5 } }] : [],
          }}
          config={plotlyConfig} style={{ width: '100%' }}
        />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead><tr className="text-text-muted border-b border-white/5">
            <th className="text-left px-2 py-1.5">{t('pastas.recession.start')}</th><th className="text-left px-2 py-1.5">{t('pastas.recession.end')}</th>
            <th className="text-right px-2 py-1.5">{t('pastas.recession.tColumn')}</th><th className="text-right px-2 py-1.5">{t('pastas.recession.rSquared')}</th>
            <th className="text-right px-2 py-1.5">{t('pastas.recession.duration')}</th>
          </tr></thead>
          <tbody>{data.segments.map((s: any) => (
            <tr key={s.start} className="border-b border-white/5 hover:bg-bg-hover">
              <td className="px-2 py-1 text-text-secondary">{s.start}</td>
              <td className="px-2 py-1 text-text-secondary">{s.end}</td>
              <td className="px-2 py-1 text-right font-mono text-accent-cyan">{s.T_days}</td>
              <td className="px-2 py-1 text-right font-mono text-text-secondary">{s.r_squared}</td>
              <td className="px-2 py-1 text-right text-text-muted">{s.duration_days} j</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  )
}
