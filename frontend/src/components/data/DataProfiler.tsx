import Plot from 'react-plotly.js'
import { useTranslation } from 'react-i18next'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Layout } from 'plotly.js-dist-min'

interface ProfileStats {
  column: string
  count: number
  mean: number | null
  std: number | null
  min: number | null
  max: number | null
  missing: number
  missingPct: number
}

interface DataProfilerProps {
  stats: ProfileStats[]
  className?: string
}

export function DataProfiler({ stats, className = '' }: DataProfilerProps) {
  const { t } = useTranslation()
  const missingLayout: Partial<Layout> = {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: { text: t('sharedComponents.charts.missingPercent') }, range: [0, 100] },
    yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const },
    margin: { t: 10, r: 20, b: 40, l: 120 },
    height: Math.max(200, stats.length * 28),
  }

  const headers = [
    t('sharedComponents.charts.variable'),
    t('sharedComponents.charts.n'),
    t('sharedComponents.charts.mean'),
    t('sharedComponents.charts.stddev'),
    t('sharedComponents.charts.min'),
    t('sharedComponents.charts.max'),
    t('sharedComponents.charts.missingPercent'),
  ]

  return (
    <div className={`space-y-4 ${className}`}>
      <h4 className="text-sm font-semibold text-text-primary">{t('sharedComponents.charts.descriptiveStats')}</h4>

      <div className="overflow-x-auto rounded-lg border border-white/5">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-bg-hover">
              {headers.map(
                (h) => (
                  <th
                    key={h}
                    className="px-3 py-2 text-left font-medium text-text-secondary uppercase tracking-wide"
                  >
                    {h}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {stats.map((s) => (
              <tr key={s.column} className="border-t border-white/5">
                <td className="px-3 py-1.5 text-text-primary font-medium">{s.column}</td>
                <td className="px-3 py-1.5 text-text-primary">{s.count}</td>
                <td className="px-3 py-1.5 text-text-primary">
                  {s.mean !== null ? s.mean.toFixed(3) : '—'}
                </td>
                <td className="px-3 py-1.5 text-text-primary">
                  {s.std !== null ? s.std.toFixed(3) : '—'}
                </td>
                <td className="px-3 py-1.5 text-text-primary">
                  {s.min !== null ? s.min.toFixed(3) : '—'}
                </td>
                <td className="px-3 py-1.5 text-text-primary">
                  {s.max !== null ? s.max.toFixed(3) : '—'}
                </td>
                <td className="px-3 py-1.5">
                  <span
                    className={
                      s.missingPct > 10
                        ? 'text-accent-red'
                        : s.missingPct > 0
                          ? 'text-accent-amber'
                          : 'text-accent-green'
                    }
                  >
                    {s.missingPct.toFixed(1)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h4 className="text-sm font-semibold text-text-primary">{t('sharedComponents.charts.missingValues')}</h4>
      <div className="h-[250px]">
        <Plot
          data={[
            {
              type: 'bar',
              orientation: 'h',
              y: stats.map((s) => s.column),
              x: stats.map((s) => s.missingPct),
              marker: {
                color: stats.map((s) =>
                  s.missingPct > 10 ? '#ef4444' : s.missingPct > 0 ? '#f59e0b' : '#10b981',
                ),
              },
            },
          ]}
          layout={missingLayout}
          config={plotlyConfig}
          useResizeHandler
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  )
}
