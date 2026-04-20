import Plot from 'react-plotly.js'
import type { PastasStationPreview } from '@/lib/types'

interface Props {
  preview: PastasStationPreview
  onRangeChange?: (tmin: string, tmax: string) => void
}

export function DataPreviewPanel({ preview, onRangeChange }: Props) {
  const { piezo, precip, evap, stats, metadata } = preview

  return (
    <div className="space-y-3">
      <div className="flex items-baseline gap-3">
        <span className="text-sm font-mono text-accent-cyan">{preview.code_bss}</span>
        <span className="text-xs text-text-muted">
          {metadata.nom_commune as string} ({metadata.code_departement as string})
        </span>
      </div>

      <div className="grid grid-cols-5 gap-2">
        {[
          { label: 'Observations', value: stats.n_obs_piezo },
          {
            label: 'Période',
            value: Array.isArray(stats.date_range)
              ? `${(stats.date_range[0] as string)?.slice(0, 4)}–${(stats.date_range[1] as string)?.slice(0, 4)}`
              : '—',
          },
          {
            label: 'Niveau moyen',
            value: typeof stats.piezo_mean === 'number' ? `${(stats.piezo_mean as number).toFixed(2)} m` : '—',
          },
          {
            label: 'Gap max',
            value: typeof stats.piezo_max_gap_days === 'number' ? `${stats.piezo_max_gap_days} j` : '—',
          },
          {
            label: 'Couv. journalière',
            value:
              typeof stats.piezo_pct_daily === 'number'
                ? `${(stats.piezo_pct_daily as number).toFixed(0)}%`
                : '—',
          },
        ].map(({ label, value }) => (
          <div
            key={label}
            className="bg-bg-primary rounded-lg p-2 border border-white/5 text-center"
          >
            <div className="text-[10px] text-text-muted">{label}</div>
            <div className="text-sm font-semibold text-text-primary">{value}</div>
          </div>
        ))}
      </div>

      <div className="bg-bg-card rounded-lg border border-white/5 p-2">
        <Plot
          data={[
            {
              x: piezo.index,
              y: piezo.values,
              name: 'Piézo (m)',
              type: 'scatter',
              mode: 'lines',
              line: { color: '#60a5fa', width: 1 },
              xaxis: 'x',
              yaxis: 'y',
            },
            {
              x: precip.index,
              y: precip.values,
              name: 'Précip (mm/j)',
              type: 'bar',
              marker: { color: 'rgba(59,130,246,0.3)' },
              xaxis: 'x',
              yaxis: 'y2',
            },
            {
              x: evap.index,
              y: evap.values,
              name: 'ETP (mm/j)',
              type: 'scatter',
              mode: 'lines',
              line: { color: '#f97316', width: 1 },
              xaxis: 'x',
              yaxis: 'y3',
            },
          ]}
          layout={{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#9ca3af', size: 10 },
            margin: { t: 10, r: 20, b: 40, l: 50 },
            height: 380,
            showlegend: false,
            grid: {
              rows: 3,
              columns: 1,
              subplots: ['xy', 'xy2', 'xy3'],
              roworder: 'top to bottom' as const,
            },
            xaxis: {
              gridcolor: 'rgba(255,255,255,0.03)',
              rangeslider: { visible: true, thickness: 0.06 },
            },
            yaxis: {
              title: { text: 'Piézo (m)' },
              gridcolor: 'rgba(255,255,255,0.05)',
              domain: [0.7, 1],
            },
            yaxis2: {
              title: { text: 'P (mm/j)' },
              gridcolor: 'rgba(255,255,255,0.05)',
              domain: [0.38, 0.65],
            },
            yaxis3: {
              title: { text: 'ETP (mm/j)' },
              gridcolor: 'rgba(255,255,255,0.05)',
              domain: [0.0, 0.3],
            },
          }}
          useResizeHandler
          className="w-full"
          onRelayout={(e: Record<string, unknown>) => {
            if (onRangeChange && e['xaxis.range[0]'] && e['xaxis.range[1]']) {
              onRangeChange(
                String(e['xaxis.range[0]']).slice(0, 10),
                String(e['xaxis.range[1]']).slice(0, 10),
              )
            }
          }}
        />
      </div>
    </div>
  )
}
