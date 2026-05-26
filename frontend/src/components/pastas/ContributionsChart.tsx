import Plot from 'react-plotly.js'
import type { TimeSeriesData } from '@/lib/types'

const COLORS = ['#60a5fa', '#34d399', '#f97316', '#a78bfa', '#f43f5e']

const LABELS: Record<string, string> = {
  recharge: 'Recharge (P − f·E)',
  constant_d: 'Niveau de base',
  temperature: 'Effet de la température',
}

const DESCRIPTIONS: Record<string, string> = {
  recharge: "Comment les précipitations moins l'évapotranspiration influencent la nappe. Positif = la nappe monte (période humide), négatif = elle baisse (période sèche). La forme et la chronologie révèlent la réponse de l'aquifère au climat.",
  constant_d: "Niveau de référence autour duquel la nappe fluctue. Les changements indiquent une dérive à long terme ou un niveau de référence.",
  temperature: "Effet thermique direct sur l'aquifère (dilatation, changements de viscosité). Faible par rapport à la recharge mais capture les variations saisonnières liées à la température non expliquées par la seule évapotranspiration.",
}

interface Props {
  contributions: Record<string, TimeSeriesData>
}

export function ContributionsChart({ contributions }: Props) {
  const entries = Object.entries(contributions)
  if (entries.length === 0) return null

  return (
    <div className="space-y-2">
      <p className="text-[10px] text-text-muted">
        Chaque panneau montre une contribution de stress au niveau simulé. La somme de toutes les contributions est égale à la simulation.
      </p>
      {entries.map(([name, ts], i) => {
        const color = COLORS[i % COLORS.length]
        const label = LABELS[name] ?? name
        const vals = ts.values.filter(v => Number.isFinite(v))
        const range = vals.length > 0 ? Math.max(...vals) - Math.min(...vals) : 0

        return (
          <div key={name} className="bg-bg-card rounded-lg border border-white/5 p-2">
            <div className="flex items-center justify-between mb-0.5">
              <span className="text-[10px] font-semibold" style={{ color }}>{label}</span>
              <span className="text-[10px] text-text-muted">amplitude : {range.toFixed(2)} m</span>
            </div>
            {DESCRIPTIONS[name] && <p className="text-[9px] text-text-muted mb-1 leading-relaxed">{DESCRIPTIONS[name]}</p>}
            <Plot
              data={[{
                x: ts.index,
                y: ts.values,
                name: label,
                type: 'scatter' as const,
                mode: 'lines' as const,
                line: { color, width: 1.5 },
              }]}
              layout={{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#9ca3af', size: 9 },
                margin: { t: 5, r: 15, b: 25, l: 50 },
                height: 130,
                xaxis: { gridcolor: 'rgba(255,255,255,0.03)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.05)', title: { text: 'm', font: { size: 9 } } },
                showlegend: false,
              }}
              useResizeHandler
              className="w-full"
              config={{ displayModeBar: false }}
            />
          </div>
        )
      })}
    </div>
  )
}
