import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface Props { data: any }

export function RegionalResidualsPanel({ data }: Props) {
  if (!data?.index?.length) return <p className="text-xs text-text-muted">Pas assez de données de résidus pour la comparaison</p>

  const traces: any[] = data.siblings?.map((s: any, i: number) => ({
    x: data.index, y: s.zscore, type: 'scatter' as const, mode: 'lines' as const,
    name: s.code_bss.split('/').pop() ?? s.code_bss, line: { color: 'rgba(156,163,175,0.25)', width: 1 },
    showlegend: i === 0, ...(i === 0 ? { name: 'Voisins' } : {}),
  })) ?? []

  traces.push({
    x: data.index, y: data.model_residual_zscore, type: 'scatter' as const, mode: 'lines' as const,
    name: 'Résidus du modèle', line: { color: '#22d3ee', width: 2 },
  })

  return (
    <div>
      <Plot
        data={traces}
        layout={{ ...darkLayout, margin: { t: 10, r: 20, b: 40, l: 55 }, height: 250,
          yaxis: { ...darkLayout.yaxis, title: { text: 'Score Z' } },
          shapes: [{ type: 'line', x0: data.index[0], x1: data.index[data.index.length - 1], y0: 0, y1: 0, line: { color: 'rgba(255,255,255,0.1)', width: 1 } }],
        }}
        config={plotlyConfig} style={{ width: '100%' }}
      />
      {data.correlation_with_siblings != null && (
        <p className="text-[10px] text-text-muted mt-1">
          Corrélation moyenne avec les voisins : <span className="font-mono text-text-secondary">{data.correlation_with_siblings}</span>
          {data.correlation_with_siblings > 0.5 ? <span className="text-blue-400"> — les résidus suivent un schéma régional</span> : <span className="text-yellow-400"> — les résidus semblent spécifiques à la station</span>}
        </p>
      )}
    </div>
  )
}
