import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import type { Data, Layout } from 'plotly.js-dist-min'

interface EmbeddingPoint {
  id: string
  coords: [number, number] | [number, number, number]
  cluster_label: number
  metadata: Record<string, unknown>
  highlighted: boolean
}

interface EmbeddingScatterProps {
  points: EmbeddingPoint[]
  mode: '2d' | '3d'
  colorBy: string
  onPointClick?: (id: string) => void
  loading?: boolean
  className?: string
}

// Qualitative palette for categorical attributes
const CATEGORICAL_COLORS = [
  '#06b6d4', '#8b5cf6', '#f59e0b', '#10b981', '#ef4444',
  '#3b82f6', '#ec4899', '#14b8a6', '#f97316', '#a78bfa',
  '#84cc16', '#fb7185', '#22d3ee', '#fbbf24', '#60a5fa',
]

function isAltitudeColorBy(colorBy: string): boolean {
  return colorBy === 'altitude'
}

function isCategorical(colorBy: string): boolean {
  return !isAltitudeColorBy(colorBy)
}

export function EmbeddingScatter({
  points,
  mode,
  colorBy,
  onPointClick,
  loading = false,
  className = '',
}: EmbeddingScatterProps) {
  const traces: Data[] = []

  const highlighted = points.filter((p) => p.highlighted)
  const others = points.filter((p) => !p.highlighted)

  // Non-highlighted points rendered as a single dim trace
  if (others.length > 0) {
    if (mode === '3d') {
      traces.push({
        type: 'scatter3d',
        name: 'Others',
        x: others.map((p) => (p.coords as [number, number, number])[0]),
        y: others.map((p) => (p.coords as [number, number, number])[1]),
        z: others.map((p) => (p.coords as [number, number, number])[2]),
        mode: 'markers',
        marker: { size: 3, color: '#4b5563', opacity: 0.15 },
        customdata: others.map((p) => [p.id, p.metadata]),
        hovertemplate: '<b>%{customdata[0]}</b><extra></extra>',
        showlegend: true,
      } as Data)
    } else {
      traces.push({
        type: 'scattergl',
        name: 'Others',
        x: others.map((p) => p.coords[0]),
        y: others.map((p) => p.coords[1]),
        mode: 'markers',
        marker: { size: 4, color: '#4b5563', opacity: 0.15 },
        customdata: others.map((p) => [p.id, p.metadata]),
        hovertemplate: '<b>%{customdata[0]}</b><extra></extra>',
        showlegend: true,
      } as Data)
    }
  }

  // Highlighted points
  if (highlighted.length > 0) {
    if (isAltitudeColorBy(colorBy)) {
      // Single trace with continuous color scale
      const altValues = highlighted.map((p) => (p.metadata['altitude'] as number) ?? 0)
      const hoverMeta = highlighted.map((p) => {
        const keys = Object.keys(p.metadata).filter((k) => p.metadata[k] != null && p.metadata[k] !== '').slice(0, 3)
        return keys.map((k) => `${k}: ${p.metadata[k]}`).join('<br>')
      })

      if (mode === '3d') {
        traces.push({
          type: 'scatter3d',
          name: 'Altitude',
          x: highlighted.map((p) => (p.coords as [number, number, number])[0]),
          y: highlighted.map((p) => (p.coords as [number, number, number])[1]),
          z: highlighted.map((p) => (p.coords as [number, number, number])[2]),
          mode: 'markers',
          marker: {
            size: 5,
            color: altValues,
            colorscale: 'Viridis',
            showscale: true,
            colorbar: { tickfont: { color: '#9ca3af' }, title: { text: 'Altitude', font: { color: '#9ca3af' } } },
          },
          customdata: highlighted.map((p, i) => [p.id, hoverMeta[i]]),
          hovertemplate: '<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
        } as Data)
      } else {
        traces.push({
          type: 'scattergl',
          name: 'Altitude',
          x: highlighted.map((p) => p.coords[0]),
          y: highlighted.map((p) => p.coords[1]),
          mode: 'markers',
          marker: {
            size: 6,
            color: altValues,
            colorscale: 'Viridis',
            showscale: true,
            colorbar: { tickfont: { color: '#9ca3af' }, title: { text: 'Altitude', font: { color: '#9ca3af' } } },
          },
          customdata: highlighted.map((p, i) => [p.id, hoverMeta[i]]),
          hovertemplate: '<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
        } as Data)
      }
    } else if (isCategorical(colorBy)) {
      // Group by category, one trace per group
      const groups = new Map<string, EmbeddingPoint[]>()
      for (const p of highlighted) {
        const key =
          colorBy === 'cluster_label' || colorBy === 'cluster_id'
            ? String(p.cluster_label)
            : String(p.metadata[colorBy] ?? 'N/A')
        if (!groups.has(key)) groups.set(key, [])
        groups.get(key)!.push(p)
      }

      let colorIdx = 0
      for (const [groupKey, groupPoints] of groups) {
        const color = CATEGORICAL_COLORS[colorIdx % CATEGORICAL_COLORS.length]
        colorIdx++

        const hoverMeta = groupPoints.map((p) => {
          const keys = Object.keys(p.metadata).filter((k) => p.metadata[k] != null && p.metadata[k] !== '').slice(0, 3)
          return keys.map((k) => `${k}: ${p.metadata[k]}`).join('<br>')
        })

        if (mode === '3d') {
          traces.push({
            type: 'scatter3d',
            name: groupKey,
            x: groupPoints.map((p) => (p.coords as [number, number, number])[0]),
            y: groupPoints.map((p) => (p.coords as [number, number, number])[1]),
            z: groupPoints.map((p) => (p.coords as [number, number, number])[2]),
            mode: 'markers',
            marker: { size: 5, color, opacity: 0.85 },
            customdata: groupPoints.map((p, i) => [p.id, hoverMeta[i]]),
            hovertemplate: '<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
          } as Data)
        } else {
          traces.push({
            type: 'scattergl',
            name: groupKey,
            x: groupPoints.map((p) => p.coords[0]),
            y: groupPoints.map((p) => p.coords[1]),
            mode: 'markers',
            marker: { size: 6, color, opacity: 0.85 },
            customdata: groupPoints.map((p, i) => [p.id, hoverMeta[i]]),
            hovertemplate: '<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
          } as Data)
        }
      }
    }
  }

  const layout: Partial<Layout> = {
    ...darkLayout,
    dragmode: mode === '2d' ? 'pan' : undefined,
    margin: { t: 20, r: 20, b: 40, l: 40 },
    legend: {
      bgcolor: 'transparent',
      font: { color: '#9ca3af', size: 11 },
      itemsizing: 'constant',
    },
    hovermode: 'closest',
  }

  const handleClick = onPointClick
    ? (eventData: Readonly<Plotly.PlotMouseEvent>) => {
        const pt = eventData.points?.[0]
        if (pt && Array.isArray(pt.customdata) && pt.customdata[0]) {
          onPointClick(pt.customdata[0] as string)
        }
      }
    : undefined

  return (
    <div className={`relative ${className}`}>
      <Plot
        data={traces}
        layout={layout}
        config={{ ...plotlyConfig, scrollZoom: true }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
        onClick={handleClick}
      />
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-bg-card/70 rounded-xl z-10">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin" />
            <span className="text-text-secondary text-sm">Computing UMAP...</span>
          </div>
        </div>
      )}
    </div>
  )
}
