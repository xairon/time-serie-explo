import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'
import { useStationWindows } from '@/hooks/useLatentSpace'

interface WindowScatterProps {
  domain: string
  stationId: string
  space: string
}

export function WindowScatter({ domain, stationId, space }: WindowScatterProps) {
  const { data, isLoading } = useStationWindows(domain, stationId, space)

  if (isLoading) {
    return (
      <div className="h-44 flex items-center justify-center">
        <div className="w-5 h-5 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const windows = (data as Record<string, unknown>)?.windows as
    | Array<{ window_start: string; window_end: string; umap_x: number; umap_y: number }>
    | undefined

  if (!windows || windows.length === 0) {
    return <p className="text-text-muted text-xs py-2">Aucune fenêtre disponible</p>
  }

  const years = windows.map((w) => parseInt(w.window_start.slice(0, 4)))

  return (
    <Plot
      data={[
        {
          type: 'scattergl',
          x: windows.map((w) => w.umap_x),
          y: windows.map((w) => w.umap_y),
          mode: 'markers',
          marker: {
            size: 6,
            color: years,
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {
              thickness: 10,
              len: 0.8,
              tickfont: { color: '#9ca3af', size: 9 },
              title: { text: 'Year', font: { color: '#9ca3af', size: 9 } },
            },
          },
          customdata: windows.map((w) => [`${w.window_start} → ${w.window_end}`]),
          hovertemplate: '<b>%{customdata[0]}</b><extra></extra>',
        },
      ]}
      layout={{
        ...darkLayout,
        margin: { t: 5, r: 40, b: 5, l: 5 },
        xaxis: { visible: false },
        yaxis: { visible: false },
        hovermode: 'closest',
        height: 180,
      }}
      config={{ ...plotlyConfig, displayModeBar: false }}
      useResizeHandler
      style={{ width: '100%', height: '180px' }}
    />
  )
}
