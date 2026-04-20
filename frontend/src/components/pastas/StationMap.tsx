import Plot from 'react-plotly.js'

interface Props {
  lat: number | null
  lon: number | null
  label: string
}

export function StationMap({ lat, lon, label }: Props) {
  if (lat == null || lon == null) return null

  return (
    <div className="bg-bg-card rounded-lg border border-white/5 overflow-hidden">
      <Plot
        data={[
          {
            type: 'scattermapbox' as const,
            lat: [lat],
            lon: [lon],
            mode: 'markers',
            marker: { size: 12, color: '#22d3ee' },
            text: [label],
            hoverinfo: 'text',
          },
        ]}
        layout={{
          mapbox: {
            style: 'carto-darkmatter',
            center: { lat, lon },
            zoom: 9,
          },
          margin: { t: 0, r: 0, b: 0, l: 0 },
          height: 180,
          paper_bgcolor: 'transparent',
        }}
        useResizeHandler
        className="w-full"
        config={{ displayModeBar: false }}
      />
    </div>
  )
}
