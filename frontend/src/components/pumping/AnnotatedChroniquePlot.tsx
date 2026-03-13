import Plot from 'react-plotly.js'
import type { Data, Layout, Shape, Annotations } from 'plotly.js-dist-min'

interface TimePoint {
  time: string
  value: number
}

interface SuspectWindow {
  start: string
  end: string
  confidence: number
  label?: string
}

interface AnnotatedChroniquePlotProps {
  data: TimePoint[]
  suspectWindows?: SuspectWindow[]
  title?: string
  className?: string
}

const DARK_BASE: Partial<Layout> = {
  plot_bgcolor: 'transparent',
  paper_bgcolor: 'transparent',
  font: { color: '#e2e8f0' },
}

function confidenceFill(confidence: number): string {
  if (confidence >= 0.7) return 'rgba(239,68,68,0.22)'
  if (confidence >= 0.4) return 'rgba(249,115,22,0.18)'
  return 'rgba(234,179,8,0.14)'
}

function confidenceLine(confidence: number): string {
  if (confidence >= 0.7) return 'rgba(239,68,68,0.7)'
  if (confidence >= 0.4) return 'rgba(249,115,22,0.7)'
  return 'rgba(234,179,8,0.7)'
}

export function AnnotatedChroniquePlot({
  data,
  suspectWindows = [],
  title = 'Piezometric chronicle',
  className = '',
}: AnnotatedChroniquePlotProps) {
  const traces: Data[] = [
    {
      x: data.map(p => p.time),
      y: data.map(p => p.value),
      type: 'scatter',
      mode: 'lines',
      name: 'Piezometric level',
      line: { color: '#22d3ee', width: 1.5 },
      hovertemplate: '%{x}<br>%{y:.3f} m<extra></extra>',
    },
  ]

  const shapes: Partial<Shape>[] = suspectWindows.map(w => ({
    type: 'rect' as const,
    xref: 'x' as const,
    yref: 'paper' as const,
    x0: w.start,
    x1: w.end,
    y0: 0,
    y1: 1,
    fillcolor: confidenceFill(w.confidence),
    line: { color: confidenceLine(w.confidence), width: 1 },
  }))

  const annotations: Partial<Annotations>[] = suspectWindows.map(w => ({
    x: w.start,
    yref: 'paper' as const,
    y: 1.02,
    text: w.label ?? `conf: ${(w.confidence * 100).toFixed(0)}%`,
    showarrow: false,
    font: { size: 10, color: confidenceLine(w.confidence) },
    xanchor: 'left' as const,
  }))

  const layout: Partial<Layout> = {
    ...DARK_BASE,
    title: { text: title, font: { color: '#e2e8f0', size: 14 } },
    xaxis: {
      gridcolor: 'rgba(255,255,255,0.05)',
      linecolor: 'rgba(255,255,255,0.1)',
      tickfont: { color: '#94a3b8' },
    },
    yaxis: {
      title: { text: 'Level (m NGF)', font: { color: '#94a3b8' } },
      gridcolor: 'rgba(255,255,255,0.05)',
      linecolor: 'rgba(255,255,255,0.1)',
      tickfont: { color: '#94a3b8' },
    },
    shapes,
    annotations,
    margin: { t: 50, r: 20, b: 50, l: 60 },
    showlegend: false,
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: 320 }}
      className={className}
    />
  )
}
