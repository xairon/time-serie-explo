import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import { plotlyConfig } from '@/lib/plotly-theme'
import type { TimeSeriesData } from '@/lib/types'

const CONTRIB_COLORS: Record<string, string> = {
  recharge: '#60a5fa',
  temperature: '#a78bfa',
  constant_d: '#34d399',
  Constant_d: '#34d399',
}
const FALLBACK_COLORS = ['#f97316', '#f43f5e', '#fbbf24', '#14b8a6']

const CONTRIB_LABELS: Record<string, string> = {
  recharge: 'Recharge (P − f·E)',
  temperature: 'Temperature',
  constant_d: 'Base level',
  Constant_d: 'Base level',
}

interface Props {
  obsX: string[]
  obsY: number[]
  simX: string[]
  simY: number[]
  contributions: Record<string, TimeSeriesData>
  monthlyDates: string[]
  monthlyResiduals: number[]
  threshold: number
  outlierMonths: Set<string>
  selectedOutlierDate: string | null
  onOutlierClick: (date: string) => void
  periodColor: string
  splitDate?: string
  viewPeriod: string
  selectedOutlierHighlight: { range: [string, string]; shape: any } | null
}

export function UnifiedAnalysisChart({
  obsX, obsY, simX, simY,
  contributions, monthlyDates, monthlyResiduals, threshold,
  outlierMonths, selectedOutlierDate, onOutlierClick,
  periodColor, splitDate, viewPeriod, selectedOutlierHighlight,
}: Props) {

  const selectedYM = selectedOutlierDate?.slice(0, 7)

  // --- Build all traces ---
  const { traces, shapes, annotations } = useMemo(() => {
    const t: any[] = []
    const s: any[] = []
    const a: any[] = []

    // ═══ PANEL 1: Water Level ═══
    t.push({
      x: obsX, y: obsY,
      type: 'scatter', mode: 'lines',
      name: 'Observed',
      line: { color: '#9ca3af', width: 1 },
      yaxis: 'y',
      hovertemplate: '<b>Observed</b>: %{y:.2f} m<extra></extra>',
    })
    t.push({
      x: simX, y: simY,
      type: 'scatter', mode: 'lines',
      name: 'Simulated',
      line: { color: periodColor, width: 2 },
      yaxis: 'y',
      hovertemplate: '<b>Simulated</b>: %{y:.2f} m<extra></extra>',
    })

    // ═══ PANEL 2: Contributions ═══
    const contribEntries = Object.entries(contributions)
    let colorIdx = 0
    for (const [name, ts] of contribEntries) {
      const color = CONTRIB_COLORS[name] ?? FALLBACK_COLORS[colorIdx++ % FALLBACK_COLORS.length]
      const label = CONTRIB_LABELS[name] ?? name

      // Slice contributions to match observation period
      const startD = obsX[0]
      const endD = obsX[obsX.length - 1]
      const filteredX: string[] = []
      const filteredY: number[] = []
      ts.index.forEach((d, i) => {
        if (d >= startD && d <= endD) {
          filteredX.push(d)
          filteredY.push(ts.values[i])
        }
      })

      t.push({
        x: filteredX, y: filteredY,
        type: 'scatter', mode: 'lines',
        name: label,
        line: { color, width: 1.5 },
        yaxis: 'y2',
        hovertemplate: `<b>${label}</b>: %{y:+.3f} m<extra></extra>`,
      })
    }

    // Zero line for contributions
    if (contribEntries.length > 0) {
      s.push({
        type: 'line',
        x0: obsX[0], x1: obsX[obsX.length - 1],
        y0: 0, y1: 0,
        yref: 'y2',
        line: { color: 'rgba(255,255,255,0.08)', width: 1 },
      })
    }

    // ═══ PANEL 3: Residuals (monthly bars) ═══
    if (monthlyDates.length > 0) {
      const barColors = monthlyResiduals.map((_v, i) => {
        const ym = monthlyDates[i].slice(0, 7)
        if (outlierMonths.size > 0) {
          if (!outlierMonths.has(ym)) return 'rgba(245,158,11,0.5)'
          return selectedYM === ym ? 'rgba(239,68,68,1.0)' : 'rgba(239,68,68,0.7)'
        }
        return Math.abs(monthlyResiduals[i]) <= threshold ? 'rgba(245,158,11,0.5)' : 'rgba(239,68,68,0.7)'
      })

      t.push({
        x: monthlyDates, y: monthlyResiduals,
        type: 'bar',
        name: 'Residual',
        marker: { color: barColors },
        yaxis: 'y3',
        hovertemplate: '<b>Residual</b>: %{y:+.3f} m<extra></extra>',
      })

      // ±2σ threshold lines
      s.push(
        { type: 'line', x0: monthlyDates[0], x1: monthlyDates[monthlyDates.length - 1], y0: threshold, y1: threshold, yref: 'y3', line: { color: 'rgba(239,68,68,0.25)', dash: 'dot', width: 1 } },
        { type: 'line', x0: monthlyDates[0], x1: monthlyDates[monthlyDates.length - 1], y0: -threshold, y1: -threshold, yref: 'y3', line: { color: 'rgba(239,68,68,0.25)', dash: 'dot', width: 1 } },
      )
    }

    // ═══ SHARED: Train/Test split marker ═══
    if (viewPeriod === 'full' && splitDate) {
      s.push({ type: 'line', x0: splitDate, x1: splitDate, y0: 0, y1: 1, yref: 'paper', line: { color: 'rgba(251,191,36,0.35)', dash: 'dash', width: 1.5 } })
      a.push({ x: splitDate, y: 1.01, yref: 'paper', text: 'Train → Test', showarrow: false, font: { size: 9, color: '#fbbf24' } })
    }

    // ═══ SHARED: Outlier highlight band ═══
    if (selectedOutlierHighlight) {
      s.push(selectedOutlierHighlight.shape)
    }

    return { traces: t, shapes: s, annotations: a }
  }, [obsX, obsY, simX, simY, contributions, monthlyDates, monthlyResiduals, threshold, outlierMonths, selectedYM, periodColor, splitDate, viewPeriod, selectedOutlierHighlight])

  // --- Layout ---
  const layout: any = useMemo(() => {
    const base = {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#9ca3af', size: 10 },
      margin: { t: 12, r: 20, b: 30, l: 55 },
      height: 520,
      showlegend: true,
      legend: {
        orientation: 'h' as const,
        y: 1.06,
        x: 0,
        font: { size: 9, color: '#9ca3af' },
        bgcolor: 'transparent',
      },
      hovermode: 'x unified' as const,
      hoverlabel: {
        bgcolor: 'rgba(17,24,39,0.95)',
        bordercolor: 'rgba(255,255,255,0.1)',
        font: { size: 10, color: '#e5e7eb', family: 'monospace' },
      },

      // Shared X axis
      xaxis: {
        gridcolor: 'rgba(255,255,255,0.03)',
        showline: false,
        zeroline: false,
        ...(selectedOutlierHighlight ? { range: selectedOutlierHighlight.range } : {}),
      },

      // Panel 1: Water Level (top 58%)
      yaxis: {
        domain: [0.38, 1.0],
        title: { text: 'm NGF', font: { size: 9 }, standoff: 8 },
        gridcolor: 'rgba(255,255,255,0.04)',
        zeroline: false,
      },

      // Panel 2: Contributions (middle 22%)
      yaxis2: {
        domain: [0.14, 0.35],
        title: { text: 'Contribution (m)', font: { size: 9 }, standoff: 8 },
        gridcolor: 'rgba(255,255,255,0.04)',
        zeroline: false,
      },

      // Panel 3: Residuals (bottom 11%)
      yaxis3: {
        domain: [0.0, 0.11],
        title: { text: 'Error (m)', font: { size: 9 }, standoff: 8 },
        gridcolor: 'rgba(255,255,255,0.04)',
        zeroline: false,
      },

      shapes,
      annotations: [
        ...annotations,
        // Panel labels (left side)
        { x: -0.01, y: 0.69, xref: 'paper', yref: 'paper', text: 'Water Level', showarrow: false, font: { size: 8, color: 'rgba(255,255,255,0.2)' }, textangle: -90 },
        { x: -0.01, y: 0.245, xref: 'paper', yref: 'paper', text: 'Contributions', showarrow: false, font: { size: 8, color: 'rgba(255,255,255,0.2)' }, textangle: -90 },
        { x: -0.01, y: 0.055, xref: 'paper', yref: 'paper', text: 'Error', showarrow: false, font: { size: 8, color: 'rgba(255,255,255,0.2)' }, textangle: -90 },
      ],
    }
    return base
  }, [shapes, annotations, selectedOutlierHighlight])

  return (
    <Plot
      data={traces}
      layout={layout}
      config={plotlyConfig}
      style={{ width: '100%' }}
      onClick={(event: any) => {
        if (!event.points?.[0]) return
        const point = event.points[0]
        if (point.data?.yaxis !== 'y3') return
        const clickedYM = String(point.x).slice(0, 7)
        onOutlierClick(clickedYM)
      }}
    />
  )
}
