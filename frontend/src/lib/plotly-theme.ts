import type { Config, Layout } from 'plotly.js-dist-min'

export const darkLayout: Partial<Layout> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#e5e7eb', family: 'Inter' },
  xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
  yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
  margin: { t: 30, r: 20, b: 40, l: 50 },
  legend: { bgcolor: 'transparent', font: { color: '#9ca3af' } },
}

export const plotlyConfig: Partial<Config> = {
  displayModeBar: false,
  responsive: true,
}
