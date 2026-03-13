import { useState, useMemo } from 'react'
import Plot from 'react-plotly.js'
import { darkLayout, plotlyConfig } from '@/lib/plotly-theme'

interface MetadataDistributionsProps {
  distributions: Record<string, unknown>[]
  domain: 'piezo' | 'hydro'
}

const PIEZO_KEYS = ['milieu_eh', 'theme_eh', 'etat_eh', 'nature_eh', 'departement']
const HYDRO_KEYS = ['nom_cours_eau', 'departement']

const COLORS = [
  '#06b6d4', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981',
  '#ec4899', '#3b82f6', '#f97316', '#14b8a6', '#a855f7',
  '#eab308', '#6366f1', '#84cc16', '#e11d48', '#0ea5e9',
]

export function MetadataDistributions({ distributions, domain }: MetadataDistributionsProps) {
  const keys = domain === 'piezo' ? PIEZO_KEYS : HYDRO_KEYS
  const [selectedKey, setSelectedKey] = useState(keys[0])

  const dist = useMemo(() => {
    const d = distributions.find((d) => (d as { key: string }).key === selectedKey)
    return d ? (d as { key: string; clusters: Record<string, Record<string, number>> }).clusters : {}
  }, [distributions, selectedKey])

  const sortedClusters = useMemo(() => {
    return Object.entries(dist)
      .map(([cid, vals]) => ({
        cid,
        total: Object.values(vals).reduce((a, b) => a + b, 0),
        vals,
      }))
      .sort((a, b) => b.total - a.total)
  }, [dist])

  const allValues = useMemo(() => {
    const s = new Set<string>()
    for (const { vals } of sortedClusters) {
      for (const v of Object.keys(vals)) s.add(v)
    }
    return Array.from(s).sort()
  }, [sortedClusters])

  const traces = allValues.map((val, i) => ({
    type: 'bar' as const,
    orientation: 'h' as const,
    name: val.length > 25 ? val.slice(0, 22) + '...' : val,
    y: sortedClusters.map((c) => `Cluster ${c.cid}`),
    x: sortedClusters.map((c) => c.vals[val] ?? 0),
    marker: { color: COLORS[i % COLORS.length] },
    hovertemplate: `%{y}<br>${val}: %{x}<extra></extra>`,
  }))

  const selectClass =
    'bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:border-accent-cyan/50 transition-colors'

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-text-primary text-sm font-medium">Metadata Distributions</h3>
        <select
          className={selectClass}
          value={selectedKey}
          onChange={(e) => setSelectedKey(e.target.value)}
        >
          {keys.map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </div>
      <Plot
        data={traces}
        layout={{
          ...darkLayout,
          barmode: 'stack',
          margin: { l: 90, r: 20, t: 10, b: 30 },
          height: Math.max(200, sortedClusters.length * 35 + 50),
          yaxis: { ...darkLayout.yaxis, autorange: 'reversed' as const },
          legend: { ...darkLayout.legend, orientation: 'h', y: -0.15, font: { size: 10 } },
          showlegend: allValues.length <= 15,
        }}
        config={plotlyConfig}
        useResizeHandler
        className="w-full"
      />
    </div>
  )
}
