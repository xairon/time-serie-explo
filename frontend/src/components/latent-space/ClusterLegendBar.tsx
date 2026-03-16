import { useMemo, useState } from 'react'

interface ClusterInfo {
  id: number
  color: string
  count: number
}

interface ClusterLegendBarProps {
  clusters: ClusterInfo[]
  selectedCluster: number | null
  onSelectCluster: (id: number | null) => void
}

const MAX_VISIBLE = 20

export function ClusterLegendBar({ clusters, selectedCluster, onSelectCluster }: ClusterLegendBarProps) {
  const [expanded, setExpanded] = useState(false)

  const sorted = useMemo(
    () => [...clusters].sort((a, b) => b.count - a.count),
    [clusters],
  )

  const noise = sorted.find((c) => c.id === -1)
  const real = sorted.filter((c) => c.id !== -1)
  const visible = expanded ? real : real.slice(0, MAX_VISIBLE)
  const hidden = real.length - visible.length

  function handleClick(id: number) {
    onSelectCluster(selectedCluster === id ? null : id)
  }

  return (
    <div className="flex items-center gap-1 overflow-x-auto py-1 px-1 scrollbar-thin">
      <button
        onClick={() => onSelectCluster(null)}
        className={`shrink-0 px-2 py-0.5 rounded text-[10px] border transition-colors ${
          selectedCluster === null
            ? 'border-accent-cyan/50 text-accent-cyan bg-accent-cyan/10'
            : 'border-white/10 text-text-muted hover:text-text-primary hover:bg-bg-hover'
        }`}
      >
        All ({clusters.reduce((s, c) => s + c.count, 0)})
      </button>

      {visible.map((c) => (
        <button
          key={c.id}
          onClick={() => handleClick(c.id)}
          className={`shrink-0 flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] border transition-colors ${
            selectedCluster === c.id
              ? 'border-accent-cyan/50 text-text-primary bg-accent-cyan/10'
              : selectedCluster !== null
                ? 'border-white/5 text-text-muted opacity-40'
                : 'border-white/10 text-text-muted hover:text-text-primary hover:bg-bg-hover'
          }`}
        >
          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: c.color }} />
          <span>{c.id}</span>
          <span className="text-text-muted">({c.count})</span>
        </button>
      ))}

      {hidden > 0 && !expanded && (
        <button
          onClick={() => setExpanded(true)}
          className="shrink-0 px-2 py-0.5 rounded text-[10px] border border-white/10 text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors"
        >
          +{hidden} more
        </button>
      )}

      {noise && (
        <button
          onClick={() => handleClick(-1)}
          className={`shrink-0 flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] border transition-colors ${
            selectedCluster === -1
              ? 'border-accent-cyan/50 text-text-primary bg-accent-cyan/10'
              : 'border-white/10 text-text-muted hover:text-text-primary hover:bg-bg-hover'
          }`}
        >
          <span className="w-2 h-2 rounded-full shrink-0 bg-gray-600" />
          <span>Noise</span>
          <span className="text-text-muted">({noise.count})</span>
        </button>
      )}
    </div>
  )
}
