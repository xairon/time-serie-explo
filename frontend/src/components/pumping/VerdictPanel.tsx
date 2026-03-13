interface SuspectWindow {
  start: string
  end: string
  confidence: number
  label?: string
  layers?: string[]
}

interface LayerConcordance {
  layer: string
  score: number
  n_windows: number
}

interface VerdictPanelProps {
  globalScore?: number
  suspectWindows?: SuspectWindow[]
  layerConcordance?: LayerConcordance[]
  verdictLabel?: string
}

function ConfidenceBadge({ confidence }: { confidence: number }) {
  if (confidence >= 0.7) {
    return (
      <span className="px-2 py-0.5 rounded text-xs font-medium bg-red-500/20 text-red-400 border border-red-500/30">
        High ({(confidence * 100).toFixed(0)}%)
      </span>
    )
  }
  if (confidence >= 0.4) {
    return (
      <span className="px-2 py-0.5 rounded text-xs font-medium bg-orange-500/20 text-orange-400 border border-orange-500/30">
        Medium ({(confidence * 100).toFixed(0)}%)
      </span>
    )
  }
  return (
    <span className="px-2 py-0.5 rounded text-xs font-medium bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
      Low ({(confidence * 100).toFixed(0)}%)
    </span>
  )
}

function ScoreGauge({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(1, score))
  const angle = -90 + pct * 180

  const color = pct >= 0.7 ? '#ef4444' : pct >= 0.4 ? '#f97316' : '#22c55e'

  // Polar to Cartesian helper for gauge arc endpoint
  const rad = (angle * Math.PI) / 180
  const cx = 80
  const cy = 80
  const r = 55
  const x = cx + r * Math.cos(rad)
  const y = cy + r * Math.sin(rad)

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width="160" height="90" viewBox="0 0 160 90" aria-label={`Global score: ${(pct * 100).toFixed(0)}%`}>
        {/* Background arc */}
        <path
          d="M 25 80 A 55 55 0 0 1 135 80"
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Foreground arc */}
        <path
          d={`M 25 80 A 55 55 0 ${angle > 0 ? 1 : 0} 1 ${x.toFixed(1)} ${y.toFixed(1)}`}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          opacity={pct > 0.01 ? 1 : 0}
        />
        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={cx + 45 * Math.cos(rad)}
          y2={cy + 45 * Math.sin(rad)}
          stroke={color}
          strokeWidth="2.5"
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r="4" fill={color} />
        {/* Score text */}
        <text x={cx} y={cy + 20} textAnchor="middle" fill="#e2e8f0" fontSize="20" fontWeight="bold">
          {(pct * 100).toFixed(0)}
        </text>
        <text x={cx} y={cy + 32} textAnchor="middle" fill="#94a3b8" fontSize="9">
          / 100
        </text>
      </svg>
      <span className="text-xs text-text-secondary">Global suspicion score</span>
    </div>
  )
}

export function VerdictPanel({
  globalScore,
  suspectWindows = [],
  layerConcordance = [],
  verdictLabel,
}: VerdictPanelProps) {
  if (globalScore == null && !suspectWindows.length) {
    return (
      <div className="flex items-center justify-center h-32 text-text-secondary text-sm">
        Waiting for fusion verdict...
      </div>
    )
  }

  return (
    <div className="space-y-5">
      {/* Score gauge + label */}
      <div className="flex flex-col sm:flex-row items-center gap-4">
        {globalScore != null && <ScoreGauge score={globalScore} />}
        {verdictLabel && (
          <div className="flex-1">
            <p className="text-text-primary font-semibold text-base">{verdictLabel}</p>
          </div>
        )}
      </div>

      {/* Suspect windows */}
      {suspectWindows.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-text-primary">Suspect windows ({suspectWindows.length})</h3>
          <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
            {suspectWindows.map((w, i) => (
              <div
                key={i}
                className="flex items-center justify-between gap-3 bg-bg-primary/40 border border-white/5 rounded-lg px-3 py-2"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-mono text-text-primary truncate">
                    {w.start} → {w.end}
                  </p>
                  {w.layers && (
                    <p className="text-xs text-text-secondary mt-0.5">
                      Layers: {w.layers.join(', ')}
                    </p>
                  )}
                </div>
                <ConfidenceBadge confidence={w.confidence} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Layer concordance */}
      {layerConcordance.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-text-primary">Layer concordance</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {layerConcordance.map((l, i) => (
              <div
                key={i}
                className="bg-bg-primary/40 border border-white/5 rounded-lg px-3 py-2 flex items-center justify-between gap-2"
              >
                <span className="text-xs text-text-secondary capitalize">{l.layer}</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 h-1.5 rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full bg-accent-cyan rounded-full"
                      style={{ width: `${Math.min(100, l.score * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-text-primary w-10 text-right">
                    {(l.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
