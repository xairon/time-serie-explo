import { useState } from 'react'
import { CheckCircle, AlertTriangle, XCircle, ChevronDown, ChevronRight, Droplets, Thermometer, Clock, Zap } from 'lucide-react'
import type { CounterfactualResult } from '@/lib/types'

const THETA_LABELS: Record<string, string> = {
  s_P_DJF: 'Winter precipitation (DJF)',
  s_P_MAM: 'Spring precipitation (MAM)',
  s_P_JJA: 'Summer precipitation (JJA)',
  s_P_SON: 'Autumn precipitation (SON)',
  delta_T: 'Temperature (C)',
  delta_etp: 'Residual evapotranspiration',
  delta_s: 'Time lag (days)',
}

function interpretTheta(theta: Record<string, number>): string[] {
  const lines: string[] = []
  for (const [key, season] of [
    ['s_P_DJF', 'winter'],
    ['s_P_MAM', 'spring'],
    ['s_P_JJA', 'summer'],
    ['s_P_SON', 'autumn'],
  ] as const) {
    const v = theta[key]
    if (v != null && Math.abs(v - 1) > 0.05) {
      const pct = Math.round((v - 1) * 100)
      lines.push(`${season.charAt(0).toUpperCase() + season.slice(1)} precipitation: ${pct > 0 ? '+' : ''}${pct}%`)
    }
  }
  if (theta.delta_T != null && Math.abs(theta.delta_T) > 0.1) {
    lines.push(`Temperature: ${theta.delta_T > 0 ? '+' : ''}${theta.delta_T.toFixed(1)}C`)
  }
  if (theta.delta_etp != null && Math.abs(theta.delta_etp) > 0.01) {
    lines.push(`Residual ETP: ${theta.delta_etp > 0 ? '+' : ''}${theta.delta_etp.toFixed(3)}`)
  }
  if (theta.delta_s != null && Math.abs(theta.delta_s) > 1) {
    lines.push(`Time lag: ${theta.delta_s > 0 ? '+' : ''}${Math.round(theta.delta_s)} days`)
  }
  return lines
}

interface CFResultViewProps {
  result: CounterfactualResult | null
  isLoading: boolean
  className?: string
}

export function CFResultView({ result, isLoading, className = '' }: CFResultViewProps) {
  const [showTheta, setShowTheta] = useState(false)
  const [showConvergence, setShowConvergence] = useState(false)

  if (isLoading) {
    return (
      <div className={`bg-bg-card rounded-xl border border-white/5 p-6 ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-bg-hover rounded w-1/3" />
          <div className="h-8 bg-bg-hover rounded w-2/3" />
          <p className="text-xs text-text-secondary text-center">
            {result?.status === 'pending' ? 'Waiting for processing...' : 'Generating...'}
          </p>
        </div>
      </div>
    )
  }

  if (result?.error) {
    return (
      <div className={`bg-bg-card rounded-xl border border-accent-red/20 p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-2">
          <XCircle className="w-5 h-5 text-accent-red" />
          <span className="text-sm font-semibold text-accent-red">Failed</span>
        </div>
        <p className="text-sm text-text-secondary">{result.error}</p>
      </div>
    )
  }

  if (!result || !result.result) {
    return (
      <div className={`bg-bg-card rounded-xl border border-white/5 p-6 flex items-center justify-center ${className}`}>
        <p className="text-text-secondary text-sm">
          Configure and run a counterfactual analysis to see results.
        </p>
      </div>
    )
  }

  const inner = result.result
  const metrics = inner.metrics ?? {}
  const theta = inner.theta ?? {}
  const m = metrics as Record<string, unknown>
  const converged = m.converged === true || m.converged === 'true'
  const wallTime = typeof m.wall_clock_s === 'number' ? m.wall_clock_s : 0
  const nIter = typeof m.n_iter === 'number' ? m.n_iter : 0
  const bestLoss = typeof m.best_loss === 'number' ? m.best_loss : null

  const interpretation = interpretTheta(theta)

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Status banner */}
      <div className={`rounded-lg p-3 flex items-center gap-3 ${
        converged
          ? 'bg-emerald-500/10 border border-emerald-500/20'
          : 'bg-amber-500/10 border border-amber-500/20'
      }`}>
        {converged ? (
          <CheckCircle className="w-5 h-5 text-emerald-400 shrink-0" />
        ) : (
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <span className={`text-sm font-semibold ${converged ? 'text-emerald-400' : 'text-amber-400'}`}>
            {converged ? 'Converged' : 'Partial convergence'}
          </span>
          <span className="text-xs text-text-secondary ml-2">
            {inner.method.toUpperCase()} | {nIter} iterations | {wallTime.toFixed(1)}s
          </span>
        </div>
      </div>

      {/* Plausibility indicators */}
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-bg-card rounded-lg p-3 border border-white/5 text-center">
          <Zap className="w-4 h-4 text-accent-cyan mx-auto mb-1" />
          <p className="text-[10px] text-text-secondary uppercase">Convergence</p>
          <p className="text-sm font-bold text-text-primary">
            {converged ? 'Yes' : 'Partial'}
          </p>
        </div>
        <div className="bg-bg-card rounded-lg p-3 border border-white/5 text-center">
          <Droplets className="w-4 h-4 text-accent-indigo mx-auto mb-1" />
          <p className="text-[10px] text-text-secondary uppercase">Final loss</p>
          <p className="text-sm font-bold text-text-primary">
            {bestLoss != null ? bestLoss.toFixed(4) : '—'}
          </p>
        </div>
      </div>

      {/* Interpretation */}
      {interpretation.length > 0 && (
        <div className="bg-bg-card rounded-xl border border-white/5 p-4">
          <h4 className="text-xs text-text-secondary uppercase mb-2 flex items-center gap-1.5">
            <Thermometer className="w-3.5 h-3.5" />
            Interpretation
          </h4>
          <ul className="space-y-1.5">
            {interpretation.map((line, i) => (
              <li key={i} className="text-sm text-text-primary flex items-start gap-2">
                <span className="text-accent-cyan mt-0.5">{'>'}</span>
                {line}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Theta accordion */}
      {Object.keys(theta).length > 0 && (
        <div className="bg-bg-card rounded-xl border border-white/5">
          <button
            onClick={() => setShowTheta(!showTheta)}
            className="w-full px-4 py-3 flex items-center gap-2 text-xs text-text-secondary uppercase hover:text-text-primary transition-colors"
          >
            {showTheta ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            <Clock className="w-3.5 h-3.5" />
            Optimized parameters (theta)
          </button>
          {showTheta && (
            <div className="px-4 pb-4 space-y-2">
              {Object.entries(theta).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between text-sm">
                  <span className="text-text-secondary">{THETA_LABELS[key] ?? key}</span>
                  <span className="text-text-primary font-mono">
                    {typeof val === 'number' ? val.toFixed(4) : String(val)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Convergence accordion */}
      {inner.convergence && inner.convergence.length > 0 && (
        <div className="bg-bg-card rounded-xl border border-white/5">
          <button
            onClick={() => setShowConvergence(!showConvergence)}
            className="w-full px-4 py-3 flex items-center gap-2 text-xs text-text-secondary uppercase hover:text-text-primary transition-colors"
          >
            {showConvergence ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            Courbe de convergence ({inner.convergence.length} points)
          </button>
          {showConvergence && (
            <div className="px-4 pb-4">
              <ConvergenceMini losses={inner.convergence} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/** Minimal inline convergence chart using a simple SVG sparkline */
function ConvergenceMini({ losses }: { losses: number[] }) {
  if (losses.length < 2) return null
  const w = 400
  const h = 80
  const min = Math.min(...losses)
  const max = Math.max(...losses)
  const range = max - min || 1

  const points = losses.map((v, i) => {
    const x = (i / (losses.length - 1)) * w
    const y = h - ((v - min) / range) * (h - 8) - 4
    return `${x},${y}`
  })

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-20">
      <polyline
        points={points.join(' ')}
        fill="none"
        stroke="#06b6d4"
        strokeWidth="1.5"
      />
      <text x="4" y="12" fontSize="10" fill="#9ca3af">{max.toFixed(3)}</text>
      <text x="4" y={h - 2} fontSize="10" fill="#9ca3af">{min.toFixed(3)}</text>
    </svg>
  )
}
