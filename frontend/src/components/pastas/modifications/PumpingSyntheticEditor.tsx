import { useMemo, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import type { PumpingProfileData, PumpingRange, AdaptiveBoundsData } from '@/lib/types'

interface PumpingSyntheticData {
  type: 'pumping_synthetic'
  usage?: 'aep' | 'irrigation' | 'industrial'
  pattern: 'constant' | 'seasonal' | 'pulse'
  rate_m3d: number
  distance_m: number
  start: string
  end: string
  rfunc: 'Exponential' | 'Hantush'
  season_months?: number[]
  peak_months?: number[]
  peak_factor?: number
  pulse_duration_days?: number
}

interface PumpingSyntheticEditorProps {
  data: PumpingSyntheticData
  onChange: (data: PumpingSyntheticData) => void
  profiles?: Record<string, PumpingProfileData> | null
  adaptiveBounds?: AdaptiveBoundsData | null
}

const USAGES = [
  { value: 'aep' as const, label: 'AEP', tooltip: 'Drinking water supply — typically 50-200 m³/d, constant year-round' },
  { value: 'irrigation' as const, label: 'Irrigation', tooltip: 'Agricultural — typically 200-1000 m³/d, seasonal (summer)' },
  { value: 'industrial' as const, label: 'Industrial', tooltip: 'Industrial use — typically 100-500 m³/d, constant or semi-constant' },
] as const

function RangeWarning({ value, range, label }: { value: number; range?: PumpingRange; label: string }) {
  if (!range || value <= 0) return null
  if (value < range.typical_min || value > range.typical_max) {
    return (
      <p className="text-[10px] mt-1 text-yellow-400">
        Warning: unusual {label} — typical range: {range.typical_min}–{range.typical_max}
      </p>
    )
  }
  return null
}

const PATTERNS: { value: PumpingSyntheticData['pattern']; label: string; desc: string }[] = [
  { value: 'constant', label: 'Constant', desc: 'Same rate every day over the full period' },
  { value: 'seasonal', label: 'Seasonal', desc: 'Active only during selected months, every year' },
  { value: 'pulse', label: 'Pulse', desc: 'Single pumping event of configurable duration' },
]

const SEASON_PRESETS: { label: string; months: number[] }[] = [
  { label: 'Summer (May-Sep)', months: [5, 6, 7, 8, 9] },
  { label: 'Winter (Nov-Mar)', months: [11, 12, 1, 2, 3] },
  { label: 'Irrigation (Jun-Aug)', months: [6, 7, 8] },
  { label: 'Full year', months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] },
]

const MONTH_LABELS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

const RFUNCS = ['Exponential', 'Hantush'] as const

const inputClass =
  'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-1.5 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

function generatePreview(pattern: string, rate: number, seasonMonths: number[], pulseDays: number, startDate: string, endDate: string): { dates: string[]; values: number[] } {
  if (!startDate || !endDate) return { dates: [], values: [] }
  const start = new Date(startDate)
  const end = new Date(endDate)
  const totalDays = Math.max(1, Math.round((end.getTime() - start.getTime()) / 86400000))
  const step = Math.max(1, Math.floor(totalDays / 200))
  const dates: string[] = []
  const values: number[] = []

  for (let d = 0; d < totalDays; d += step) {
    const dt = new Date(start.getTime() + d * 86400000)
    dates.push(dt.toISOString().slice(0, 10))
    switch (pattern) {
      case 'constant':
        values.push(rate)
        break
      case 'seasonal':
        values.push(seasonMonths.includes(dt.getMonth() + 1) ? rate : 0)
        break
      case 'pulse': {
        const mid = totalDays / 2
        const half = pulseDays / 2
        values.push(d >= mid - half && d <= mid + half ? rate : 0)
        break
      }
      default:
        values.push(0)
    }
  }
  return { dates, values }
}

function DrawdownIndicator({ rate, bounds }: { rate: number; bounds: AdaptiveBoundsData }) {
  if (rate <= 0) return null
  const drawdown = Math.abs(rate * bounds.step_response_at_t)
  const isSafe = drawdown < bounds.soft_drawdown_m
  const isWarning = drawdown >= bounds.soft_drawdown_m && drawdown < bounds.hard_drawdown_m
  const isDanger = drawdown >= bounds.hard_drawdown_m
  const color = isSafe ? 'text-green-400' : isWarning ? 'text-yellow-400' : 'text-red-400'
  const bgColor = isSafe ? 'bg-green-500/5 border-green-500/20' : isWarning ? 'bg-yellow-500/5 border-yellow-500/20' : 'bg-red-500/5 border-red-500/20'
  const label = isSafe ? 'Realistic' : isWarning ? 'High' : 'Unrealistic'
  const hint = isDanger
    ? `This rate would lower the water table by ${drawdown.toFixed(1)} m — far beyond typical aquifer capacity. Try reducing the flow rate below ${Math.floor(bounds.soft_drawdown_m / Math.abs(bounds.step_response_at_t))} m³/d.`
    : isWarning
      ? `Significant drawdown (${drawdown.toFixed(1)} m). The aquifer can likely sustain this, but it's at the upper limit.`
      : `Drawdown of ${drawdown.toFixed(2)} m — well within the aquifer's capacity.`
  return (
    <div className={`text-[10px] mt-1.5 px-2 py-1.5 rounded border ${bgColor}`}>
      <div className="flex items-center gap-1.5">
        <span className={`font-semibold ${color}`}>{label}</span>
        <span className="text-text-muted">— estimated water level drop: {drawdown.toFixed(2)} m after {bounds.t_final_days} days</span>
      </div>
      <p className="text-text-muted mt-0.5">{hint}</p>
    </div>
  )
}

function ExpertPanel({ bounds, staticRange }: { bounds: AdaptiveBoundsData; staticRange?: PumpingRange }) {
  const [open, setOpen] = useState(false)
  const isWell = bounds.source === 'calibrated_well'

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[10px] text-text-muted hover:text-text-secondary transition-colors"
      >
        <ChevronDown className={`w-3 h-3 transition-transform ${open ? '' : '-rotate-90'}`} />
        {isWell ? 'Calibrated model' : 'Aquifer sensitivity'}
      </button>
      {open && (
        <div className="mt-1.5 bg-bg-primary border border-white/5 rounded-lg p-2.5 space-y-1">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px]">
            <span className="text-text-muted">Gain A</span>
            <span className="text-text-secondary font-mono">{bounds.gain_A.toExponential(3)} m/(m³/d)</span>
            <span className="text-text-muted">t95</span>
            <span className="text-text-secondary font-mono">{Math.round(bounds.t95_days)} d</span>
            <span className="text-text-muted">Step response (at t)</span>
            <span className="text-text-secondary font-mono">{bounds.step_response_at_t.toExponential(3)}</span>
            {isWell && bounds.Q_soft != null && (
              <>
                <span className="text-text-muted">Q soft (model)</span>
                <span className="text-text-secondary font-mono">{Math.round(bounds.Q_soft)} m³/d</span>
              </>
            )}
            {isWell && bounds.Q_hard != null && (
              <>
                <span className="text-text-muted">Q hard (model)</span>
                <span className="text-text-secondary font-mono">{Math.round(bounds.Q_hard)} m³/d</span>
              </>
            )}
            {staticRange && (
              <>
                <span className="text-text-muted">Q soft (ref.)</span>
                <span className="text-text-secondary font-mono">{staticRange.typical_max} m³/d</span>
                <span className="text-text-muted">Q hard (ref.)</span>
                <span className="text-text-secondary font-mono">{staticRange.hard_max} m³/d</span>
              </>
            )}
          </div>
          <div className="pt-1 border-t border-white/5">
            <span className={`text-[9px] px-1.5 py-0.5 rounded border ${
              isWell
                ? 'border-accent-cyan/20 text-accent-cyan bg-accent-cyan/5'
                : 'border-white/10 text-text-muted'
            }`}>
              Source: {isWell ? 'calibrated model' : 'static reference'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export function PumpingSyntheticEditor({ data, onChange, profiles, adaptiveBounds }: PumpingSyntheticEditorProps) {
  const profile = profiles?.[data.usage ?? 'aep'] ?? null

  function update(patch: Partial<PumpingSyntheticData>) {
    onChange({ ...data, ...patch })
  }

  function selectUsage(usage: 'aep' | 'irrigation' | 'industrial' | undefined) {
    const patch: Partial<PumpingSyntheticData> = { usage: usage || undefined }
    const targetProfile = usage ? profiles?.[usage] : null
    if (targetProfile) {
      patch.rate_m3d = targetProfile.rate_m3d.default
      patch.distance_m = targetProfile.distance_m.default
      patch.pattern = targetProfile.pattern as 'constant' | 'seasonal' | 'pulse'
      patch.season_months = targetProfile.active_months
      patch.peak_months = targetProfile.peak_months
      patch.peak_factor = targetProfile.peak_factor
      patch.rfunc = targetProfile.rfunc as 'Exponential' | 'Hantush'
    }
    onChange({ ...data, ...patch })
  }

  const rateRange = profile?.rate_m3d
  const distRange = profile?.distance_m

  const effectiveHardMax = useMemo(() => {
    const staticMax = rateRange?.hard_max
    const adaptiveMax = adaptiveBounds?.Q_hard
    if (staticMax != null && adaptiveMax != null) return Math.min(staticMax, adaptiveMax)
    return staticMax ?? adaptiveMax ?? undefined
  }, [rateRange, adaptiveBounds])

  const activeMonths = data.season_months ?? [5, 6, 7, 8, 9]
  const pulseDays = data.pulse_duration_days ?? 30

  const preview = useMemo(
    () => generatePreview(data.pattern, data.rate_m3d, activeMonths, pulseDays, data.start, data.end),
    [data.pattern, data.rate_m3d, activeMonths, pulseDays, data.start, data.end],
  )
  const maxVal = Math.max(...(preview.values.length > 0 ? preview.values : [1]), 1)

  function toggleMonth(m: number) {
    const next = activeMonths.includes(m)
      ? activeMonths.filter(x => x !== m)
      : [...activeMonths, m].sort((a, b) => a - b)
    update({ season_months: next.length > 0 ? next : [m] })
  }

  return (
    <div className="space-y-3">
      {/* Usage selector */}
      <div>
        <label className="block text-xs text-text-muted mb-1.5">Pumping type</label>
        <div className="flex gap-1">
          {USAGES.map(u => (
            <button
              key={u.value}
              onClick={() => selectUsage(u.value)}
              title={u.tooltip}
              className={`flex-1 px-2 py-1.5 rounded-lg text-[10px] font-medium transition-colors ${
                data.usage === u.value
                  ? 'bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30'
                  : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10'
              }`}
            >
              {u.label}
            </button>
          ))}
          <button
            onClick={() => selectUsage(undefined)}
            className={`flex-1 px-2 py-1.5 rounded-lg text-[10px] font-medium transition-colors ${
              !data.usage
                ? 'bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30'
                : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10'
            }`}
          >
            Custom
          </button>
        </div>
      </div>

      {/* Pattern selector */}
      <div>
        <label className="block text-xs text-text-muted mb-1.5">Pumping pattern</label>
        <div className="space-y-1">
          {PATTERNS.map((p) => (
            <button
              key={p.value}
              onClick={() => update({ pattern: p.value })}
              className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors ${
                data.pattern === p.value
                  ? 'bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30'
                  : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10 hover:text-text-secondary'
              }`}
            >
              <span className="font-medium">{p.label}</span>
              <span className="text-[10px] ml-1.5 opacity-70">{p.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Season month picker (seasonal only) */}
      {data.pattern === 'seasonal' && (
        <div>
          <label className="block text-xs text-text-muted mb-1.5">Active months</label>
          <div className="flex gap-1 mb-2">
            {SEASON_PRESETS.map(p => {
              const match = p.months.length === activeMonths.length && p.months.every(m => activeMonths.includes(m))
              return (
                <button
                  key={p.label}
                  onClick={() => update({ season_months: p.months })}
                  className={`flex-1 px-1 py-1 rounded text-[10px] font-medium transition-colors ${
                    match
                      ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
                      : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10'
                  }`}
                >
                  {p.label}
                </button>
              )
            })}
          </div>
          <div className="flex gap-0.5">
            {MONTH_LABELS.map((label, i) => {
              const month = i + 1
              const active = activeMonths.includes(month)
              return (
                <button
                  key={month}
                  onClick={() => toggleMonth(month)}
                  className={`flex-1 py-1.5 rounded text-[10px] font-medium transition-colors ${
                    active
                      ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                      : 'bg-bg-primary text-text-muted border border-white/5 hover:border-white/10'
                  }`}
                >
                  {label}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Pulse duration (pulse only) */}
      {data.pattern === 'pulse' && (
        <div>
          <label className="block text-xs text-text-muted mb-1">Pulse duration (days)</label>
          <input
            type="number"
            value={pulseDays}
            onChange={(e) => update({ pulse_duration_days: Math.max(1, parseInt(e.target.value) || 30) })}
            className={inputClass}
            step="1"
            min="1"
          />
        </div>
      )}

      {/* Pumping preview */}
      {preview.values.length > 0 && (
        <div className="bg-bg-primary rounded-lg border border-white/5 p-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-text-muted">Pumping schedule preview</span>
            <span className="text-[10px] text-text-muted">peak {data.rate_m3d} m³/d</span>
          </div>
          <div className="flex items-end gap-px h-12">
            {preview.values.map((v, i) => (
              <div
                key={i}
                className="flex-1 rounded-t-sm transition-colors"
                style={{
                  height: `${(v / maxVal) * 100}%`,
                  minHeight: v > 0 ? 1 : 0,
                  backgroundColor: v > 0 ? 'rgba(96,165,250,0.6)' : 'rgba(255,255,255,0.03)',
                }}
              />
            ))}
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[9px] text-text-muted">{data.start}</span>
            <span className="text-[9px] text-text-muted">{data.end}</span>
          </div>
        </div>
      )}

      <div>
        <label className="block text-xs text-text-muted mb-1">
          Flow rate (m³/d)
          {adaptiveBounds?.Q_soft != null && rateRange && adaptiveBounds.Q_soft < rateRange.typical_max && (
            <span className="ml-1 text-[9px] text-accent-cyan">Cal. model</span>
          )}
          {!adaptiveBounds?.Q_soft && rateRange && (
            <span className="ml-1 text-[9px] text-text-muted">Ref.</span>
          )}
        </label>
        <input
          type="number"
          value={data.rate_m3d || ''}
          onChange={(e) => update({ rate_m3d: e.target.value === '' ? 0 : parseFloat(e.target.value) })}
          onBlur={() => { if (!data.rate_m3d) update({ rate_m3d: 0 }) }}
          placeholder="e.g. 100"
          className={inputClass}
          step="10"
          min={rateRange?.hard_min ?? 0}
          max={effectiveHardMax}
        />
        <RangeWarning value={data.rate_m3d} range={rateRange} label="Flow rate" />
        {adaptiveBounds && <DrawdownIndicator rate={data.rate_m3d} bounds={adaptiveBounds} />}
      </div>

      {data.rfunc === 'Hantush' && (
        <div>
          <label className="block text-xs text-text-muted mb-1" title="Horizontal distance between the well and the observation piezometer. Only used with the Hantush response function (confined aquifers).">Distance to piezometer (m)</label>
          <input
            type="number"
            value={data.distance_m || ''}
            onChange={(e) => update({ distance_m: e.target.value === '' ? 0 : parseFloat(e.target.value) })}
            onBlur={() => { if (!data.distance_m) update({ distance_m: 100 }) }}
            placeholder="e.g. 500"
            className={inputClass}
            step="100"
            min={distRange?.hard_min ?? 1}
            max={distRange?.hard_max}
          />
          <RangeWarning value={data.distance_m} range={distRange} label="Distance" />
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-text-muted mb-1">Start</label>
          <input type="date" value={data.start} onChange={(e) => update({ start: e.target.value })} className={inputClass} />
        </div>
        <div>
          <label className="block text-xs text-text-muted mb-1">End</label>
          <input type="date" value={data.end} onChange={(e) => update({ end: e.target.value })} className={inputClass} />
        </div>
      </div>

      <div>
        <label className="block text-xs text-text-muted mb-1" title="How the aquifer responds to pumping. Exponential: simple decay (most aquifers). Hantush: accounts for distance and confinement (confined aquifers only).">Response function</label>
        <select
          value={data.rfunc}
          onChange={(e) => update({ rfunc: e.target.value as 'Exponential' | 'Hantush' })}
          className={inputClass}
        >
          {RFUNCS.map((r) => (
            <option key={r} value={r}>{r === 'Exponential' ? 'Exponential (unconfined)' : 'Hantush (confined)'}</option>
          ))}
        </select>
      </div>

      {adaptiveBounds && (
        <ExpertPanel bounds={adaptiveBounds} staticRange={rateRange ?? undefined} />
      )}
    </div>
  )
}

export type { PumpingSyntheticData }
