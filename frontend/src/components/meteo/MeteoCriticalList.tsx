import { useState } from 'react'

export interface CriticalSector {
  code: string
  name: string
  classLabel: string
  colorHex: string
  trend: 'hausse' | 'stable' | 'baisse' | null
}

interface Props {
  sectors: CriticalSector[]
  onSelect: (code: string) => void
}

// Rotation: hausse = 0° (point up), stable = 90° (point right), baisse = 180° (point down).
const TREND_ROTATION: Record<'hausse' | 'stable' | 'baisse', number> = {
  hausse: 0,
  stable: 90,
  baisse: 180,
}

function TrendArrow({ trend }: { trend: 'hausse' | 'stable' | 'baisse' | null }) {
  if (trend === null) {
    return (
      <span
        className="inline-block w-3.5 text-center text-slate-400 font-bold text-xs leading-none flex-shrink-0"
        aria-label="Tendance inconnue"
      >
        –
      </span>
    )
  }

  const rotation = TREND_ROTATION[trend]
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      aria-label={trend}
      style={{ transform: `rotate(${rotation}deg)`, flexShrink: 0 }}
    >
      <path
        d="M7 2.5 L12 11 L2 11 Z"
        fill="#0f172a"
        stroke="#fff"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/** Collapsible white card listing the driest/alert sectors for the /meteo page. */
export function MeteoCriticalList({ sectors, onSelect }: Props) {
  const [open, setOpen] = useState(true)

  return (
    <div className="bg-white rounded-lg shadow-md border border-slate-200 overflow-hidden">

      {/* Header — toggle button */}
      <button
        type="button"
        onClick={() => setOpen(prev => !prev)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-slate-50 transition-colors"
        aria-expanded={open}
      >
        <span className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-700 uppercase tracking-wide">
            Secteurs en alerte
          </span>
          {sectors.length > 0 && (
            <span className="bg-red-100 text-red-700 text-[10px] font-bold px-1.5 py-0.5 rounded-full leading-none">
              {sectors.length}
            </span>
          )}
        </span>
        {/* Chevron */}
        <svg
          width="14"
          height="14"
          viewBox="0 0 14 14"
          aria-hidden="true"
          className="text-slate-400 flex-shrink-0 transition-transform"
          style={{ transform: open ? 'rotate(0deg)' : 'rotate(-90deg)' }}
        >
          <path
            d="M2 5 L7 10 L12 5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {/* Body */}
      {open && (
        <div className="border-t border-slate-100">
          {sectors.length === 0 ? (
            <p className="px-3 py-3 text-xs text-slate-500">
              Aucun secteur en alerte.
            </p>
          ) : (
            <ul
              className="overflow-y-auto divide-y divide-slate-50"
              style={{ maxHeight: '50vh' }}
            >
              {sectors.map(s => (
                <li key={s.code}>
                  <button
                    type="button"
                    onClick={() => onSelect(s.code)}
                    title={s.name}
                    className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-slate-50 transition-colors"
                  >
                    {/* Colored dot */}
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: s.colorHex }}
                      aria-hidden="true"
                    />

                    {/* Name (truncated) + class label */}
                    <span className="flex-1 min-w-0">
                      <span className="block text-xs font-medium text-slate-800 truncate">
                        {s.name}
                      </span>
                      <span className="block text-[10px] text-slate-500 truncate">
                        {s.classLabel}
                      </span>
                    </span>

                    {/* Trend arrow */}
                    <TrendArrow trend={s.trend} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
