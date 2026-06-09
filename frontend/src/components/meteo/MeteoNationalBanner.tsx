interface Props {
  dominantClassLabel: string
  dominantColorHex: string
  trendSummary: string         // e.g. "majorité en baisse"
  criticalCount: number
  totalSectors: number
  pctBelowNormal?: number | null
}

/** Full-width summary banner for the /meteo national situation. */
export function MeteoNationalBanner({
  dominantClassLabel,
  dominantColorHex,
  trendSummary,
  criticalCount,
  totalSectors,
  pctBelowNormal,
}: Props) {
  const capitalised =
    dominantClassLabel.length > 0
      ? dominantClassLabel.charAt(0).toUpperCase() + dominantClassLabel.slice(1)
      : '—'

  return (
    <div className="bg-white/95 backdrop-blur rounded-lg shadow px-4 py-2 flex flex-wrap items-center gap-4">

      {/* Title block */}
      <div className="min-w-0">
        <p className="text-sm font-bold text-slate-800 leading-tight whitespace-nowrap">
          Météo des nappes
        </p>
        <p className="text-[10px] text-slate-500 leading-tight whitespace-nowrap">
          Situation des nappes par secteur
        </p>
      </div>

      {/* Divider */}
      <div className="hidden sm:block h-8 w-px bg-slate-200 flex-shrink-0" aria-hidden="true" />

      {/* Situation dominante chip */}
      <div className="flex flex-col gap-0.5 min-w-0">
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide leading-none">
          Situation dominante
        </span>
        <span className="inline-flex items-center gap-1.5 mt-0.5">
          <span
            className="w-3 h-3 rounded-sm flex-shrink-0"
            style={{ backgroundColor: dominantColorHex }}
            aria-hidden="true"
          />
          <span className="text-xs font-medium text-slate-800">{capitalised}</span>
        </span>
      </div>

      {/* Divider */}
      <div className="hidden sm:block h-8 w-px bg-slate-200 flex-shrink-0" aria-hidden="true" />

      {/* Critical count metric */}
      <div className="flex flex-col gap-0.5">
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide leading-none">
          Secteurs en alerte
        </span>
        <span className="flex items-baseline gap-1 mt-0.5">
          <span className="text-lg font-bold text-slate-800 leading-none">{criticalCount}</span>
          <span className="text-xs text-slate-500 leading-none">/ {totalSectors} secteurs</span>
        </span>
      </div>

      {/* Divider */}
      <div className="hidden sm:block h-8 w-px bg-slate-200 flex-shrink-0" aria-hidden="true" />

      {/* Trend summary */}
      <div className="flex flex-col gap-0.5 min-w-0">
        <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide leading-none">
          Évolution
        </span>
        <span className="text-xs text-slate-700 mt-0.5 leading-tight">{trendSummary}</span>
      </div>

      {/* Below-normal percentage — only when provided */}
      {pctBelowNormal != null && (
        <>
          <div className="hidden sm:block h-8 w-px bg-slate-200 flex-shrink-0" aria-hidden="true" />
          <div className="flex flex-col gap-0.5">
            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide leading-none">
              Sous la normale
            </span>
            <span className="text-xs font-medium text-slate-800 mt-0.5">
              {Math.round(pctBelowNormal)}&nbsp;% des nappes
            </span>
          </div>
        </>
      )}
    </div>
  )
}
