interface Props {
  filteredPiezo: number
  filteredHydro: number
  totalPiezo: number
  totalHydro: number
}

export function KPIBar({ filteredPiezo, filteredHydro, totalPiezo, totalHydro }: Props) {
  const isFiltered = filteredPiezo !== totalPiezo || filteredHydro !== totalHydro
  return (
    <div className="absolute bottom-0 left-0 right-0 z-10 bg-bg-card/90 backdrop-blur-md border-t border-white/5">
      <div className="flex items-center justify-center gap-6 px-4 py-2">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full shrink-0 bg-accent-cyan" aria-hidden="true" />
          <span className="text-xs text-text-secondary">Piézo</span>
          <span className="text-sm font-semibold text-text-primary font-mono">
            {isFiltered ? `${filteredPiezo.toLocaleString()} / ${totalPiezo.toLocaleString()}` : totalPiezo.toLocaleString()}
          </span>
        </div>
        <span className="text-white/20">&middot;</span>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full shrink-0 bg-accent-indigo" aria-hidden="true" />
          <span className="text-xs text-text-secondary">Hydro</span>
          <span className="text-sm font-semibold text-text-primary font-mono">
            {isFiltered ? `${filteredHydro.toLocaleString()} / ${totalHydro.toLocaleString()}` : totalHydro.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  )
}
