interface Props {
  periods: string[]
  selectedPeriod: string | null
  onChange: (p: string) => void
}

const FR_MONTHS = [
  'janv.', 'févr.', 'mars', 'avr.', 'mai', 'juin',
  'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.',
]

function formatPeriodFR(period: string): string {
  const [year, month] = period.split('-')
  const monthIndex = parseInt(month, 10) - 1
  if (monthIndex < 0 || monthIndex > 11) return period
  return `${FR_MONTHS[monthIndex]} ${year}`
}

export function SituationTimelineSlider({ periods, selectedPeriod, onChange }: Props) {
  if (periods.length === 0) return null

  const currentIndex = selectedPeriod
    ? Math.max(0, periods.indexOf(selectedPeriod))
    : periods.length - 1

  const displayPeriod = periods[currentIndex] ?? periods[periods.length - 1]

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const idx = parseInt(e.target.value, 10)
    const clamped = Math.max(0, Math.min(idx, periods.length - 1))
    onChange(periods[clamped])
  }

  return (
    <div className="absolute bottom-0 left-0 right-0 h-12 z-10 bg-white border-t border-slate-200 flex items-center px-4 gap-3 shadow-md">
      {/* Current period label */}
      <span className="text-xs font-semibold text-slate-700 whitespace-nowrap min-w-[70px]">
        {formatPeriodFR(displayPeriod)}
      </span>

      {/* Range slider */}
      <input
        type="range"
        min={0}
        max={periods.length - 1}
        value={currentIndex}
        onChange={handleChange}
        className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer"
        style={{
          accentColor: '#3b7fc8',
          background: `linear-gradient(to right, #3b7fc8 0%, #3b7fc8 ${(currentIndex / (periods.length - 1)) * 100}%, #cbd5e1 ${(currentIndex / (periods.length - 1)) * 100}%, #cbd5e1 100%)`,
        }}
        aria-label="Sélectionner une période"
        aria-valuetext={formatPeriodFR(displayPeriod)}
      />

      {/* First and last period labels */}
      <span className="text-[10px] text-slate-400 whitespace-nowrap">
        {formatPeriodFR(periods[0])} – {formatPeriodFR(periods[periods.length - 1])}
      </span>
    </div>
  )
}
