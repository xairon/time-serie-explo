// Pure year multi-select logic for the Climat Comparaison view (Task B3) — bounds
// (2-6 years) and famous-drought presets. No React/DOM here so it's trivial to unit test.

/** GET /observatory/climat/compare-years accepts up to 15 years server-side, but the
 *  Comparaison UI (multi-select chips + superposed curves) is only readable up to 6. */
export const MIN_COMPARE_YEARS = 2
export const MAX_COMPARE_YEARS = 6

/** Famous French drought years (plan Task B3), offered as one-click preset chips. */
export const DROUGHT_YEAR_PRESETS = [1976, 1989, 2003, 2022] as const

export function isValidYearSelection(years: readonly number[]): boolean {
  return years.length >= MIN_COMPARE_YEARS && years.length <= MAX_COMPARE_YEARS
}

/** Default selection on first render: the drought presets plus the current year,
 *  deduplicated and capped at MAX_COMPARE_YEARS (oldest presets dropped first if the
 *  current year happens to coincide with one, since the set already fits within 5). */
export function defaultCompareYears(currentYear: number): number[] {
  const years = Array.from(new Set<number>([...DROUGHT_YEAR_PRESETS, currentYear]))
  years.sort((a, b) => a - b)
  return years.slice(0, MAX_COMPARE_YEARS)
}

/** Toggle one year in/out of the selection, refusing moves that would break the
 *  2-6 bounds (adding past MAX, or removing below MIN). Returns the SAME array
 *  reference when the toggle is refused, so callers can skip a re-render/refetch. */
export function toggleYear(selected: readonly number[], year: number): number[] {
  const isSelected = selected.includes(year)
  if (isSelected) {
    if (selected.length <= MIN_COMPARE_YEARS) return selected as number[]
    return selected.filter((y) => y !== year)
  }
  if (selected.length >= MAX_COMPARE_YEARS) return selected as number[]
  return [...selected, year].sort((a, b) => a - b)
}
