// frontend/src/lib/meteo-timeline.ts
// Pure helpers for the MétéEau-style rolling monthly timeline.
// Periods are 'YYYY-MM' strings (zero-padded, so string compare == chronological).

export interface TimelineCell {
  period: string      // 'YYYY-MM'
  available: boolean  // a data point exists for this month
  future: boolean     // after the latest available month (greyed forecast slot)
  showYear: boolean   // render the year under the label (January cells)
}

export const FR_MONTHS_SHORT = [
  'janv.', 'févr.', 'mars', 'avr.', 'mai', 'juin',
  'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.',
] as const

export const FR_MONTHS_LONG = [
  'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
  'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre',
] as const

export function comparePeriods(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0
}

export function addMonths(period: string, delta: number): string {
  const [y, m] = period.split('-').map(Number)
  const total = y * 12 + (m - 1) + delta
  const ny = Math.floor(total / 12)
  const nm = total - ny * 12
  return `${ny}-${String(nm + 1).padStart(2, '0')}`
}

function monthIndex(period: string): number {
  return parseInt(period.split('-')[1], 10) - 1
}

export function formatPeriodShortFR(period: string): string {
  const i = monthIndex(period)
  return i >= 0 && i < 12 ? FR_MONTHS_SHORT[i] : period
}

export function formatPeriodLongFR(period: string): string {
  const i = monthIndex(period)
  return i >= 0 && i < 12 ? `${FR_MONTHS_LONG[i]} ${period.split('-')[0]}` : period
}

/**
 * Build the rolling window of timeline cells.
 * @param allPeriods 'YYYY-MM' strings sorted ascending (as served by the timeline API).
 * - Default: `size` months ending at the latest data month, plus `futureSlots`
 *   greyed calendar months after it (the original's forecast slots).
 * - If `selected` falls before that window, the window is re-centered on it.
 * - Never starts before the first data month.
 */
export function buildTimelineWindow(
  allPeriods: string[],
  selected: string,
  size = 12,
  futureSlots = 3,
): TimelineCell[] {
  if (allPeriods.length === 0) return []
  const first = allPeriods[0]
  const latest = allPeriods[allPeriods.length - 1]
  const available = new Set(allPeriods)

  let start = addMonths(latest, -(size - 1))
  if (comparePeriods(selected, start) < 0) {
    start = addMonths(selected, -Math.floor((size - 1) / 2))
  }
  if (comparePeriods(start, first) < 0) start = first

  const cells: TimelineCell[] = []
  for (let i = 0; i < size + futureSlots; i++) {
    const p = addMonths(start, i)
    cells.push({
      period: p,
      available: available.has(p),
      future: comparePeriods(p, latest) > 0,
      showYear: p.endsWith('-01'),
    })
  }
  return cells
}
