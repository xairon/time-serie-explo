// frontend/src/lib/period-arithmetic.ts
// Pure month-period arithmetic ('YYYY-MM' strings, zero-padded so string
// compare == chronological). Shared by the Climat MonthStepper/DayStepper —
// extracted from the (now removed) MétéEAU rolling timeline, which was the
// only consumer of the rest of that module (window building, FR formatting).

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
