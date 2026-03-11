// Ordered from lowest to highest — index = ordinal position
export const IPS_CLASS_ORDER = [
  'very_low',
  'low',
  'moderately_low',
  'normal',
  'moderately_high',
  'high',
  'very_high',
] as const

export type IPSClass = (typeof IPS_CLASS_ORDER)[number]

export const IPS_LABELS: Record<string, string> = {
  very_low: 'Tres bas',
  low: 'Bas',
  moderately_low: 'Mod. bas',
  normal: 'Normal',
  moderately_high: 'Mod. haut',
  high: 'Haut',
  very_high: 'Tres haut',
}

export const IPS_COLORS: Record<string, string> = {
  very_low: '#d73027',
  low: '#fc8d59',
  moderately_low: '#fee08b',
  normal: '#ffffbf',
  moderately_high: '#d9ef8b',
  high: '#91cf60',
  very_high: '#1a9850',
}

// BRGM standard thresholds: Seguin (2014) RP-64147-FR
export const IPS_THRESHOLDS: [string, number, number][] = [
  ['very_low', -Infinity, -1.28],
  ['low', -1.28, -0.84],
  ['moderately_low', -0.84, -0.25],
  ['normal', -0.25, 0.25],
  ['moderately_high', 0.25, 0.84],
  ['high', 0.84, 1.28],
  ['very_high', 1.28, Infinity],
]

const MONTH_NAMES = ['', 'Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aou', 'Sep', 'Oct', 'Nov', 'Dec']

/** Classify a z-score into an IPS class */
export function classifyZ(z: number): string {
  for (const [cls, lo, hi] of IPS_THRESHOLDS) {
    if (z >= lo && z < hi) return cls
  }
  return 'normal'
}

/** Shift a class by delta steps (clamped at extremes). delta can be negative. */
export function shiftClass(cls: string, delta: number): string {
  const idx = IPS_CLASS_ORDER.indexOf(cls as IPSClass)
  if (idx === -1) return cls
  const newIdx = Math.max(0, Math.min(IPS_CLASS_ORDER.length - 1, idx + delta))
  return IPS_CLASS_ORDER[newIdx]
}

/** True if two classes are at most 1 step apart in the ordering */
export function areAdjacent(a: string, b: string): boolean {
  const idxA = IPS_CLASS_ORDER.indexOf(a as IPSClass)
  const idxB = IPS_CLASS_ORDER.indexOf(b as IPSClass)
  if (idxA === -1 || idxB === -1) return false
  return Math.abs(idxA - idxB) <= 1
}

/** Get the ordinal index of a class (0=very_low, 6=very_high). Returns -1 if unknown. */
export function classIndex(cls: string): number {
  return IPS_CLASS_ORDER.indexOf(cls as IPSClass)
}

/** Get the mode (most frequent value) of an array of strings */
export function mode(values: string[]): string | null {
  if (values.length === 0) return null
  const counts = new Map<string, number>()
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1)
  let best = values[0]
  let bestCount = 0
  for (const [v, c] of counts) {
    if (c > bestCount) { best = v; bestCount = c }
  }
  return best
}

export interface MonthIps {
  yearMonth: string    // "2024-01"
  monthNumber: number  // 1-12
  monthLabel: string   // "Jan 2024"
  cls: string          // IPS class key
  zScore: number       // raw z-score
  mean: number         // mean GWL for that month
}

/**
 * Classify daily values into per-year-month IPS classes.
 * Groups by year-month (not month number alone).
 */
export function computeMonthlyIps(
  dates: string[],
  values: number[],
  refStats: Record<string, [number, number]>,
): MonthIps[] {
  const groups = new Map<string, { vals: number[]; monthNumber: number; year: string }>()

  for (let i = 0; i < dates.length; i++) {
    const d = new Date(dates[i])
    const monthNumber = d.getMonth() + 1
    const year = String(d.getFullYear())
    const yearMonth = `${year}-${String(monthNumber).padStart(2, '0')}`
    if (!groups.has(yearMonth)) groups.set(yearMonth, { vals: [], monthNumber, year })
    if (!isNaN(values[i]) && values[i] !== null) groups.get(yearMonth)!.vals.push(values[i])
  }

  const result: MonthIps[] = []
  for (const [yearMonth, { vals, monthNumber, year }] of groups) {
    const stats = refStats[String(monthNumber)]
    if (!stats || vals.length === 0) continue
    const [mu, sigma] = stats
    if (sigma <= 0) continue
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length
    const zScore = (mean - mu) / sigma
    result.push({
      yearMonth,
      monthNumber,
      monthLabel: `${MONTH_NAMES[monthNumber]} ${year}`,
      cls: classifyZ(zScore),
      zScore,
      mean,
    })
  }

  return result
}

export type ConcordanceStatus = 'exact' | 'adjacent' | 'divergent'
export type QualityVerdict = 'qualified' | 'partial' | 'not_qualified'

export interface MonthConcordance {
  yearMonth: string
  monthLabel: string
  gtClass: string
  predClass: string
  status: ConcordanceStatus
}

export function computeQualityGate(
  gtIps: MonthIps[],
  predIps: MonthIps[],
): { months: MonthConcordance[]; verdict: QualityVerdict } {
  const predMap = new Map(predIps.map((m) => [m.yearMonth, m]))
  const months: MonthConcordance[] = []

  for (const gt of gtIps) {
    const pred = predMap.get(gt.yearMonth)
    if (!pred) continue
    let status: ConcordanceStatus
    if (gt.cls === pred.cls) status = 'exact'
    else if (areAdjacent(gt.cls, pred.cls)) status = 'adjacent'
    else status = 'divergent'
    months.push({ yearMonth: gt.yearMonth, monthLabel: gt.monthLabel, gtClass: gt.cls, predClass: pred.cls, status })
  }

  const divergentCount = months.filter((m) => m.status === 'divergent').length
  const totalCount = months.length

  let verdict: QualityVerdict
  if (divergentCount === 0) verdict = 'qualified'
  else if (divergentCount < totalCount / 2) verdict = 'partial'
  else verdict = 'not_qualified'

  return { months, verdict }
}
