// Pure data-shaping helpers for the Climat Comparaison chart (Task B3) — turn the
// GET /observatory/climat/compare-years response into Plotly-ready series (one trace
// per selected year, jan→déc, plus a single reference "normale" trace). Kept free of
// React/Plotly imports so the shaping logic is unit-testable on its own.
import type { ClimatCompareYears } from './observatory-types'

export interface CompareCumulSeries {
  year: number
  x: number[]
  y: (number | null)[]
}

export interface CompareNormalSeries {
  x: number[]
  y: (number | null)[]
}

const MONTHS = Array.from({ length: 12 }, (_, i) => i + 1)

/** One cumulative-precipitation trace per requested year, months 1-12 (missing
 *  months — e.g. the current year isn't over yet — become null so Plotly breaks
 *  the line instead of drawing a false drop to 0). */
export function buildCumulSeries(data: ClimatCompareYears | undefined, years: number[]): CompareCumulSeries[] {
  if (!data) return []
  return years.map((year) => {
    const yearData = data.years[String(year)]
    const byMonth = new Map((yearData?.cumul_mensuel ?? []).map((m) => [m.mois, m.cumul]))
    return { year, x: MONTHS, y: MONTHS.map((mo) => byMonth.get(mo) ?? null) }
  })
}

/** Single reference "normale" trace (1991-2020 climatology cumulative sum) — the
 *  normal is identical across years for the same cell/month, but a year with fewer
 *  months available (e.g. the current year, still in progress) would truncate it, so
 *  we read it off whichever selected year has the most complete monthly coverage. */
export function buildNormalSeries(data: ClimatCompareYears | undefined, years: number[]): CompareNormalSeries {
  if (!data) return { x: MONTHS, y: MONTHS.map(() => null) }
  let best: ClimatCompareYears['years'][string] | undefined
  for (const year of years) {
    const yearData = data.years[String(year)]
    if (yearData && (!best || yearData.cumul_mensuel.length > best.cumul_mensuel.length)) {
      best = yearData
    }
  }
  const byMonth = new Map((best?.cumul_mensuel ?? []).map((m) => [m.mois, m.cumul_normal]))
  return { x: MONTHS, y: MONTHS.map((mo) => byMonth.get(mo) ?? null) }
}
