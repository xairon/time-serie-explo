// Pure day-arithmetic helpers for the Climat daily-temperature DayStepper — a
// day-granularity sibling of period-arithmetic's addMonths/comparePeriods. Kept in
// its own module (like period-arithmetic) so DayStepper stays a thin render layer
// and the logic is unit-testable without React.
import { comparePeriods } from './period-arithmetic'

export { comparePeriods }

/** Add `delta` days to a 'YYYY-MM-DD' date string, returning 'YYYY-MM-DD'.
 *  Uses UTC internally so DST transitions never shift the day. */
export function addDays(dateStr: string, delta: number): string {
  const [y, m, d] = dateStr.split('-').map(Number)
  const dt = new Date(Date.UTC(y, m - 1, d))
  dt.setUTCDate(dt.getUTCDate() + delta)
  return dt.toISOString().slice(0, 10)
}

/** Default day to show when a daily-temp variable is first selected — the most
 *  recent day covered by the ingestion (`max_date` from GET /daily-temp-range),
 *  so users land on live data instead of an empty grid. Null-safe: the range
 *  endpoint returns null before the first ingestion row lands. */
export function resolveDefaultDay(maxDate: string | null | undefined): string {
  return maxDate ? maxDate.slice(0, 10) : ''
}

/** Locale-aware "28 juin 2026" / "June 28, 2026" label for a 'YYYY-MM-DD' date —
 *  shared by DayStepper and ClimatLegend so the two never drift apart. */
export function formatDayLabel(dateStr: string, locale: string): string {
  const m = dateStr.match(/^(\d{4})-(\d{2})-(\d{2})/)
  if (!m) return dateStr
  return new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'long', year: 'numeric' }).format(
    new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])),
  )
}
