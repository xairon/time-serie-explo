// Observatory utility functions

export function formatNumber(n: number | null | undefined, decimals = 1): string {
  if (n == null) return 'N/A'
  return n.toLocaleString('fr-FR', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

export function formatDate(d: string | null | undefined): string {
  if (!d) return 'N/A'
  return new Date(d).toLocaleDateString('fr-FR', { year: 'numeric', month: 'short', day: '2-digit' })
}

// Number of days after which a station's last measurement is considered stale.
// Single source of truth for the "active station" badge/banner. Mirrors the API
// threshold in api/routers/observatory_common.py (recent_cutoff = today - 90 days).
export const ACTIVE_STATION_DAYS = 90

// Days used by the map "current year only" filter — a distinct, user-toggled
// concept (a rolling year), intentionally more lenient than the activity badge.
export const ACTIVE_FILTER_DAYS = 365

// True if the station's last measurement is recent enough to be considered active.
// Compares calendar dates as YYYY-MM-DD strings to avoid the timezone/edge bug of
// `new Date(dateStr) >= localDateWithTime` (a same-day boundary value was wrongly
// flagged inactive because the cutoff carried the current local time-of-day).
export function isStationActive(
  dateStr: string | null | undefined,
  days = ACTIVE_STATION_DAYS,
): boolean {
  if (!dateStr) return false
  const cutoff = new Date()
  cutoff.setDate(cutoff.getDate() - days)
  return dateStr.slice(0, 10) >= cutoff.toISOString().slice(0, 10)
}
