// Observatory utility functions

export function formatNumber(n: number | null | undefined, decimals = 1): string {
  if (n == null) return 'N/A'
  return n.toLocaleString('en-GB', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

export function formatDate(d: string | null | undefined): string {
  if (!d) return 'N/A'
  return new Date(d).toLocaleDateString('en-GB', { year: 'numeric', month: 'short', day: '2-digit' })
}
