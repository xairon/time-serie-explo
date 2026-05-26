// Observatory utility functions

export function formatNumber(n: number | null | undefined, decimals = 1): string {
  if (n == null) return 'N/A'
  return n.toLocaleString('fr-FR', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

export function formatDate(d: string | null | undefined): string {
  if (!d) return 'N/A'
  return new Date(d).toLocaleDateString('fr-FR')
}
