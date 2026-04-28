export interface CsvColumn {
  header: string
  values: (string | number | null)[]
}

function escapeCell(value: string | number | null): string {
  if (value === null || value === undefined) return ''
  const str = String(value)
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return '"' + str.replace(/"/g, '""') + '"'
  }
  return str
}

export function downloadCsv(filename: string, columns: CsvColumn[]): void {
  if (columns.length === 0) return

  const headers = columns.map(c => escapeCell(c.header)).join(',')
  const rowCount = Math.max(...columns.map(c => c.values.length))
  const rows: string[] = [headers]

  for (let i = 0; i < rowCount; i++) {
    rows.push(columns.map(c => escapeCell(c.values[i] ?? null)).join(','))
  }

  const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
