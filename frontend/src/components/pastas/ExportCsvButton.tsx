import { Download } from 'lucide-react'
import { downloadCsv, type CsvColumn } from '@/lib/csv-export'

interface Props {
  filename: string
  getColumns: () => CsvColumn[]
  title?: string
}

export function ExportCsvButton({ filename, getColumns, title = 'Export CSV' }: Props) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation()
        downloadCsv(filename, getColumns())
      }}
      className="p-1 rounded text-text-muted hover:text-text-primary hover:bg-white/5 transition-colors"
      title={title}
    >
      <Download className="w-3.5 h-3.5" />
    </button>
  )
}
