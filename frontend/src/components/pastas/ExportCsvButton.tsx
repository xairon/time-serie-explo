import { Download } from 'lucide-react'
import { downloadCsv, type CsvColumn } from '@/lib/csv-export'

interface Props {
  filename: string
  getColumns: () => CsvColumn[]
  title?: string
  label?: string
}

export function ExportCsvButton({ filename, getColumns, title = 'Exporter CSV', label }: Props) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation()
        downloadCsv(filename, getColumns())
      }}
      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] text-text-muted hover:text-text-primary hover:bg-white/5 transition-colors border border-white/5"
      title={title}
    >
      <Download className="w-3 h-3" />
      {label && <span>{label}</span>}
    </button>
  )
}
