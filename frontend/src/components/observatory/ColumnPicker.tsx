import { useEffect, useRef, useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { EXPORT_COLUMN_GROUPS, type ExportColumnGroup } from '@/lib/observatory-api'

interface Props {
  selected: ExportColumnGroup[]
  onChange: (next: ExportColumnGroup[]) => void
}

export default function ColumnPicker({ selected, onChange }: Props) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  function toggle(group: ExportColumnGroup) {
    onChange(
      selected.includes(group)
        ? selected.filter((g) => g !== group)
        : [...EXPORT_COLUMN_GROUPS].filter((g) => g === group || selected.includes(g)),
    )
  }

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-bg-card border border-white/10 text-text-secondary hover:text-text-primary transition-colors"
      >
        {t('mainPages.station.exportColumnsButton')} ({selected.length}/{EXPORT_COLUMN_GROUPS.length})
        <ChevronDown className="w-3.5 h-3.5" />
      </button>
      {open && (
        <div className="absolute right-0 mt-1 z-20 w-44 bg-bg-card border border-white/10 rounded-lg p-2 shadow-lg">
          {EXPORT_COLUMN_GROUPS.map((group) => (
            <label key={group} className="flex items-center gap-2 px-2 py-1.5 text-xs text-text-secondary hover:text-text-primary cursor-pointer">
              <input
                type="checkbox"
                checked={selected.includes(group)}
                onChange={() => toggle(group)}
                className="accent-accent-cyan"
              />
              {t(`mainPages.station.exportColumns.${group}`)}
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
