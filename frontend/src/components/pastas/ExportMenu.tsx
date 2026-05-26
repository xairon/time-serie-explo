import { useState, useRef, useEffect } from 'react'
import { Download } from 'lucide-react'
import { API_BASE } from '@/lib/constants'

interface Props {
  runId: string
}

export function ExportMenu({ runId }: Props) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen(!open)}
        className="p-1 hover:bg-bg-hover rounded text-text-muted hover:text-text-primary transition-colors">
        <Download className="w-4 h-4" />
      </button>
      {open && (
        <div className="absolute right-0 mt-1 bg-bg-card border border-white/10 rounded-lg shadow-xl z-10 py-1 w-44">
          <a href={`${API_BASE}/pastas/models/${runId}/export/pas`}
            className="block px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover transition-colors">
            Télécharger .pas
          </a>
          <a href={`${API_BASE}/pastas/models/${runId}/export/csv`}
            className="block px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover transition-colors">
            Télécharger métriques CSV
          </a>
          <button
            onClick={() => { navigator.clipboard.writeText(runId); setOpen(false) }}
            className="block w-full text-left px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover transition-colors">
            Copier l'identifiant d'exécution
          </button>
        </div>
      )}
    </div>
  )
}
