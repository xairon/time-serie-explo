import { useState } from 'react'
import { ExternalLink, Maximize2, Minimize2 } from 'lucide-react'

const OBSERVATORY_URL = `${window.location.protocol}//${window.location.hostname}:49510`

export default function ObservatoryPage() {
  const [fullscreen, setFullscreen] = useState(false)

  return (
    <div className={`flex flex-col ${fullscreen ? 'fixed inset-0 z-50 bg-bg-primary' : 'h-full'}`}>
      <div className="flex items-center justify-between px-4 py-2 bg-bg-card border-b border-white/5 shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-indigo" />
          <span className="text-sm font-medium text-text-primary">Hydro Observatory</span>
          <span className="text-xs text-text-secondary">— Observatoire hydrologique France</span>
        </div>
        <div className="flex items-center gap-2">
          <a
            href={OBSERVATORY_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-text-secondary hover:text-accent-cyan transition-colors"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            Ouvrir
          </a>
          <button
            onClick={() => setFullscreen(!fullscreen)}
            className="flex items-center gap-1 text-xs text-text-secondary hover:text-accent-cyan transition-colors"
            title={fullscreen ? 'Réduire' : 'Plein écran'}
          >
            {fullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>
      <iframe
        src={OBSERVATORY_URL}
        className="flex-1 w-full border-0"
        title="Hydro Observatory"
        allow="fullscreen"
      />
    </div>
  )
}
