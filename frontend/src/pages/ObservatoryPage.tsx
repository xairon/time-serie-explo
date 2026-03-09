import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { ExternalLink, Maximize2, Minimize2, Zap, X } from 'lucide-react'

const OBSERVATORY_URL = `${window.location.protocol}//${window.location.hostname}:49510`
// Proxied URL injects the bridge script via nginx sub_filter
const OBSERVATORY_PROXY = '/observatory-app/'

interface StationMessage {
  code: string
  type: 'piezo' | 'hydro'
  name: string
  dept: string
}

export default function ObservatoryPage() {
  const [fullscreen, setFullscreen] = useState(false)
  const [selectedStation, setSelectedStation] = useState<StationMessage | null>(null)
  const navigate = useNavigate()

  const handleMessage = useCallback((event: MessageEvent) => {
    const data = event.data
    if (!data || typeof data !== 'object') return

    if (data.type === 'OBSERVATORY_STATION_SELECTED') {
      setSelectedStation(data.station)
    } else if (data.type === 'OBSERVATORY_STATION_DESELECTED') {
      setSelectedStation(null)
    } else if (data.type === 'OBSERVATORY_TRAIN') {
      // Direct click on "Analyser dans Junon" button
      const station = data.station as StationMessage
      navigate(`/data?station=${encodeURIComponent(station.code)}`)
    }
  }, [navigate])

  useEffect(() => {
    window.addEventListener('message', handleMessage)
    return () => window.removeEventListener('message', handleMessage)
  }, [handleMessage])

  const goToImport = () => {
    if (selectedStation) {
      navigate(`/data?station=${encodeURIComponent(selectedStation.code)}`)
    }
  }

  return (
    <div className={`flex flex-col ${fullscreen ? 'fixed inset-0 z-50 bg-bg-primary' : 'h-full'}`}>
      <div className="flex items-center justify-between px-4 py-2 bg-bg-card border-b border-white/5 shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-indigo" />
          <span className="text-sm font-medium text-text-primary">Hydro Observatory</span>
          <span className="text-xs text-text-secondary hidden sm:block">— Observatoire hydrologique France</span>
        </div>
        <div className="flex items-center gap-2">
          {selectedStation && (
            <div className="flex items-center gap-2 bg-accent-cyan/10 rounded-lg px-3 py-1">
              <span className="text-xs font-mono text-accent-cyan">{selectedStation.code}</span>
              <button
                onClick={goToImport}
                className="flex items-center gap-1 text-xs font-medium text-white bg-accent-cyan/80 hover:bg-accent-cyan px-2 py-0.5 rounded transition-colors"
              >
                <Zap className="w-3 h-3" />
                Importer & Analyser
              </button>
              <button onClick={() => setSelectedStation(null)} className="text-text-secondary hover:text-text-primary">
                <X className="w-3 h-3" />
              </button>
            </div>
          )}
          <a
            href={OBSERVATORY_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-text-secondary hover:text-accent-cyan transition-colors"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Ouvrir</span>
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
        src={OBSERVATORY_PROXY}
        className="flex-1 w-full border-0"
        title="Hydro Observatory"
        allow="fullscreen"
      />
    </div>
  )
}
