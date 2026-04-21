import { NavLink, Outlet } from 'react-router-dom'
import { MapPin, SlidersHorizontal, BarChart3, FlaskConical, LayoutGrid } from 'lucide-react'
import { GuidedExpertToggle } from '@/components/pastas/GuidedExpertToggle'
import { useState, createContext, useContext, useEffect } from 'react'

export const PastasModeContext = createContext<{
  mode: 'guided' | 'expert'
  setMode: (m: 'guided' | 'expert') => void
}>({ mode: 'guided', setMode: () => {} })

export function usePastasMode() {
  return useContext(PastasModeContext)
}

const STORAGE_KEY = 'pastas_mode'

const TABS = [
  { to: '/pastas/station', icon: MapPin, label: 'Station' },
  { to: '/pastas/calibrate', icon: SlidersHorizontal, label: 'Calibrate' },
  { to: '/pastas/results', icon: BarChart3, label: 'Results' },
  { to: '/pastas/scenarios', icon: FlaskConical, label: 'Scenarios' },
  { to: '/pastas/gallery', icon: LayoutGrid, label: 'Gallery' },
] as const

function readStoredMode(): 'guided' | 'expert' {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'guided' || v === 'expert') return v
  } catch { /* ignore */ }
  return 'guided'
}

export default function PastasLayout() {
  const [mode, setModeState] = useState<'guided' | 'expert'>(readStoredMode)

  function setMode(m: 'guided' | 'expert') {
    setModeState(m)
    try {
      localStorage.setItem(STORAGE_KEY, m)
    } catch { /* ignore */ }
  }

  // Sync across tabs
  useEffect(() => {
    function onStorage(e: StorageEvent) {
      if (e.key === STORAGE_KEY && (e.newValue === 'guided' || e.newValue === 'expert')) {
        setModeState(e.newValue)
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  return (
    <PastasModeContext.Provider value={{ mode, setMode }}>
      <div className="flex flex-col h-full">
        <div className="bg-bg-card border-b border-white/5 shrink-0">
          <div className="flex items-center justify-between px-4">
            <div className="flex items-center gap-1">
              {TABS.map(({ to, icon: Icon, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 ${
                      isActive
                        ? 'border-accent-cyan text-text-primary'
                        : 'border-transparent text-text-muted hover:text-text-secondary'
                    }`
                  }
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </NavLink>
              ))}
            </div>
            <GuidedExpertToggle mode={mode} onChange={setMode} />
          </div>
        </div>
        <div className="flex-1 min-h-0 overflow-auto">
          <Outlet />
        </div>
      </div>
    </PastasModeContext.Provider>
  )
}
