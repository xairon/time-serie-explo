import { useState, useEffect } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
  Waves,
  Brain,
  Map,
  GitCompare,
  Menu,
  X,
} from 'lucide-react'
import { useHealth } from '@/hooks/useHealth'

const NAV_ITEMS = [
  { to: '/', icon: Map, label: 'Observatoire', end: true, tour: 'nav-observatory' },
  { to: '/compare', icon: GitCompare, label: 'Comparer', end: false, tour: 'nav-compare' },
  { to: '/pastas', icon: Waves, label: 'Pastas Lab', end: false, tour: 'nav-pastas' },
  { to: '/ai', icon: Brain, label: 'Lab IA', end: false, tour: 'nav-ai' },
] as const

export function TopNav() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const { data: health } = useHealth()

  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  const isHealthy = health?.status === 'ok'

  return (
    <nav className="h-12 bg-bg-card border-b border-white/5 flex items-center px-4 shrink-0 z-30 relative">
      <NavLink to="/" className="flex items-center gap-2 mr-6">
        <div className="w-8 h-8 rounded-lg bg-accent-cyan/20 flex items-center justify-center">
          <span className="text-accent-cyan font-bold text-sm">J</span>
        </div>
        <span className="text-sm font-semibold text-text-primary hidden sm:block">
          Junon
        </span>
      </NavLink>

      <div className="hidden md:flex items-center gap-1">
        {NAV_ITEMS.map(({ to, icon: Icon, label, end, tour }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            data-tour={tour}
            className={({ isActive }) =>
              `flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-accent-cyan/10 text-accent-cyan'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
              }`
            }
          >
            <Icon className="w-4 h-4" />
            {label}
          </NavLink>
        ))}
      </div>

      <div className="ml-auto flex items-center gap-3">
        {health?.gpu?.available && (
          <span className="text-[10px] text-text-muted hidden sm:block" title={health.gpu.device ?? ''}>
            GPU
          </span>
        )}
        <div className="flex items-center gap-1.5" title={isHealthy ? 'API connectée' : 'API indisponible'}>
          <div
            className={`w-2 h-2 rounded-full ${
              isHealthy ? 'bg-accent-green' : 'bg-accent-red'
            }`}
          />
          <span className="text-xs text-text-secondary hidden sm:block">
            {isHealthy ? 'OK' : 'Hors ligne'}
          </span>
        </div>

        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="md:hidden p-2 hover:bg-bg-hover rounded-lg"
          aria-label={mobileOpen ? 'Fermer le menu' : 'Ouvrir le menu'}
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {mobileOpen && (
        <div className="md:hidden absolute top-12 left-0 right-0 bg-bg-card border-b border-white/10 shadow-xl z-40">
          {NAV_ITEMS.map(({ to, icon: Icon, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 text-sm transition-colors ${
                  isActive
                    ? 'bg-accent-cyan/10 text-accent-cyan'
                    : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </div>
      )}
    </nav>
  )
}
