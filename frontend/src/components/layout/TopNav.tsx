import { useState, useEffect } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Database,
  GraduationCap,
  TrendingUp,
  GitBranch,
  Map,
  Menu,
  X,
} from 'lucide-react'
import { useHealth } from '@/hooks/useHealth'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/data', icon: Database, label: 'Data' },
  { to: '/training', icon: GraduationCap, label: 'Training' },
  { to: '/forecasting', icon: TrendingUp, label: 'Forecasting' },
  { to: '/counterfactual', icon: GitBranch, label: 'Counterfactual' },
  { to: '/observatory', icon: Map, label: 'Observatory' },
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
          Junon Explorer
        </span>
      </NavLink>

      <div className="hidden md:flex items-center gap-1">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
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
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
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
