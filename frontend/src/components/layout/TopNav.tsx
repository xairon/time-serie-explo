import { useState, useEffect } from 'react'
import { NavLink, Link, useLocation, useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Waves,
  Brain,
  Map,
  GitCompare,
  Menu,
  X,
} from 'lucide-react'
import { useHealth } from '@/hooks/useHealth'
import { useCompareSelection } from '@/contexts/CompareSelection'
import { LanguageSwitcher } from '../LanguageSwitcher'
import { useAuth } from '@/contexts/AuthContext'

export function TopNav() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const { t } = useTranslation()
  const { data: health } = useHealth()
  const { count: compareCount, type: compareType } = useCompareSelection()
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const navItems = [
    { to: '/', icon: Map, label: t('nav.observatory'), end: true, tour: 'nav-observatory' },
    // Météo des nappes désactivée pour l'instant — réactiver: restaurer cette entrée + l'import CloudRain
    // { to: '/meteo', icon: CloudRain, label: t('nav.meteo'), end: false, tour: 'nav-meteo' },
    { to: '/compare', icon: GitCompare, label: t('nav.compare'), end: false, tour: 'nav-compare' },
    { to: '/pastas', icon: Waves, label: t('nav.pastasLab'), end: false, tour: 'nav-pastas' },
    { to: '/ai', icon: Brain, label: t('nav.aiLab'), end: false, tour: 'nav-ai' },
  ] as const

  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  const isHealthy = health?.status === 'ok'
  const compareTo = compareType ? `/compare/${compareType}` : '/compare'

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
        {navItems.map(({ to, icon: Icon, label, end, tour }) => {
          // The Comparer link points to the type currently selected so a single
          // click always lands on a populated page when the basket is non-empty.
          const effectiveTo = to === '/compare' ? compareTo : to
          const showBadge = to === '/compare' && compareCount > 0
          return (
            <NavLink
              key={to}
              to={effectiveTo}
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
              {showBadge && (
                <span className="ml-1 inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full bg-accent-cyan text-bg-primary text-[10px] font-bold leading-none">
                  {compareCount}
                </span>
              )}
            </NavLink>
          )
        })}
      </div>

      <div className="ml-auto flex items-center gap-3">
        {health?.gpu?.available && (
          <span className="text-[10px] text-text-muted hidden sm:block" title={health.gpu.device ?? ''}>
            GPU
          </span>
        )}
        <div className="flex items-center gap-1.5" title={isHealthy ? t('nav.apiConnected') : t('nav.apiUnavailable')}>
          <div
            className={`w-2 h-2 rounded-full ${
              isHealthy ? 'bg-accent-green' : 'bg-accent-red'
            }`}
          />
          <span className="text-xs text-text-secondary hidden sm:block">
            {isHealthy ? 'OK' : t('common.error')}
          </span>
        </div>

        <LanguageSwitcher />

        {/* Account area */}
        {user ? (
          <div className="hidden md:flex items-center gap-2">
            {user.role === 'admin' && (
              <Link
                to="/admin/users"
                className="text-xs px-2.5 py-1 rounded-lg text-text-secondary border border-white/10 hover:text-text-primary hover:border-white/20 transition-colors"
              >
                {t('nav.admin')}
              </Link>
            )}
            <span className="text-xs text-text-secondary hidden lg:block">{user.display_name}</span>
            <button
              onClick={async () => { await logout(); navigate('/') }}
              className="text-xs px-2.5 py-1 rounded-lg text-text-secondary border border-white/10 hover:text-text-primary hover:border-white/20 transition-colors"
            >
              {t('auth.logout')}
            </button>
          </div>
        ) : (
          <Link
            to="/login"
            className="hidden md:inline-flex text-xs px-2.5 py-1 rounded-lg text-accent-cyan border border-accent-cyan/30 hover:bg-accent-cyan/10 transition-colors"
          >
            {t('auth.login')}
          </Link>
        )}

        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="md:hidden p-2 hover:bg-bg-hover rounded-lg"
          aria-label={mobileOpen ? t('common.close') : t('common.search')}
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {mobileOpen && (
        <div className="md:hidden absolute top-12 left-0 right-0 bg-bg-card border-b border-white/10 shadow-xl z-40">
          {navItems.map(({ to, icon: Icon, label, end }) => {
            const effectiveTo = to === '/compare' ? compareTo : to
            const showBadge = to === '/compare' && compareCount > 0
            return (
              <NavLink
                key={to}
                to={effectiveTo}
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
                {showBadge && (
                  <span className="ml-auto inline-flex items-center justify-center min-w-[20px] h-[20px] px-1.5 rounded-full bg-accent-cyan text-bg-primary text-[11px] font-bold">
                    {compareCount}
                  </span>
                )}
              </NavLink>
            )
          })}
        </div>
      )}
    </nav>
  )
}
