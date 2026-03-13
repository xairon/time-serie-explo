import { NavLink, Outlet } from 'react-router-dom'
import { Waypoints, GitCompareArrows, Droplets } from 'lucide-react'

const LAB_TABS = [
  { to: '/lab/latent-space', icon: Waypoints, label: 'Espace Latent' },
  { to: '/lab/counterfactual', icon: GitCompareArrows, label: 'Contrefactuel' },
  { to: '/lab/pumping-detection', icon: Droplets, label: 'Détection Pompage' },
] as const

export default function LabLayout() {
  return (
    <div className="flex flex-col h-full">
      <div className="bg-bg-card border-b border-white/5 shrink-0">
        <div className="flex items-center px-4 gap-1">
          {LAB_TABS.map(({ to, icon: Icon, label }) => (
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
      </div>
      <div className="flex-1 min-h-0 overflow-auto">
        <Outlet />
      </div>
    </div>
  )
}
