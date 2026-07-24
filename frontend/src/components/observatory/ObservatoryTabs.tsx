import { NavLink } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

/**
 * Sub-tab bar for the two Observatoire compartments (Nappes & rivières /
 * Climat). Rendered by ObservatoryShell above the routed page.
 *
 * Deliberately visually subordinate to TopNav: smaller text, an underline
 * for the active state rather than a filled pill, so the hierarchy reads as
 * two distinct levels (top-level domain vs. compartment within it).
 */
export function ObservatoryTabs() {
  const { t } = useTranslation()

  const tabs = [
    { to: '/', label: t('observatory.tabs.groundwater'), end: true, tour: undefined as string | undefined },
    // Carries the 'nav-climat' guided-tour anchor moved off TopNav, since
    // Climat is no longer a top-level nav item (cf. TopNav.tsx).
    { to: '/climat', label: t('nav.climat'), end: false, tour: 'nav-climat' },
  ]

  return (
    <nav className="flex items-center gap-5 px-4 h-9 border-b border-white/5 bg-bg-card/60 shrink-0">
      {tabs.map(({ to, label, end, tour }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          data-tour={tour}
          className={({ isActive }) =>
            `text-xs font-medium py-1.5 border-b-2 transition-colors ${
              isActive
                ? 'border-accent-cyan text-text-primary'
                : 'border-transparent text-text-secondary hover:text-text-primary'
            }`
          }
        >
          {label}
        </NavLink>
      ))}
    </nav>
  )
}
