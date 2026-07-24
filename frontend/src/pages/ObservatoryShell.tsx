import { Outlet } from 'react-router-dom'
import { ObservatoryTabs } from '@/components/observatory/ObservatoryTabs'

/**
 * Pathless layout route wrapping the Observatoire compartments (Nappes &
 * rivières / Climat). Purely a shell: it renders the sub-tab bar then hands
 * off to whichever compartment route matched (see routes.tsx). No state, no
 * map/layer logic here — that stays inside ObservatoryPage/ClimatPage.
 */
export default function ObservatoryShell() {
  return (
    <div className="flex flex-col h-full">
      <ObservatoryTabs />
      <div className="relative flex-1 min-h-0">
        <Outlet />
      </div>
    </div>
  )
}
