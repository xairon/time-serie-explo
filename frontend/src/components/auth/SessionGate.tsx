import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'

/**
 * Forces a logged-in user with a reset (temporary) password to change it before
 * using the rest of the app. Public pages and the account/login pages stay reachable.
 */
export function SessionGate({ children }: { children: React.ReactNode }) {
  const { user } = useAuth()
  const loc = useLocation()
  const exempt = loc.pathname.startsWith('/account') || loc.pathname.startsWith('/login')
  if (user?.must_change_password && !exempt) {
    return <Navigate to="/account" replace />
  }
  return <>{children}</>
}
