import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'

export function RequireAuth({ children, adminOnly = false }: { children: React.ReactNode; adminOnly?: boolean }) {
  const { user, loading } = useAuth()
  const loc = useLocation()
  if (loading) return <div className="flex items-center justify-center h-full text-text-secondary">Chargement...</div>
  if (!user) return <Navigate to="/login" replace state={{ from: loc.pathname + loc.search + loc.hash }} />
  if (adminOnly && user.role !== 'admin') return <Navigate to="/" replace />
  return <>{children}</>
}
