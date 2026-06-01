import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { authApi, type AuthUser } from '@/lib/auth'

interface AuthState {
  user: AuthUser | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const Ctx = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    authApi.me().then(setUser).catch(() => setUser(null)).finally(() => setLoading(false))
    const onUnauth = () => setUser(null)
    window.addEventListener('auth:unauthorized', onUnauth)
    return () => window.removeEventListener('auth:unauthorized', onUnauth)
  }, [])

  const login = async (email: string, password: string) => setUser(await authApi.login(email, password))
  const logout = async () => { await authApi.logout(); setUser(null) }

  return <Ctx.Provider value={{ user, loading, login, logout }}>{children}</Ctx.Provider>
}

export function useAuth() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useAuth must be used within AuthProvider')
  return v
}
