import { API_BASE } from './constants'

export interface AuthUser {
  id: string; email: string; display_name: string; role: 'admin' | 'user'; is_active: boolean
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    credentials: 'include',
    headers: { 'Accept': 'application/json', 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) throw new Error(`${res.status}`)
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const authApi = {
  me: () => req<AuthUser>('/auth/me'),
  login: (email: string, password: string) =>
    req<AuthUser>('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  logout: () => req<void>('/auth/logout', { method: 'POST' }),
}
