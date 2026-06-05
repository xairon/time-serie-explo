import { API_BASE } from './constants'

export interface AuthUser {
  id: string; email: string; display_name: string; role: 'admin' | 'user'; is_active: boolean
  must_change_password?: boolean
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
  changePassword: (old_password: string, new_password: string) =>
    req<void>('/auth/change-password', { method: 'POST', body: JSON.stringify({ old_password, new_password }) }),
  deleteAccount: () => req<void>('/auth/me', { method: 'DELETE' }),
}

export const adminApi = {
  list: () => req<AuthUser[]>('/admin/users'),
  create: (body: { email: string; display_name: string; role: string; initial_password: string }) =>
    req<AuthUser>('/admin/users', { method: 'POST', body: JSON.stringify(body) }),
  update: (id: string, body: { is_active?: boolean; role?: string; new_password?: string }) =>
    req<AuthUser>(`/admin/users/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  resetPassword: (id: string) =>
    req<{ temporary_password: string }>(`/admin/users/${id}/reset-password`, { method: 'POST' }),
  remove: (id: string) => req<void>(`/admin/users/${id}`, { method: 'DELETE' }),
}
