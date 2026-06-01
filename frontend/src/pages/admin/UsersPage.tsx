import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { adminApi } from '@/lib/auth'
import type { AuthUser } from '@/lib/auth'

const ROLE_LABELS: Record<string, string> = {
  admin: 'Administrateur',
  user: 'Utilisateur',
}

function CreateUserForm({ onCreated }: { onCreated: () => void }) {
  const [open, setOpen] = useState(false)
  const [email, setEmail] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [role, setRole] = useState('user')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const reset = () => { setEmail(''); setDisplayName(''); setRole('user'); setPassword(''); setError(null) }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (password.length < 8) { setError('Le mot de passe doit contenir au moins 8 caractères.'); return }
    setError(null)
    setSubmitting(true)
    try {
      await adminApi.create({ email, display_name: displayName, role, initial_password: password })
      reset()
      setOpen(false)
      onCreated()
    } catch {
      setError("Erreur lors de la création de l'utilisateur.")
    } finally {
      setSubmitting(false)
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 transition-colors"
      >
        + Nouvel utilisateur
      </button>
    )
  }

  return (
    <div className="bg-bg-card border border-white/5 rounded-xl p-5 space-y-4">
      <h2 className="text-sm font-semibold text-text-primary">Créer un utilisateur</h2>
      <form onSubmit={submit} className="space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <label className="block text-xs font-medium text-text-secondary">Adresse e-mail</label>
            <input
              type="email"
              required
              value={email}
              onChange={e => setEmail(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-bg-primary border border-white/10 text-text-primary text-sm placeholder:text-text-muted focus:outline-none focus:border-accent-cyan/50 focus:ring-1 focus:ring-accent-cyan/30 transition-colors"
              placeholder="utilisateur@brgm.fr"
            />
          </div>
          <div className="space-y-1.5">
            <label className="block text-xs font-medium text-text-secondary">Nom d'affichage</label>
            <input
              type="text"
              required
              value={displayName}
              onChange={e => setDisplayName(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-bg-primary border border-white/10 text-text-primary text-sm placeholder:text-text-muted focus:outline-none focus:border-accent-cyan/50 focus:ring-1 focus:ring-accent-cyan/30 transition-colors"
              placeholder="Prénom Nom"
            />
          </div>
          <div className="space-y-1.5">
            <label className="block text-xs font-medium text-text-secondary">Rôle</label>
            <select
              value={role}
              onChange={e => setRole(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-bg-primary border border-white/10 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50 focus:ring-1 focus:ring-accent-cyan/30 transition-colors"
            >
              <option value="user">Utilisateur</option>
              <option value="admin">Administrateur</option>
            </select>
          </div>
          <div className="space-y-1.5">
            <label className="block text-xs font-medium text-text-secondary">Mot de passe initial (min. 8 caractères)</label>
            <input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-bg-primary border border-white/10 text-text-primary text-sm placeholder:text-text-muted focus:outline-none focus:border-accent-cyan/50 focus:ring-1 focus:ring-accent-cyan/30 transition-colors"
              placeholder="••••••••"
            />
          </div>
        </div>

        {error && (
          <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
            {error}
          </p>
        )}

        <div className="flex gap-2">
          <button
            type="submit"
            disabled={submitting}
            className="px-4 py-2 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? 'Création…' : 'Créer'}
          </button>
          <button
            type="button"
            onClick={() => { reset(); setOpen(false) }}
            className="px-4 py-2 rounded-lg text-text-secondary text-sm hover:text-text-primary hover:bg-bg-hover transition-colors"
          >
            Annuler
          </button>
        </div>
      </form>
    </div>
  )
}

function ResetPasswordModal({ user, onClose }: { user: AuthUser; onClose: () => void }) {
  const [newPassword, setNewPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const queryClient = useQueryClient()

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (newPassword.length < 8) { setError('Le mot de passe doit contenir au moins 8 caractères.'); return }
    setError(null)
    setSubmitting(true)
    try {
      await adminApi.update(user.id, { new_password: newPassword })
      queryClient.invalidateQueries({ queryKey: ['admin-users'] })
      onClose()
    } catch {
      setError('Erreur lors de la réinitialisation du mot de passe.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-bg-card border border-white/10 rounded-xl p-6 w-full max-w-sm space-y-4">
        <h2 className="text-sm font-semibold text-text-primary">
          Réinitialiser le mot de passe — <span className="text-text-secondary font-normal">{user.email}</span>
        </h2>
        <form onSubmit={submit} className="space-y-3">
          <div className="space-y-1.5">
            <label className="block text-xs font-medium text-text-secondary">Nouveau mot de passe (min. 8 caractères)</label>
            <input
              type="password"
              required
              minLength={8}
              value={newPassword}
              onChange={e => setNewPassword(e.target.value)}
              autoFocus
              className="w-full px-3 py-2 rounded-lg bg-bg-primary border border-white/10 text-text-primary text-sm placeholder:text-text-muted focus:outline-none focus:border-accent-cyan/50 focus:ring-1 focus:ring-accent-cyan/30 transition-colors"
              placeholder="••••••••"
            />
          </div>
          {error && (
            <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}
          <div className="flex gap-2">
            <button
              type="submit"
              disabled={submitting}
              className="px-4 py-2 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {submitting ? 'Enregistrement…' : 'Enregistrer'}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-text-secondary text-sm hover:text-text-primary hover:bg-bg-hover transition-colors"
            >
              Annuler
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function UsersPage() {
  const queryClient = useQueryClient()
  const [resetTarget, setResetTarget] = useState<AuthUser | null>(null)

  const { data: users, isLoading, isError } = useQuery({
    queryKey: ['admin-users'],
    queryFn: () => adminApi.list(),
  })

  const toggleActive = useMutation({
    mutationFn: ({ id, is_active }: { id: string; is_active: boolean }) =>
      adminApi.update(id, { is_active }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['admin-users'] }),
  })

  if (isLoading) {
    return (
      <div className="p-6 max-w-5xl mx-auto space-y-6">
        <div className="h-8 w-48 bg-white/10 rounded animate-pulse" />
        <div className="bg-bg-card border border-white/5 rounded-xl">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-center gap-4 px-4 py-3 border-b border-white/5">
              <div className="h-4 w-48 bg-white/10 rounded animate-pulse" />
              <div className="h-4 w-32 bg-white/5 rounded animate-pulse" />
              <div className="h-4 w-20 bg-white/5 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <p className="text-red-400 text-sm">Erreur lors du chargement des utilisateurs.</p>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-text-primary">Gestion des utilisateurs</h1>
          <p className="text-sm text-text-secondary mt-0.5">{users?.length ?? 0} utilisateur{(users?.length ?? 0) !== 1 ? 's' : ''}</p>
        </div>
        <CreateUserForm onCreated={() => queryClient.invalidateQueries({ queryKey: ['admin-users'] })} />
      </div>

      <div className="bg-bg-card border border-white/5 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm min-w-[640px]">
            <thead>
              <tr className="border-b border-white/5">
                <th className="px-4 py-3 text-left text-xs font-medium text-text-secondary">E-mail</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-text-secondary">Nom d'affichage</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-text-secondary">Rôle</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-text-secondary">Statut</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-text-secondary">Actions</th>
              </tr>
            </thead>
            <tbody>
              {(users ?? []).map((u: AuthUser) => (
                <tr key={u.id} className="border-b border-white/5 hover:bg-bg-hover transition-colors">
                  <td className="px-4 py-3 text-text-primary text-sm font-mono">{u.email}</td>
                  <td className="px-4 py-3 text-text-primary text-sm">{u.display_name}</td>
                  <td className="px-4 py-3">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                      u.role === 'admin'
                        ? 'bg-accent-cyan/20 text-accent-cyan'
                        : 'bg-white/5 text-text-secondary'
                    }`}>
                      {ROLE_LABELS[u.role] ?? u.role}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                      u.is_active
                        ? 'bg-accent-green/20 text-accent-green'
                        : 'bg-white/5 text-text-muted'
                    }`}>
                      {u.is_active ? 'Actif' : 'Inactif'}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => toggleActive.mutate({ id: u.id, is_active: !u.is_active })}
                        disabled={toggleActive.isPending}
                        className="text-xs px-2.5 py-1 rounded-lg text-text-secondary border border-white/10 hover:text-text-primary hover:border-white/20 disabled:opacity-40 transition-colors"
                        title={u.is_active ? 'Désactiver le compte' : 'Activer le compte'}
                      >
                        {u.is_active ? 'Désactiver' : 'Activer'}
                      </button>
                      <button
                        onClick={() => setResetTarget(u)}
                        className="text-xs px-2.5 py-1 rounded-lg text-text-secondary border border-white/10 hover:text-text-primary hover:border-white/20 transition-colors"
                        title="Réinitialiser le mot de passe"
                      >
                        Réinit. MDP
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {(users ?? []).length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-text-secondary text-sm">
                    Aucun utilisateur trouvé.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {resetTarget && (
        <ResetPasswordModal user={resetTarget} onClose={() => setResetTarget(null)} />
      )}
    </div>
  )
}
