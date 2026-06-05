import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'
import { authApi } from '@/lib/auth'

export default function AccountPage() {
  const { user, logout, refresh } = useAuth()
  const nav = useNavigate()

  const [oldPwd, setOldPwd] = useState('')
  const [newPwd, setNewPwd] = useState('')
  const [pwdMsg, setPwdMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const [pwdBusy, setPwdBusy] = useState(false)

  const [confirmDelete, setConfirmDelete] = useState(false)
  const [delBusy, setDelBusy] = useState(false)
  const [delErr, setDelErr] = useState<string | null>(null)

  const mustChange = user?.must_change_password === true

  const changePassword = async (e: React.FormEvent) => {
    e.preventDefault()
    setPwdMsg(null)
    if (newPwd.length < 8) {
      setPwdMsg({ ok: false, text: 'Le nouveau mot de passe doit faire au moins 8 caractères.' })
      return
    }
    setPwdBusy(true)
    try {
      await authApi.changePassword(oldPwd, newPwd)
      await refresh()
      setOldPwd(''); setNewPwd('')
      setPwdMsg({ ok: true, text: 'Mot de passe mis à jour.' })
    } catch {
      setPwdMsg({ ok: false, text: "Échec : vérifiez votre ancien mot de passe." })
    } finally {
      setPwdBusy(false)
    }
  }

  const deleteAccount = async () => {
    setDelErr(null)
    setDelBusy(true)
    try {
      await authApi.deleteAccount()
      await logout()
      nav('/login', { replace: true })
    } catch {
      setDelErr('La suppression a échoué. Réessayez plus tard.')
      setDelBusy(false)
    }
  }

  return (
    <div className="flex justify-center p-6">
      <div className="w-full max-w-lg space-y-6">
        <div>
          <h1 className="text-xl font-bold text-text-primary">Mon compte</h1>
          {user && <p className="text-sm text-text-secondary mt-1">{user.email}</p>}
        </div>

        {mustChange && (
          <p className="text-sm text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2">
            Votre mot de passe a été réinitialisé. Vous devez le changer avant de continuer.
          </p>
        )}

        <form onSubmit={changePassword} className="bg-bg-card border border-white/5 rounded-xl p-6 space-y-4">
          <h2 className="text-sm font-semibold text-text-primary">Changer le mot de passe</h2>
          <div className="space-y-1.5">
            <label htmlFor="old" className="block text-sm font-medium text-text-primary">Mot de passe actuel</label>
            <input id="old" type="password" autoComplete="current-password" required value={oldPwd}
              onChange={e => setOldPwd(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-bg-primary border border-white/10 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50" />
          </div>
          <div className="space-y-1.5">
            <label htmlFor="new" className="block text-sm font-medium text-text-primary">Nouveau mot de passe</label>
            <input id="new" type="password" autoComplete="new-password" required minLength={8} value={newPwd}
              onChange={e => setNewPwd(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-bg-primary border border-white/10 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50" />
          </div>
          {pwdMsg && (
            <p className={`text-sm rounded-lg px-3 py-2 border ${pwdMsg.ok ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20' : 'text-red-400 bg-red-500/10 border-red-500/20'}`}>
              {pwdMsg.text}
            </p>
          )}
          <button type="submit" disabled={pwdBusy}
            className="px-4 py-2 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 transition-colors">
            {pwdBusy ? 'Enregistrement…' : 'Mettre à jour'}
          </button>
        </form>

        <div className="bg-bg-card border border-red-500/20 rounded-xl p-6 space-y-3">
          <h2 className="text-sm font-semibold text-red-400">Supprimer mon compte</h2>
          <p className="text-sm text-text-secondary">
            Conformément au RGPD, vous pouvez supprimer définitivement votre compte et les données
            associées (jeux de données importés, analyses et modèles). Cette action est irréversible.
          </p>
          {delErr && <p className="text-sm text-red-400">{delErr}</p>}
          {!confirmDelete ? (
            <button onClick={() => setConfirmDelete(true)}
              className="px-4 py-2 rounded-lg bg-red-500/15 text-red-400 text-sm font-medium border border-red-500/30 hover:bg-red-500/25 transition-colors">
              Supprimer mon compte
            </button>
          ) : (
            <div className="flex items-center gap-3">
              <button onClick={deleteAccount} disabled={delBusy}
                className="px-4 py-2 rounded-lg bg-red-500/25 text-red-300 text-sm font-medium border border-red-500/40 hover:bg-red-500/35 disabled:opacity-40 transition-colors">
                {delBusy ? 'Suppression…' : 'Confirmer la suppression définitive'}
              </button>
              <button onClick={() => setConfirmDelete(false)} disabled={delBusy}
                className="px-4 py-2 rounded-lg text-text-secondary text-sm hover:text-text-primary transition-colors">
                Annuler
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
