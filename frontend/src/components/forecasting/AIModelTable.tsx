import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Trash2, TrendingUp } from 'lucide-react'
import { useModels, useDeleteModel } from '@/hooks/useModels'
import type { ModelSummary } from '@/lib/types'

type SortKey = 'model_name' | 'created_at' | 'MAE' | 'RMSE' | 'NSE'

function metric(m: ModelSummary, k: string): number | null {
  const v = m.metrics?.[k] ?? m.metrics?.[k.toLowerCase()]
  return typeof v === 'number' ? v : null
}

/** Sortable, filterable table of trained AI (Darts) models — AI-side counterpart
 *  of the Pastas gallery. Reads the same /models list the forecasting page uses. */
export function AIModelTable() {
  const navigate = useNavigate()
  const { data: models, isLoading } = useModels()
  const del = useDeleteModel()
  const [filter, setFilter] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('created_at')
  const [asc, setAsc] = useState(false)

  const rows = useMemo(() => {
    const f = filter.toLowerCase()
    const list = (models ?? []).filter((m) =>
      m.model_name.toLowerCase().includes(f) ||
      (m.primary_station ?? '').toLowerCase().includes(f) ||
      (m.data_source ?? m.dataset_id ?? '').toLowerCase().includes(f),
    )
    return [...list].sort((a, b) => {
      let va: number | string, vb: number | string
      if (sortKey === 'model_name') { va = a.model_name; vb = b.model_name }
      else if (sortKey === 'created_at') { va = a.created_at; vb = b.created_at }
      else { va = metric(a, sortKey) ?? -Infinity; vb = metric(b, sortKey) ?? -Infinity }
      const cmp = typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb))
      return asc ? cmp : -cmp
    })
  }, [models, filter, sortKey, asc])

  const toggleSort = (k: SortKey) => { if (sortKey === k) setAsc(!asc); else { setSortKey(k); setAsc(false) } }

  if (isLoading) return <div className="text-text-muted text-sm py-8 text-center">Chargement…</div>
  if (!models?.length) {
    return <p className="text-sm text-text-secondary italic py-8 text-center">Aucun modèle entraîné. Lancez un entraînement dans l'onglet « Entraînement ».</p>
  }

  const Th = ({ k, label, right }: { k: SortKey; label: string; right?: boolean }) => (
    <th className={`px-3 py-2 cursor-pointer select-none ${right ? 'text-right' : 'text-left'} ${sortKey === k ? 'text-text-primary' : ''}`} onClick={() => toggleSort(k)}>
      {label}{sortKey === k ? (asc ? ' ↑' : ' ↓') : ''}
    </th>
  )

  return (
    <div className="space-y-3">
      <div className="relative max-w-sm">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
        <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filtrer (modèle, station, dataset)…"
          className="w-full bg-bg-input border border-white/10 rounded-lg pl-9 pr-3 py-2 text-sm text-text-primary placeholder:text-text-muted" />
      </div>
      <div className="overflow-x-auto rounded-lg border border-white/5">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-bg-hover text-text-secondary text-xs uppercase tracking-wide">
              <Th k="model_name" label="Modèle" />
              <th className="px-3 py-2 text-left">Type</th>
              <th className="px-3 py-2 text-left">Station</th>
              <th className="px-3 py-2 text-left">Dataset</th>
              <Th k="MAE" label="MAE" right />
              <Th k="RMSE" label="RMSE" right />
              <Th k="NSE" label="NSE" right />
              <Th k="created_at" label="Date" />
              <th className="px-3 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => {
              const station = m.primary_station || m.stations?.[0] || '—'
              const mae = metric(m, 'MAE'), rmse = metric(m, 'RMSE'), nse = metric(m, 'NSE')
              const date = m.created_at ? new Date(m.created_at) : null
              return (
                <tr key={m.model_id} className="border-t border-white/5 hover:bg-bg-hover/50">
                  <td className="px-3 py-1.5 text-text-primary font-medium">{m.model_name}</td>
                  <td className="px-3 py-1.5 text-text-secondary">{m.model_type}</td>
                  <td className="px-3 py-1.5 text-text-secondary">{station === 'default' ? '—' : station}</td>
                  <td className="px-3 py-1.5 text-text-secondary text-xs">{m.data_source || m.dataset_id || '—'}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-text-primary">{mae != null ? mae.toFixed(3) : '—'}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-text-secondary">{rmse != null ? rmse.toFixed(3) : '—'}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-text-secondary">{nse != null ? nse.toFixed(3) : '—'}</td>
                  <td className="px-3 py-1.5 text-text-secondary text-xs">{date && !isNaN(date.getTime()) ? date.toLocaleDateString('fr-FR') : '—'}</td>
                  <td className="px-3 py-1.5">
                    <div className="flex items-center gap-1 justify-end">
                      <button title="Prévoir avec ce modèle" onClick={() => navigate(`/ai/forecasting?model=${m.model_id}`)} className="p-1.5 rounded hover:bg-white/10 text-accent-cyan"><TrendingUp className="w-3.5 h-3.5" /></button>
                      <button title="Supprimer" onClick={() => { if (window.confirm('Supprimer ce modèle ?')) del.mutate(m.model_id) }} className="p-1.5 rounded hover:bg-white/10 text-accent-red"><Trash2 className="w-3.5 h-3.5" /></button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
