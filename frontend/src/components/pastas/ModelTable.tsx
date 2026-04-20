import { useState } from 'react'
import { Trash2, ArrowUpDown } from 'lucide-react'
import { usePastasModels, usePastasDeleteModel } from '@/hooks/usePastas'
import { ExportMenu } from './ExportMenu'
import type { PastasModelSummary } from '@/lib/types'

type SortKey = keyof PastasModelSummary

export function ModelTable() {
  const { data: models, isLoading } = usePastasModels()
  const deleteMut = usePastasDeleteModel()
  const [sortKey, setSortKey] = useState<SortKey>('created_at')
  const [sortAsc, setSortAsc] = useState(false)
  const [filter, setFilter] = useState('')

  const filtered = (models ?? []).filter(m =>
    m.code_bss.toLowerCase().includes(filter.toLowerCase()) ||
    m.name.toLowerCase().includes(filter.toLowerCase())
  )

  const sorted = [...filtered].sort((a, b) => {
    const va = a[sortKey] ?? ''
    const vb = b[sortKey] ?? ''
    const cmp = typeof va === 'number' && typeof vb === 'number'
      ? va - vb
      : String(va).localeCompare(String(vb))
    return sortAsc ? cmp : -cmp
  })

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc)
    else { setSortKey(key); setSortAsc(true) }
  }

  const SortHeader = ({ k, label }: { k: SortKey; label: string }) => (
    <th className="px-3 py-2 text-left cursor-pointer hover:text-text-primary transition-colors"
      onClick={() => toggleSort(k)}>
      <span className="flex items-center gap-1">
        {label}
        {sortKey === k && <ArrowUpDown className="w-3 h-3" />}
      </span>
    </th>
  )

  if (isLoading) return <div className="text-text-muted text-sm">Loading models...</div>

  return (
    <div className="space-y-3">
      <input
        type="text"
        placeholder="Filter by station or name..."
        value={filter}
        onChange={e => setFilter(e.target.value)}
        className="w-full max-w-sm bg-bg-primary border border-white/10 rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-muted"
      />

      {sorted.length === 0 ? (
        <div className="text-text-muted text-sm py-8 text-center">
          {filter ? 'No models match your filter.' : 'No models fitted yet.'}
        </div>
      ) : (
        <div className="bg-bg-card rounded-lg border border-white/5 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-text-muted border-b border-white/5 text-xs uppercase tracking-wide">
                  <SortHeader k="name" label="Name" />
                  <SortHeader k="code_bss" label="Station" />
                  <SortHeader k="response_type" label="Response" />
                  <SortHeader k="evp" label="EVP %" />
                  <SortHeader k="rmse" label="RMSE" />
                  <SortHeader k="created_at" label="Created" />
                  <th className="px-3 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map(m => (
                  <tr key={m.run_id} className="border-b border-white/5 hover:bg-bg-hover transition-colors">
                    <td className="px-3 py-2 text-text-primary font-medium">{m.name || m.run_id.slice(0, 8)}</td>
                    <td className="px-3 py-2 text-accent-cyan font-mono">{m.code_bss}</td>
                    <td className="px-3 py-2 text-text-secondary">{m.response_type}</td>
                    <td className="px-3 py-2 text-text-primary">{m.evp?.toFixed(1) ?? '—'}</td>
                    <td className="px-3 py-2 text-text-primary">{m.rmse?.toFixed(4) ?? '—'}</td>
                    <td className="px-3 py-2 text-text-muted">{new Date(Number(m.created_at)).toLocaleDateString()}</td>
                    <td className="px-3 py-2 text-right">
                      <div className="flex items-center justify-end gap-1">
                        <ExportMenu runId={m.run_id} />
                        <button
                          onClick={() => deleteMut.mutate(m.run_id)}
                          className="p-1 hover:bg-bg-hover rounded text-text-muted hover:text-red-400 transition-colors"
                          title="Delete model"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="text-xs text-text-muted">{sorted.length} model(s)</div>
    </div>
  )
}
