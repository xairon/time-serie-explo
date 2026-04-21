import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Trash2, Eye, Search, ArrowUpDown, LayoutGrid, List } from 'lucide-react'
import { usePastasModels, usePastasDeleteModel } from '@/hooks/usePastas'
import { ExportMenu } from './ExportMenu'
import type { PastasModelSummary } from '@/lib/types'

type SortKey = 'name' | 'code_bss' | 'evp' | 'nse' | 'created_at'

function nseColor(v: number | null): string {
  if (v == null) return 'text-text-muted'
  if (v > 0.7) return 'text-green-400'
  if (v > 0.4) return 'text-accent-cyan'
  return 'text-red-400'
}

function StowaMiniBadge({ nse, evp }: { nse: number | null; evp: number | null }) {
  const evpPass = evp != null && evp >= 70
  const nsePass = nse != null && nse >= 0.5
  return (
    <div className="flex items-center gap-0.5" title={`EVP ${evpPass ? 'pass' : 'fail'}, NSE ${nsePass ? 'pass' : 'fail'}`}>
      <span className={`w-2 h-2 rounded-full ${evpPass ? 'bg-green-400' : 'bg-red-400'}`} />
      <span className={`w-2 h-2 rounded-full ${nsePass ? 'bg-green-400' : 'bg-red-400'}`} />
    </div>
  )
}

function ConfigTag({ label, color }: { label: string; color: string }) {
  return (
    <span className={`inline-flex px-1.5 py-0.5 rounded text-[10px] font-medium border ${color}`}>
      {label}
    </span>
  )
}

function ModelCard({ m, onView, onDelete }: {
  m: PastasModelSummary
  onView: () => void
  onDelete: () => void
}) {
  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4 hover:border-white/10 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <div className="text-sm font-medium text-text-primary truncate">{m.name || m.run_id.slice(0, 8)}</div>
            <StowaMiniBadge nse={m.nse} evp={m.evp} />
          </div>
          <a href={`/station/piezo/${encodeURIComponent(m.code_bss)}`} className="text-xs font-mono text-accent-cyan hover:underline mt-0.5 block">{m.code_bss}</a>
        </div>
        <div className="flex items-center gap-0.5 shrink-0 ml-2">
          <button onClick={onView} className="p-1.5 hover:bg-bg-hover rounded-lg text-text-muted hover:text-accent-cyan transition-colors" title="View results">
            <Eye className="w-4 h-4" />
          </button>
          <ExportMenu runId={m.run_id} />
          <button onClick={onDelete} className="p-1.5 hover:bg-bg-hover rounded-lg text-text-muted hover:text-red-400 transition-colors" title="Delete">
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Config tags */}
      <div className="flex flex-wrap gap-1 mb-3">
        <ConfigTag label={m.recharge_type} color="border-blue-500/30 text-blue-400 bg-blue-500/10" />
        <ConfigTag label={m.response_type} color="border-emerald-500/30 text-emerald-400 bg-emerald-500/10" />
        {m.noise_type !== 'unknown' && m.noise_type !== 'none' && (
          <ConfigTag label={m.noise_type.replace('NoiseModel', '').replace('Noise', '')} color="border-amber-500/30 text-amber-400 bg-amber-500/10" />
        )}
        {m.noise_type === 'none' && (
          <ConfigTag label="No noise" color="border-white/10 text-text-muted bg-white/5" />
        )}
        <ConfigTag label={m.solver_type} color="border-purple-500/30 text-purple-400 bg-purple-500/10" />
        {m.include_temp && (
          <ConfigTag label="Temp." color="border-orange-500/30 text-orange-400 bg-orange-500/10" />
        )}
      </div>

      {/* Metrics */}
      <div className={`grid ${m.has_validation ? 'grid-cols-2' : 'grid-cols-1'} gap-2`}>
        <div className={`rounded-lg border border-accent-cyan/20 p-2 ${m.has_validation ? '' : ''}`}>
          {m.has_validation && <div className="text-[9px] uppercase tracking-wider text-accent-cyan font-semibold mb-1">Train</div>}
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-text-muted">NSE</span>
              <span className={`text-xs font-semibold ${nseColor(m.nse)}`}>{m.nse?.toFixed(3) ?? '—'}</span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] text-text-muted">EVP</span>
              <span className={`text-xs font-semibold ${nseColor(m.evp != null ? m.evp / 100 : null)}`}>{m.evp?.toFixed(1) ?? '—'}%</span>
            </div>
          </div>
        </div>
        {m.has_validation && (
          <div className="rounded-lg border border-orange-500/20 p-2">
            <div className="text-[9px] uppercase tracking-wider text-orange-400 font-semibold mb-1">Test</div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
              <div className="flex items-baseline justify-between">
                <span className="text-[10px] text-text-muted">NSE</span>
                <span className={`text-xs font-semibold ${nseColor(m.val_nse)}`}>{m.val_nse?.toFixed(3) ?? '—'}</span>
              </div>
              <div className="flex items-baseline justify-between">
                <span className="text-[10px] text-text-muted">EVP</span>
                <span className={`text-xs font-semibold ${nseColor(m.val_evp != null ? m.val_evp / 100 : null)}`}>{m.val_evp?.toFixed(1) ?? '—'}%</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between mt-3 pt-2 border-t border-white/5">
        <span className="text-[10px] text-text-muted">{new Date(Number(m.created_at)).toLocaleDateString('en-GB')}</span>
        {m.aic != null && (
          <span className="text-[10px] text-text-muted">AIC {m.aic.toFixed(1)}</span>
        )}
        <span className="text-[10px] text-text-muted">Pastas {m.pastas_version}</span>
      </div>
    </div>
  )
}

export function ModelTable() {
  const navigate = useNavigate()
  const { data: models, isLoading } = usePastasModels()
  const deleteMut = usePastasDeleteModel()
  const [sortKey, setSortKey] = useState<SortKey>('created_at')
  const [sortAsc, setSortAsc] = useState(false)
  const [filter, setFilter] = useState('')
  const [view, setView] = useState<'grid' | 'list'>('grid')

  const filtered = (models ?? []).filter(m =>
    m.code_bss.toLowerCase().includes(filter.toLowerCase()) ||
    m.name.toLowerCase().includes(filter.toLowerCase()) ||
    m.recharge_type.toLowerCase().includes(filter.toLowerCase()) ||
    m.response_type.toLowerCase().includes(filter.toLowerCase())
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
    else { setSortKey(key); setSortAsc(false) }
  }

  const grouped = useMemo(() => {
    const map = new Map<string, PastasModelSummary[]>()
    for (const m of sorted) {
      const key = m.code_bss
      if (!map.has(key)) map.set(key, [])
      map.get(key)!.push(m)
    }
    return map
  }, [sorted])

  if (isLoading) return <div className="text-text-muted text-sm py-8 text-center">Loading...</div>

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            placeholder="Filter by station, name, config..."
            value={filter}
            onChange={e => setFilter(e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded-lg pl-9 pr-3 py-2 text-sm text-text-primary placeholder:text-text-muted"
          />
        </div>

        {/* Sort */}
        <div className="flex items-center gap-1 text-xs text-text-muted">
          <ArrowUpDown className="w-3.5 h-3.5" />
          {(['created_at', 'nse', 'evp', 'code_bss'] as SortKey[]).map(k => (
            <button
              key={k}
              onClick={() => toggleSort(k)}
              className={`px-2 py-1 rounded transition-colors ${
                sortKey === k ? 'bg-accent-cyan/10 text-accent-cyan' : 'hover:bg-bg-hover hover:text-text-secondary'
              }`}
            >
              {k === 'created_at' ? 'Date' : k === 'nse' ? 'NSE' : k === 'evp' ? 'EVP' : 'Station'}
              {sortKey === k && (sortAsc ? ' ↑' : ' ↓')}
            </button>
          ))}
        </div>

        {/* View toggle */}
        <div className="flex border border-white/10 rounded-lg overflow-hidden">
          <button onClick={() => setView('grid')} className={`p-1.5 ${view === 'grid' ? 'bg-accent-cyan/10 text-accent-cyan' : 'text-text-muted hover:text-text-secondary'}`}>
            <LayoutGrid className="w-4 h-4" />
          </button>
          <button onClick={() => setView('list')} className={`p-1.5 ${view === 'list' ? 'bg-accent-cyan/10 text-accent-cyan' : 'text-text-muted hover:text-text-secondary'}`}>
            <List className="w-4 h-4" />
          </button>
        </div>
      </div>

      {sorted.length === 0 ? (
        <div className="text-text-muted text-sm py-12 text-center">
          {filter ? 'No models match.' : 'No calibrated models. Go to the Fit tab to create one.'}
        </div>
      ) : view === 'grid' ? (
        <div>
          {Array.from(grouped.entries()).map(([station, models]) => {
            const stowaPassModels = models.filter(m => m.nse != null && m.nse >= 0.5 && m.evp != null && m.evp >= 70)
            const bestNse = stowaPassModels.length > 0
              ? stowaPassModels.reduce((best, m) => (m.nse ?? -Infinity) > (best.nse ?? -Infinity) ? m : best)
              : null
            return (
              <div key={station}>
                <div className="flex items-center gap-2 mb-2 mt-4 first:mt-0">
                  <a href={`/station/piezo/${encodeURIComponent(station)}`} className="text-sm font-mono text-accent-cyan hover:underline">{station}</a>
                  <span className="text-xs text-text-muted">{models.length} model(s)</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                  {models.map(m => (
                    <div key={m.run_id} className="relative">
                      {bestNse && m.run_id === bestNse.run_id && (
                        <span className="absolute -top-1.5 -right-1.5 z-10 text-sm" title="Best STOWA-passing model">★</span>
                      )}
                      <ModelCard
                        m={m}
                        onView={() => navigate(`/pastas/fit?model=${m.run_id}`)}
                        onDelete={() => { if (confirm('Delete this model?')) deleteMut.mutate(m.run_id) }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="bg-bg-card rounded-lg border border-white/5 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-text-muted border-b border-white/5 text-xs uppercase tracking-wide">
                  <th className="px-3 py-2 text-left">Name</th>
                  <th className="px-3 py-2 text-left">Station</th>
                  <th className="px-3 py-2 text-left">Config</th>
                  <th className="px-3 py-2 text-right">NSE train</th>
                  <th className="px-3 py-2 text-right">NSE test</th>
                  <th className="px-3 py-2 text-right">EVP %</th>
                  <th className="px-3 py-2 text-left">Date</th>
                  <th className="px-3 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map(m => (
                  <tr key={m.run_id} className="border-b border-white/5 hover:bg-bg-hover transition-colors">
                    <td className="px-3 py-2 text-text-primary font-medium">{m.name || m.run_id.slice(0, 8)}</td>
                    <td className="px-3 py-2 font-mono"><a href={`/station/piezo/${encodeURIComponent(m.code_bss)}`} className="text-accent-cyan hover:underline">{m.code_bss}</a></td>
                    <td className="px-3 py-2">
                      <div className="flex flex-wrap gap-0.5">
                        <ConfigTag label={m.recharge_type} color="border-blue-500/30 text-blue-400 bg-blue-500/10" />
                        <ConfigTag label={m.response_type} color="border-emerald-500/30 text-emerald-400 bg-emerald-500/10" />
                        {m.noise_type !== 'unknown' && m.noise_type !== 'none' && (
                          <ConfigTag label={m.noise_type.replace('NoiseModel', '').replace('Noise', '')} color="border-amber-500/30 text-amber-400 bg-amber-500/10" />
                        )}
                      </div>
                    </td>
                    <td className={`px-3 py-2 text-right font-mono ${nseColor(m.nse)}`}>{m.nse?.toFixed(3) ?? '—'}</td>
                    <td className={`px-3 py-2 text-right font-mono ${nseColor(m.val_nse)}`}>{m.val_nse?.toFixed(3) ?? '—'}</td>
                    <td className="px-3 py-2 text-right text-text-primary">{m.evp?.toFixed(1) ?? '—'}</td>
                    <td className="px-3 py-2 text-text-muted">{new Date(Number(m.created_at)).toLocaleDateString('en-GB')}</td>
                    <td className="px-3 py-2 text-right">
                      <div className="flex items-center justify-end gap-0.5">
                        <button onClick={() => navigate(`/pastas/fit?model=${m.run_id}`)} className="p-1 hover:bg-bg-hover rounded text-text-muted hover:text-accent-cyan transition-colors" title="View">
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                        <ExportMenu runId={m.run_id} />
                        <button onClick={() => { if (confirm('Delete this model?')) deleteMut.mutate(m.run_id) }} className="p-1 hover:bg-bg-hover rounded text-text-muted hover:text-red-400 transition-colors" title="Delete">
                          <Trash2 className="w-3.5 h-3.5" />
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
