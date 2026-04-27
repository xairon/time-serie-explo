import { useState, useMemo, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { Download, AlertTriangle, MapPin } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { observatoryApi } from '@/lib/observatory-api'
import { CLASSIFICATION_COLORS } from '@/lib/observatory-constants'
import { formatDate } from '@/lib/observatory-utils'
import type { Alert } from '@/lib/observatory-types'

const PAGE_SIZE = 50

const escapeCSV = (v: string | number | null | undefined): string => {
  const s = String(v ?? ''); const needsQuote = s.includes(',') || s.includes('"') || s.includes('\n') || /^[=+\-@]/.test(s)
  return needsQuote ? `"${s.replace(/"/g, '""')}"` : s
}

type SortKey = 'name' | 'dept' | 'derniere_mesure' | 'depuis'
type SortDir = 'asc' | 'desc'

function SortIndicator({ column, sortKey, sortDir }: { column: SortKey; sortKey: SortKey | null; sortDir: SortDir }) {
  if (sortKey !== column) return <span className="text-text-secondary/30 ml-1">&#x25B2;</span>
  return sortDir === 'asc' ? <span className="text-accent-cyan ml-1">&#x25B2;</span> : <span className="text-accent-cyan ml-1">&#x25BC;</span>
}

const SEVERITY_TABS = [
  { key: 'EXTREMEMENT_BAS', label: 'Extr. low', description: 'Extremely low level (< -1.75s) -- exceptional situation' },
  { key: 'TRES_BAS', label: 'Very low', description: 'Very low level (-1.75 to -1.28s) -- potential intervention' },
  { key: 'BAS', label: 'Low', description: 'Low level (-1.28 to -0.84s) -- monitoring' },
  { key: 'HAUT', label: 'High', description: 'High level (0.84 to 1.28s) -- above normal' },
  { key: 'TRES_HAUT', label: 'Very high', description: 'Very high level (1.28 to 1.75s) -- flood risk' },
  { key: 'EXTREMEMENT_HAUT', label: 'Extr. high', description: 'Extremely high level (> 1.75s) -- exceptional situation' },
] as const

export default function AlertsPage() {
  const [activeTab, setActiveTab] = useState<string>('TRES_BAS')
  const [page, setPage] = useState(0)
  const [sortKey, setSortKey] = useState<SortKey | null>(null)
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  const { data: alerts, isLoading, isError } = useQuery({
    queryKey: ['obs-alerts', 'all'],
    queryFn: () => observatoryApi.common.alerts({ severity: ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT'] }),
  })

  const handleSort = useCallback((key: SortKey) => { if (sortKey === key) setSortDir(prev => prev === 'asc' ? 'desc' : 'asc'); else { setSortKey(key); setSortDir('asc') }; setPage(0) }, [sortKey])

  const grouped = useMemo(() => {
    const groups: Record<string, Array<{ code: string; name: string; dept: string | null; classification: string | null; derniere_mesure: string | null; type: string; lat: number | null; lon: number | null; alerte_depuis_annee: number | null; nb_annees_consecutives: number | null }>> = {}
    SEVERITY_TABS.forEach(t => { groups[t.key] = [] })
    ;(alerts ?? []).forEach((a: Alert) => { const cls = a.classification ?? ''; if (!groups[cls]) groups[cls] = []; groups[cls].push({ code: a.code, name: a.commune || a.code, dept: a.departement ?? a.code_departement, classification: a.classification, derniere_mesure: a.derniere_mesure, type: a.type, lat: a.latitude, lon: a.longitude, alerte_depuis_annee: a.alerte_depuis_annee, nb_annees_consecutives: a.nb_annees_consecutives }) })
    return groups
  }, [alerts])

  const currentStations = useMemo(() => {
    let stations = [...(grouped[activeTab] ?? [])]
    if (sortKey) stations.sort((a: any, b: any) => { let valA: any, valB: any; switch (sortKey) { case 'name': valA = (a.name ?? '').toLowerCase(); valB = (b.name ?? '').toLowerCase(); break; case 'dept': valA = (a.dept ?? '').toLowerCase(); valB = (b.dept ?? '').toLowerCase(); break; case 'derniere_mesure': valA = a.derniere_mesure ?? ''; valB = b.derniere_mesure ?? ''; break; case 'depuis': valA = a.nb_annees_consecutives ?? 0; valB = b.nb_annees_consecutives ?? 0; break; default: return 0 }; if (valA < valB) return sortDir === 'asc' ? -1 : 1; if (valA > valB) return sortDir === 'asc' ? 1 : -1; return 0 })
    return stations
  }, [grouped, activeTab, sortKey, sortDir])

  const paged = currentStations.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const totalPages = Math.ceil(currentStations.length / PAGE_SIZE)
  const totalAlerts = (alerts ?? []).length

  const exportCSV = () => {
    const allStations = Object.values(grouped).flat()
    const header = ['Code', 'Name', 'Department', 'Classification', 'Last measurement', 'Since', 'Consecutive years', 'Type'].map(escapeCSV).join(',') + '\n'
    const rows = allStations.map(s => [s.code, s.name, s.dept, s.classification, s.derniere_mesure, s.alerte_depuis_annee, s.nb_annees_consecutives, s.type].map(escapeCSV).join(',')).join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'station_alerts.csv'; a.click(); URL.revokeObjectURL(url)
  }

  if (isError) return (<div className="h-full flex items-center justify-center"><p className="text-red-400 text-sm" role="alert">Error loading data.</p></div>)

  const activeTabConf = SEVERITY_TABS.find(t => t.key === activeTab)!

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-6 space-y-5">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <div><h1 className="text-xl font-bold text-text-primary">Alerts</h1><p className="text-xs text-text-secondary">Active stations (measurements &lt; 90 days) in abnormal conditions</p></div>
            {!isLoading && <span className="text-sm font-mono text-accent-cyan bg-accent-cyan/10 px-2 py-0.5 rounded-full">{totalAlerts}</span>}
          </div>
          <button onClick={exportCSV} aria-label="Export CSV" className="flex items-center gap-1.5 px-3 py-1.5 bg-bg-card border border-white/10 rounded-lg text-xs text-text-secondary hover:text-text-primary transition-colors"><Download className="w-3.5 h-3.5" /> CSV</button>
        </div>

        <div className="flex gap-2 flex-wrap">
          {SEVERITY_TABS.map(tab => { const count = grouped[tab.key]?.length ?? 0; const color = CLASSIFICATION_COLORS[tab.key]; const isActive = activeTab === tab.key; return (<button key={tab.key} onClick={() => { setActiveTab(tab.key); setPage(0); setSortKey(null) }} className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-sm font-medium transition-all border" style={{ backgroundColor: isActive ? `${color}15` : 'transparent', borderColor: isActive ? color : 'rgba(255,255,255,0.08)', color: isActive ? color : '#9ca3af' }}><span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />{tab.label}<span className="font-mono text-xs px-1.5 py-0.5 rounded-md" style={{ backgroundColor: isActive ? `${color}20` : 'rgba(255,255,255,0.05)' }}>{count}</span></button>) })}
        </div>

        <p className="text-xs text-text-secondary">{activeTabConf.description}</p>

        <div className="bg-bg-card border border-white/5 rounded-xl overflow-hidden">
          {isLoading ? (<div className="animate-pulse"><div className="flex border-b border-white/5 px-4 py-3 gap-4">{Array.from({ length: 6 }).map((_, i) => (<div key={i} className="h-3 bg-white/10 rounded" style={{ width: `${10 + i * 3}%` }} />))}</div>{Array.from({ length: 8 }).map((_, i) => (<div key={i} className="flex border-b border-white/5 px-4 py-3 gap-4 items-center"><div className="h-4 w-10 bg-white/5 rounded" /><div className="h-3 w-28 bg-white/10 rounded" /><div className="h-3 bg-white/5 rounded flex-1" /><div className="h-3 w-20 bg-white/5 rounded" /><div className="h-3 w-16 bg-white/5 rounded" /></div>))}</div>) : currentStations.length === 0 ? (<div className="flex items-center justify-center py-16 text-text-secondary text-sm">No stations in "{activeTabConf.label}" alert</div>) : (<>
            <div className="overflow-x-auto"><table className="w-full text-sm min-w-[700px]"><thead><tr className="border-b border-white/5"><th className="px-4 py-3 text-left text-xs font-medium text-text-secondary w-16">Type</th><th className="px-4 py-3 text-left text-xs font-medium text-text-secondary">Code</th><th className="px-4 py-3 text-left text-xs font-medium text-text-secondary cursor-pointer select-none hover:text-text-primary" onClick={() => handleSort('name')}>Station <SortIndicator column="name" sortKey={sortKey} sortDir={sortDir} /></th><th className="px-4 py-3 text-left text-xs font-medium text-text-secondary cursor-pointer select-none hover:text-text-primary" onClick={() => handleSort('dept')}>Department <SortIndicator column="dept" sortKey={sortKey} sortDir={sortDir} /></th><th className="px-4 py-3 text-left text-xs font-medium text-text-secondary cursor-pointer select-none hover:text-text-primary" onClick={() => handleSort('derniere_mesure')}>Last measurement <SortIndicator column="derniere_mesure" sortKey={sortKey} sortDir={sortDir} /></th><th className="px-4 py-3 text-left text-xs font-medium text-text-secondary cursor-pointer select-none hover:text-text-primary" onClick={() => handleSort('depuis')}>Since <SortIndicator column="depuis" sortKey={sortKey} sortDir={sortDir} /></th><th className="px-4 py-3 text-left text-xs font-medium text-text-secondary w-12"></th></tr></thead><tbody>{paged.map((s) => (<tr key={`${s.type}-${s.code}`} className="border-b border-white/5 hover:bg-bg-hover transition-colors"><td className="px-4 py-2.5"><span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${s.type === 'piezo' ? 'bg-accent-cyan/20 text-accent-cyan' : 'bg-accent-indigo/20 text-accent-indigo'}`}>{s.type === 'piezo' ? 'PIEZO' : 'HYDRO'}</span></td><td className="px-4 py-2.5"><Link to={`/station/${s.type}/${s.code}`} className="text-accent-cyan hover:underline font-mono text-xs">{s.code}</Link></td><td className="px-4 py-2.5 text-text-primary">{s.name}</td><td className="px-4 py-2.5 text-text-secondary text-xs">{s.dept}</td><td className="px-4 py-2.5 text-text-secondary text-xs">{formatDate(s.derniere_mesure)}</td><td className="px-4 py-2.5 text-text-secondary text-xs">{s.alerte_depuis_annee != null && (<span title={`${s.nb_annees_consecutives} consecutive year${(s.nb_annees_consecutives ?? 0) > 1 ? 's' : ''}`}>{s.alerte_depuis_annee}{(s.nb_annees_consecutives ?? 0) > 1 && <span className="ml-1 text-[10px] text-text-secondary/60">({s.nb_annees_consecutives} yrs)</span>}</span>)}</td><td className="px-4 py-2.5">{s.lat != null && s.lon != null && (<Link to={`/observatory?lat=${s.lat}&lon=${s.lon}&zoom=12`} className="p-1 hover:bg-bg-hover rounded inline-flex items-center" title="View on map"><MapPin className="w-3.5 h-3.5 text-accent-cyan" /></Link>)}</td></tr>))}</tbody></table></div>
            {totalPages > 1 && (<div className="flex items-center justify-between px-4 py-3 border-t border-white/5"><p className="text-xs text-text-secondary">{currentStations.length.toLocaleString()} station{currentStations.length > 1 ? 's' : ''}</p><div className="flex gap-1"><button onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0} className="px-2.5 py-1 rounded text-xs text-text-secondary hover:text-text-primary disabled:opacity-30">Prev</button><span className="px-2.5 py-1 text-xs text-text-secondary">{page + 1} / {totalPages}</span><button onClick={() => setPage(Math.min(totalPages - 1, page + 1))} disabled={page >= totalPages - 1} className="px-2.5 py-1 rounded text-xs text-text-secondary hover:text-text-primary disabled:opacity-30">Next</button></div></div>)}
          </>)}
        </div>
      </div>
    </div>
  )
}
