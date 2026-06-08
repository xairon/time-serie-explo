import { Link } from 'react-router-dom'
import { MapPin } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { observatoryApi } from '@/lib/observatory-api'
import { formatDate } from '@/lib/observatory-utils'
import type { Alert } from '@/lib/observatory-types'

export function StationDrillTable({ codeDepartement, type }: { codeDepartement: string; type: 'piezo' | 'hydro' }) {
  const { t } = useTranslation()
  const { data, isLoading } = useQuery({
    queryKey: ['drill-alerts', codeDepartement, type],
    queryFn: () => observatoryApi.common.alerts({
      code_departement: codeDepartement, type,
      severity: ['EXTREMEMENT_BAS', 'TRES_BAS', 'BAS', 'HAUT', 'TRES_HAUT', 'EXTREMEMENT_HAUT'],
    }),
  })
  if (isLoading) return <div className="text-sm text-text-secondary p-4">…</div>
  const rows = (data ?? []) as Alert[]
  if (!rows.length) return <div className="text-sm text-text-secondary p-4">{t('cleanup.alerts.noStationsInAlert', { label: '' })}</div>
  return (
    <div className="overflow-x-auto bg-bg-card border border-white/5 rounded-xl">
      <table className="w-full text-sm min-w-[600px]">
        <thead><tr className="border-b border-white/5">
          <th className="px-4 py-2 text-left text-xs text-text-secondary">{t('cleanup.alerts.colCode')}</th>
          <th className="px-4 py-2 text-left text-xs text-text-secondary">{t('cleanup.alerts.colStation')}</th>
          <th className="px-4 py-2 text-left text-xs text-text-secondary">{t('cleanup.alerts.colLastMeasurement')}</th>
          <th className="w-10" />
        </tr></thead>
        <tbody>{rows.map(s => (
          <tr key={`${s.type}-${s.code}`} className="border-b border-white/5 hover:bg-bg-hover">
            <td className="px-4 py-2"><Link to={`/station/${s.type}/${s.code}`} className="text-accent-cyan hover:underline font-mono text-xs">{s.code}</Link></td>
            <td className="px-4 py-2 text-text-primary">{s.commune || s.code}</td>
            <td className="px-4 py-2 text-text-secondary text-xs">{formatDate(s.derniere_mesure)}</td>
            <td className="px-4 py-2">{s.latitude != null && s.longitude != null && (
              <Link to={`/?lat=${s.latitude}&lon=${s.longitude}&zoom=12`} className="inline-flex"><MapPin className="w-3.5 h-3.5 text-accent-cyan" /></Link>)}</td>
          </tr>))}
        </tbody>
      </table>
    </div>
  )
}
