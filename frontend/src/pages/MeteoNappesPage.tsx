import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { situationApi } from '@/lib/situation-api'
import { NationalBanner } from '@/components/meteo/NationalBanner'
import { TerritoryChoropleth } from '@/components/meteo/TerritoryChoropleth'
import { TerritoryRanking } from '@/components/meteo/TerritoryRanking'
import { OutlookPanel } from '@/components/meteo/OutlookPanel'
import { StationDrillTable } from '@/components/meteo/StationDrillTable'

export default function MeteoNappesPage() {
  const { t } = useTranslation()
  const [type, setType] = useState<'piezo' | 'hydro'>('piezo')
  const [region, setRegion] = useState<string | null>(null)
  const [dept, setDept] = useState<string | null>(null)

  const national = useQuery({ queryKey: ['sit-national', type], queryFn: () => situationApi.national(type) })
  const regions = useQuery({ queryKey: ['sit-territories', 'region', type], queryFn: () => situationApi.territories('region', type) })
  const departments = useQuery({ queryKey: ['sit-territories', 'department', type], queryFn: () => situationApi.territories('department', type) })

  const level: 'region' | 'department' = region ? 'department' : 'region'
  const shownTerritories = region ? (departments.data ?? []) : (regions.data ?? [])
  const deptsInAlert = (departments.data ?? []).filter(
    d => d.situation_class === 'EXTREMEMENT_BAS' || d.situation_class === 'TRES_BAS',
  ).length
  const deptName = departments.data?.find(d => d.code === dept)?.name ?? dept ?? ''

  if (national.isError || regions.isError) {
    return <div className="h-full flex items-center justify-center text-red-400 text-sm">{t('meteo.loadError')}</div>
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-6 space-y-5">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-xl font-bold text-text-primary">{t('meteo.title')}</h1>
            <p className="text-xs text-text-secondary">{t('meteo.subtitle')}</p>
          </div>
          <div className="flex gap-1 bg-bg-card border border-white/10 rounded-lg p-0.5">
            {(['piezo', 'hydro'] as const).map(ty => (
              <button key={ty} onClick={() => { setType(ty); setRegion(null); setDept(null) }}
                className={`px-3 py-1.5 rounded-md text-xs ${type === ty ? 'bg-accent-cyan/20 text-accent-cyan' : 'text-text-secondary'}`}>
                {t(ty === 'piezo' ? 'meteo.tabPiezo' : 'meteo.tabHydro')}
              </button>
            ))}
          </div>
        </div>

        {national.data && <NationalBanner data={national.data} departmentsInAlert={deptsInAlert} />}

        {region && (
          <button onClick={() => { setRegion(null); setDept(null) }} className="text-xs text-accent-cyan">{t('meteo.backToRegions')}</button>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2">
            <TerritoryChoropleth
              level={level}
              territories={shownTerritories}
              onSelectRegion={(code) => { setRegion(code); setDept(null) }}
              onSelectDepartment={(code) => setDept(code)}
            />
          </div>
          <div className="space-y-4">
            <TerritoryRanking
              territories={shownTerritories}
              onSelect={(code) => (region ? setDept(code) : setRegion(code))}
            />
            <OutlookPanel outlook={national.data?.outlook ?? null} />
          </div>
        </div>

        {dept && (
          <div className="space-y-2">
            <h2 className="text-sm font-semibold text-text-primary">{t('meteo.stationsIn', { name: deptName })}</h2>
            <StationDrillTable codeDepartement={dept} type={type} />
          </div>
        )}
      </div>
    </div>
  )
}
