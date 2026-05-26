import { useState } from 'react'
import { Plus, Droplets, TrendingUp, ArrowUpDown } from 'lucide-react'
import { ModificationCard, type ModificationData } from './ModificationCard'
import type { PumpingProfileData, PumpingRange, AdaptiveBoundsData } from '@/lib/types'

interface ScenarioComposerProps {
  modifications: ModificationData[]
  onChange: (mods: ModificationData[]) => void
  tmin: string
  tmax: string
  pumpingProfiles?: Record<string, PumpingProfileData> | null
  scaleStressLimits?: PumpingRange | null
  linearTrendLimits?: PumpingRange | null
  adaptiveBounds?: AdaptiveBoundsData | null
}

type AddMenuType = ModificationData['type']

const ADD_OPTIONS: { type: AddMenuType; label: string; description: string; icon: React.ElementType }[] = [
  { type: 'pumping_synthetic', label: 'Ajouter un forage', description: 'Simuler un forage de pompage avec débit et calendrier configurables', icon: Droplets },
  { type: 'pumping_upload', label: 'Importer des données de pompage', description: 'Importer une chronique de pompage réelle depuis un CSV', icon: Droplets },
  { type: 'linear_trend', label: 'Ajouter une tendance', description: 'Simuler une hausse ou baisse progressive du niveau au cours du temps', icon: TrendingUp },
  { type: 'scale_stress', label: 'Ajuster le climat', description: 'Modifier les précipitations ou l\'évapotranspiration (ex. -20% de précipitations pour sécheresse)', icon: ArrowUpDown },
]

function defaultForType(type: AddMenuType, tmin: string, tmax: string, profile?: PumpingProfileData | null): ModificationData {
  switch (type) {
    case 'pumping_synthetic':
      return {
        type,
        usage: 'aep',
        pattern: (profile?.pattern as 'constant' | 'seasonal' | 'pulse') ?? 'constant',
        rate_m3d: profile?.rate_m3d.default ?? 100,
        distance_m: profile?.distance_m.default ?? 500,
        start: tmin,
        end: tmax,
        rfunc: (profile?.rfunc as 'Exponential' | 'Hantush') ?? 'Exponential',
        season_months: profile?.active_months ?? [5, 6, 7, 8, 9],
        peak_months: profile?.peak_months,
        peak_factor: profile?.peak_factor,
      }
    case 'pumping_upload':
      return { type, rows: [], distance_m: profile?.distance_m.default ?? 500, rfunc: 'Exponential' }
    case 'linear_trend':
      return { type, start: tmin, end: tmax, slope_m_per_year: -0.01 }
    case 'scale_stress':
      return { type, stress: 'precip', factor: 0.8, start: tmin, end: tmax }
  }
}

export function ScenarioComposer({ modifications, onChange, tmin, tmax, pumpingProfiles, scaleStressLimits, linearTrendLimits, adaptiveBounds }: ScenarioComposerProps) {
  const [menuOpen, setMenuOpen] = useState(false)

  function addModification(type: AddMenuType) {
    onChange([...modifications, defaultForType(type, tmin, tmax, pumpingProfiles?.['aep'])])
    setMenuOpen(false)
  }

  function updateModification(index: number, data: ModificationData) {
    const next = [...modifications]
    next[index] = data
    onChange(next)
  }

  function deleteModification(index: number) {
    onChange(modifications.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-3">
      {modifications.length === 0 && (
        <p className="text-xs text-text-muted text-center py-2">
          Aucune modification pour le moment. Ajoutez-en une ci-dessous.
        </p>
      )}

      {modifications.map((mod, i) => (
        <ModificationCard
          key={i}
          index={i}
          data={mod}
          onChange={(d) => updateModification(i, d)}
          onDelete={() => deleteModification(i)}
          pumpingProfiles={pumpingProfiles}
          scaleStressLimits={scaleStressLimits}
          linearTrendLimits={linearTrendLimits}
          adaptiveBounds={adaptiveBounds}
        />
      ))}

      {/* Add button + dropdown */}
      <div className="relative">
        <button
          onClick={() => setMenuOpen((v) => !v)}
          className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-text-secondary hover:text-text-primary border border-white/10 rounded-lg hover:border-white/20 transition-colors w-full justify-center"
        >
          <Plus className="w-3.5 h-3.5" />
          Ajouter une modification
        </button>

        {menuOpen && (
          <>
            <div className="fixed inset-0 z-10" onClick={() => setMenuOpen(false)} />
            <div className="absolute bottom-full mb-1 left-0 right-0 z-20 bg-bg-card border border-white/10 rounded-lg shadow-lg overflow-hidden">
              {ADD_OPTIONS.map(({ type, label, description, icon: Icon }) => (
                <button
                  key={type}
                  onClick={() => addModification(type)}
                  className="flex items-start gap-2.5 w-full px-3 py-2.5 text-left hover:bg-bg-hover transition-colors"
                >
                  <Icon className="w-4 h-4 text-text-muted mt-0.5 shrink-0" />
                  <div>
                    <div className="text-xs font-medium text-text-secondary">{label}</div>
                    <div className="text-[10px] text-text-muted">{description}</div>
                  </div>
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
