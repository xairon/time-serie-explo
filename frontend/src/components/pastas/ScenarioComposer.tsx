import { useState } from 'react'
import { Plus, Droplets, TrendingUp, ArrowUpDown } from 'lucide-react'
import { ModificationCard, type ModificationData } from './ModificationCard'

interface ScenarioComposerProps {
  modifications: ModificationData[]
  onChange: (mods: ModificationData[]) => void
}

type AddMenuType = ModificationData['type']

const ADD_OPTIONS: { type: AddMenuType; label: string; icon: React.ElementType }[] = [
  { type: 'pumping_synthetic', label: 'Pumping (synthetic)', icon: Droplets },
  { type: 'pumping_upload', label: 'Pumping (CSV)', icon: Droplets },
  { type: 'linear_trend', label: 'Linear trend', icon: TrendingUp },
  { type: 'scale_stress', label: 'Scale stress', icon: ArrowUpDown },
]

function defaultForType(type: AddMenuType): ModificationData {
  switch (type) {
    case 'pumping_synthetic':
      return { type, pattern: 'constant', rate_m3d: 100, distance_m: 500, start: '', end: '', rfunc: 'Exponential' }
    case 'pumping_upload':
      return { type, rows: [], distance_m: 500, rfunc: 'Exponential' }
    case 'linear_trend':
      return { type, start: '', end: '', slope_m_per_year: -0.01 }
    case 'scale_stress':
      return { type, stress: 'precip', factor: 0.8, start: '', end: '' }
  }
}

export function ScenarioComposer({ modifications, onChange }: ScenarioComposerProps) {
  const [menuOpen, setMenuOpen] = useState(false)

  function addModification(type: AddMenuType) {
    onChange([...modifications, defaultForType(type)])
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
        <p className="text-xs text-text-muted text-center py-4">
          No modifications yet. Add one below.
        </p>
      )}

      {modifications.map((mod, i) => (
        <ModificationCard
          key={i}
          index={i}
          data={mod}
          onChange={(d) => updateModification(i, d)}
          onDelete={() => deleteModification(i)}
        />
      ))}

      {/* Add button + dropdown */}
      <div className="relative">
        <button
          onClick={() => setMenuOpen((v) => !v)}
          className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-text-secondary hover:text-text-primary border border-white/10 rounded-lg hover:border-white/20 transition-colors w-full justify-center"
        >
          <Plus className="w-3.5 h-3.5" />
          Add modification
        </button>

        {menuOpen && (
          <>
            <div
              className="fixed inset-0 z-10"
              onClick={() => setMenuOpen(false)}
            />
            <div className="absolute bottom-full mb-1 left-0 right-0 z-20 bg-bg-card border border-white/10 rounded-lg shadow-lg overflow-hidden">
              {ADD_OPTIONS.map(({ type, label, icon: Icon }) => (
                <button
                  key={type}
                  onClick={() => addModification(type)}
                  className="flex items-center gap-2 w-full px-3 py-2 text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors"
                >
                  <Icon className="w-3.5 h-3.5 text-text-muted" />
                  {label}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
