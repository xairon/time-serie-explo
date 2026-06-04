import { NavLink, Outlet } from 'react-router-dom'
import { SlidersHorizontal, BarChart3, FlaskConical, LayoutGrid } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { GuidedExpertToggle } from '@/components/pastas/GuidedExpertToggle'
import { useState, createContext, useContext, useEffect, useCallback } from 'react'
import type { AutoFitResult, StowaResult } from '@/lib/types'

interface PipelineState {
  codeBss: string
  autoFitResult: AutoFitResult | null
  selectedRunId: string | null
  selectedStowa: StowaResult | null
}

interface PastasModeContextValue {
  mode: 'guided' | 'expert'
  setMode: (m: 'guided' | 'expert') => void
  pipeline: PipelineState
  setCodeBss: (code: string) => void
  setAutoFitResult: (result: AutoFitResult | null) => void
  selectModel: (runId: string, stowa?: StowaResult | null) => void
  resetPipeline: () => void
}

const EMPTY_PIPELINE: PipelineState = {
  codeBss: '',
  autoFitResult: null,
  selectedRunId: null,
  selectedStowa: null,
}

export const PastasModeContext = createContext<PastasModeContextValue>({
  mode: 'guided',
  setMode: () => {},
  pipeline: EMPTY_PIPELINE,
  setCodeBss: () => {},
  setAutoFitResult: () => {},
  selectModel: () => {},
  resetPipeline: () => {},
})

export function usePastasMode() {
  return useContext(PastasModeContext)
}

const STORAGE_KEY = 'pastas_mode'

const TABS = [
  { to: '/pastas/calibrate', icon: SlidersHorizontal, labelKey: 'pastas.tabs.calibrate' },
  { to: '/pastas/results', icon: BarChart3, labelKey: 'pastas.tabs.results' },
  { to: '/pastas/scenarios', icon: FlaskConical, labelKey: 'pastas.tabs.scenarios' },
  { to: '/pastas/gallery', icon: LayoutGrid, labelKey: 'pastas.tabs.gallery' },
] as const

function readStoredMode(): 'guided' | 'expert' {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'guided' || v === 'expert') return v
  } catch { /* ignore */ }
  return 'guided'
}

export default function PastasLayout() {
  const { t } = useTranslation()
  const [mode, setModeState] = useState<'guided' | 'expert'>(readStoredMode)
  const [pipeline, setPipeline] = useState<PipelineState>(EMPTY_PIPELINE)

  function setMode(m: 'guided' | 'expert') {
    setModeState(m)
    try {
      localStorage.setItem(STORAGE_KEY, m)
    } catch { /* ignore */ }
  }

  const setCodeBss = useCallback((code: string) => {
    setPipeline(prev => prev.codeBss === code ? prev : {
      ...EMPTY_PIPELINE,
      codeBss: code,
    })
  }, [])

  const setAutoFitResult = useCallback((result: AutoFitResult | null) => {
    setPipeline(prev => ({ ...prev, autoFitResult: result }))
  }, [])

  const selectModel = useCallback((runId: string, stowa?: StowaResult | null) => {
    setPipeline(prev => ({ ...prev, selectedRunId: runId, selectedStowa: stowa ?? null }))
  }, [])

  const resetPipeline = useCallback(() => {
    setPipeline(EMPTY_PIPELINE)
  }, [])

  // Sync across tabs
  useEffect(() => {
    function onStorage(e: StorageEvent) {
      if (e.key === STORAGE_KEY && (e.newValue === 'guided' || e.newValue === 'expert')) {
        setModeState(e.newValue)
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  const contextValue: PastasModeContextValue = {
    mode, setMode, pipeline, setCodeBss, setAutoFitResult, selectModel, resetPipeline,
  }

  return (
    <PastasModeContext.Provider value={contextValue}>
      <div className="flex flex-col h-full">
        <div className="bg-bg-card border-b border-white/5 shrink-0">
          <div className="flex items-center justify-between px-4">
            <div className="flex items-center gap-1">
              {TABS.map(({ to, icon: Icon, labelKey }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 ${
                      isActive
                        ? 'border-accent-cyan text-text-primary'
                        : 'border-transparent text-text-muted hover:text-text-secondary'
                    }`
                  }
                >
                  <Icon className="w-4 h-4" />
                  {t(labelKey)}
                </NavLink>
              ))}
            </div>
            <GuidedExpertToggle mode={mode} onChange={setMode} />
          </div>
        </div>
        <div className="flex-1 min-h-0 overflow-auto">
          <Outlet />
        </div>
      </div>
    </PastasModeContext.Provider>
  )
}
