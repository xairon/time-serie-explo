import { useState, useMemo } from 'react'
import { Loader2, Play, Plus, FlaskConical, AlertTriangle, ChevronDown, Save, FolderOpen, Trash2 } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { usePastasSimulate } from '@/hooks/usePastas'
import { useScenarioPresets } from '@/hooks/useScenarioPresets'
import { useAdaptiveBounds } from '@/hooks/useAdaptiveBounds'
import { useSavedScenarios, useSaveScenario, useDeleteScenario } from '@/hooks/useSavedScenarios'
import { ScenarioComposer } from './ScenarioComposer'
import { ScenarioResultsPanel } from './ScenarioResultsPanel'
import type { PastasFitResponse, PastasScenarioResponse, AquiferFamily, PumpingProfileData } from '@/lib/types'
import type { ModificationData } from './ModificationCard'
import { OnboardingBanner } from './OnboardingBanner'
import { api } from '@/lib/api'

function modificationToPayload(mod: ModificationData): Record<string, unknown> {
  if (mod.type === 'pumping_upload') {
    return { type: mod.type, csv_rows: mod.rows, distance_m: mod.distance_m, rfunc: mod.rfunc }
  }
  if (mod.type === 'pumping_synthetic') {
    const payload: Record<string, unknown> = { ...mod }
    if (!mod.season_months) delete payload.season_months
    if (!mod.pulse_duration_days) delete payload.pulse_duration_days
    return payload
  }
  return mod as unknown as Record<string, unknown>
}

interface Props {
  model: PastasFitResponse
  codeBss: string
  onRefit?: (result: PastasFitResponse) => void
}

function detectAquiferFamily(metadata: Record<string, unknown>): AquiferFamily {
  const nature = String(metadata.nature_eh ?? '')
  const milieu = String(metadata.milieu_eh ?? '')
  if (nature === '3') return 'alluvial'
  if (nature === '4') return 'karst'
  if (nature === '6' || nature === '7') return 'volcanic'
  if (nature === '0') return 'fractured'
  if (nature === '5' && milieu === '2') return 'fractured'
  if (nature === '5') return 'sedimentary'
  if (milieu === '5') return 'alluvial'
  if (milieu === '3') return 'karst'
  return 'sedimentary'
}

export function ScenarioWorkflow({ model, codeBss }: Props) {
  const simulateMut = usePastasSimulate()

  const [modifications, setModifications] = useState<ModificationData[]>([])
  const [simResult, setSimResult] = useState<PastasScenarioResponse | null>(null)
  const [showConfig, setShowConfig] = useState(true)
  const [saveDialogOpen, setSaveDialogOpen] = useState(false)
  const [saveName, setSaveName] = useState('')
  const obsTmin = model.observed?.index?.[0]?.slice(0, 10) ?? ''
  const obsTmax = model.observed?.index?.[model.observed.index.length - 1]?.slice(0, 10) ?? ''
  const [tmin, setTmin] = useState('')
  const [tmax, setTmax] = useState('')
  const effectiveTmin = tmin || obsTmin
  const effectiveTmax = tmax || obsTmax

  // Auto-detect aquifer family — block presets until detection completes
  const { data: stationPreview, isLoading: previewLoading } = useQuery({
    queryKey: ['pastas', 'preview', codeBss],
    queryFn: () => api.pastas.preview(codeBss),
    enabled: !!codeBss,
    staleTime: 30 * 60 * 1000,
  })

  const aquiferFamily: AquiferFamily = stationPreview?.metadata
    ? detectAquiferFamily(stationPreview.metadata)
    : 'sedimentary'
  const aquiferDetected = !!stationPreview?.metadata

  // Load presets referential only after aquifer detection
  const { data: presetsData } = useScenarioPresets(aquiferDetected ? {
    aquifer_family: aquiferFamily,
    tmin: effectiveTmin || undefined,
    tmax: effectiveTmax || undefined,
  } : undefined)

  // Pumping profiles for the detected aquifer family
  const pumpingProfiles = useMemo(() => {
    if (!presetsData) return null
    const result: Record<string, PumpingProfileData> = {}
    for (const usage of ['aep', 'irrigation', 'industrial'] as const) {
      const p = presetsData.pumping_profiles[usage]?.[aquiferFamily]
      if (p) result[usage] = p
    }
    return Object.keys(result).length > 0 ? result : null
  }, [presetsData, aquiferFamily])

  // Adaptive bounds from calibrated model
  const tFinalDays = useMemo(() => {
    if (!effectiveTmin || !effectiveTmax) return null
    const d0 = new Date(effectiveTmin)
    const d1 = new Date(effectiveTmax)
    const days = Math.round((d1.getTime() - d0.getTime()) / 86400000)
    return days > 0 ? days : null
  }, [effectiveTmin, effectiveTmax])
  const { data: adaptiveBounds } = useAdaptiveBounds(model.run_id || null, tFinalDays)

  // Saved scenarios
  const { data: savedScenarios = [] } = useSavedScenarios(model.run_id || null)
  const saveScenarioMut = useSaveScenario()
  const deleteScenarioMut = useDeleteScenario()

  function applyPreset(preset: { modifications: Record<string, unknown>[] }) {
    const mods = preset.modifications.map(m => {
      const mod = { ...m } as Record<string, unknown>
      if ('start' in mod && !mod.start && effectiveTmin) mod.start = effectiveTmin
      if ('end' in mod && !mod.end && effectiveTmax) mod.end = effectiveTmax
      return mod as unknown as ModificationData
    })
    setModifications(mods)
    setSimResult(null)
  }

  async function handleSave() {
    if (!model.run_id || !saveName.trim()) return
    await saveScenarioMut.mutateAsync({
      runId: model.run_id,
      body: {
        name: saveName.trim(),
        modifications: modifications.map(modificationToPayload),
        tmin: effectiveTmin || undefined,
        tmax: effectiveTmax || undefined,
      },
    })
    setSaveDialogOpen(false)
    setSaveName('')
  }

  async function handleSimulate() {
    if (!effectiveTmin || !effectiveTmax || modifications.length === 0) return
    try {
      const res = await simulateMut.mutateAsync({
        run_id: model.run_id,
        tmin: effectiveTmin,
        tmax: effectiveTmax,
        modifications: modifications.map(modificationToPayload),
      })
      setSimResult(res)
      setShowConfig(false)
    } catch { /* mutation handles error */ }
  }

  const inputClass = 'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-1.5 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

  return (
    <div className="space-y-4">
      <OnboardingBanner
        id="scenarios"
        title="What-if Scenarios"
        description="Test how external changes would affect the groundwater level. The calibrated model parameters stay frozen — only the input stresses are modified."
        steps={[
          'Add one or more stress modifications (pumping, climate change, trend…)',
          'Adjust the simulation window if needed',
          'Click "Run Simulation" to compute the impact',
        ]}
      />

      {/* Config panel — collapsible after first simulation */}
      <div className="bg-bg-card border border-white/5 rounded-xl overflow-hidden">
        <button
          onClick={() => setShowConfig(!showConfig)}
          className="w-full flex items-center justify-between px-4 py-3 hover:bg-bg-hover transition-colors"
        >
          <div className="flex items-center gap-3">
            <FlaskConical className="w-4 h-4 text-accent-cyan" />
            <div className="text-left">
              <div className="text-sm font-medium text-text-primary">
                Scenario configuration
                {modifications.length > 0 && (
                  <span className="ml-2 text-[10px] font-normal text-text-muted">
                    {modifications.length} modification{modifications.length > 1 ? 's' : ''}
                  </span>
                )}
              </div>
              <div className="text-[10px] text-text-muted">
                Base model: <span className="font-mono text-accent-cyan">{codeBss}</span>
                {model.metrics?.nse != null && <span className="ml-2">NSE {model.metrics.nse.toFixed(3)}</span>}
                {' · '}Window: {effectiveTmin} → {effectiveTmax}
              </div>
            </div>
          </div>
          <ChevronDown className={`w-4 h-4 text-text-muted transition-transform ${showConfig ? '' : '-rotate-90'}`} />
        </button>

        {showConfig && (
          <div className="px-4 pb-4 space-y-4 border-t border-white/5">
            {/* Simulation window */}
            <div className="pt-3">
              <div className="text-xs font-semibold text-text-secondary mb-2">Simulation window</div>
              <div className="grid grid-cols-2 gap-3 max-w-md">
                <div>
                  <label className="block text-[10px] text-text-muted mb-1">Start</label>
                  <input type="date" value={tmin} onChange={e => setTmin(e.target.value)} placeholder={obsTmin} className={inputClass} />
                </div>
                <div>
                  <label className="block text-[10px] text-text-muted mb-1">End</label>
                  <input type="date" value={tmax} onChange={e => setTmax(e.target.value)} placeholder={obsTmax} className={inputClass} />
                </div>
              </div>
              <p className="text-[10px] text-text-muted mt-1">Leave empty to use the full observation period ({obsTmin} → {obsTmax})</p>
            </div>

            {/* Contextual presets */}
            {previewLoading && (
              <div className="text-xs text-text-muted py-2">Loading aquifer profile...</div>
            )}
            {presetsData && (
              <div>
                <div className="text-xs font-semibold text-text-secondary mb-2">
                  Quick scenarios
                  {presetsData.aquifer_families[aquiferFamily] && (
                    <span className="ml-2 font-normal text-text-muted">({presetsData.aquifer_families[aquiferFamily]})</span>
                  )}
                </div>
                <div className="grid grid-cols-3 gap-1.5">
                  {presetsData.presets.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => applyPreset(p)}
                      className="text-left px-3 py-2 rounded-lg border border-white/5 hover:border-accent-cyan/20 hover:bg-accent-cyan/5 transition-colors group"
                    >
                      <div className="text-[10px] font-medium text-text-secondary group-hover:text-text-primary">{p.name}</div>
                      <div className="text-[9px] text-text-muted leading-tight">{p.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Modifications */}
            <div>
              <div className="text-xs font-semibold text-text-secondary mb-2">Stress modifications</div>
              <ScenarioComposer
                modifications={modifications}
                onChange={m => { setModifications(m); setSimResult(null) }}
                tmin={effectiveTmin}
                tmax={effectiveTmax}
                pumpingProfiles={pumpingProfiles}
                scaleStressLimits={presetsData?.non_pumping_limits?.scale_stress ?? null}
                linearTrendLimits={presetsData?.non_pumping_limits?.linear_trend ?? null}
                adaptiveBounds={adaptiveBounds ?? null}
              />
            </div>

            {/* Run + Save buttons */}
            <div className="flex items-center gap-3">
              <button
                onClick={handleSimulate}
                disabled={modifications.length === 0 || simulateMut.isPending}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {simulateMut.isPending
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Simulating…</>
                  : <><Play className="w-4 h-4" /> Run Simulation</>}
              </button>
              {modifications.length > 0 && (
                <button
                  onClick={() => setSaveDialogOpen(true)}
                  className="flex items-center gap-1.5 px-3 py-2.5 rounded-lg bg-white/5 text-text-secondary text-sm border border-white/10 hover:border-white/20 transition-colors"
                >
                  <Save className="w-4 h-4" /> Save
                </button>
              )}
              {modifications.length === 0 && (
                <span className="text-xs text-text-muted flex items-center gap-1">
                  <Plus className="w-3 h-3" /> Add at least one modification to simulate
                </span>
              )}
            </div>

            {/* Save dialog */}
            {saveDialogOpen && (
              <div className="bg-bg-primary border border-white/10 rounded-lg p-3 space-y-2">
                <input
                  type="text"
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  placeholder="Scenario name..."
                  className={inputClass}
                  autoFocus
                  onKeyDown={(e) => e.key === 'Enter' && handleSave()}
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleSave}
                    disabled={!saveName.trim() || saveScenarioMut.isPending}
                    className="flex-1 px-3 py-1.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-xs font-medium border border-accent-cyan/30 disabled:opacity-40"
                  >
                    {saveScenarioMut.isPending ? 'Saving…' : 'Save'}
                  </button>
                  <button onClick={() => setSaveDialogOpen(false)} className="px-3 py-1.5 rounded-lg text-text-muted text-xs border border-white/10">
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {/* Saved scenarios */}
            {savedScenarios.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-text-secondary mb-2">Saved scenarios</div>
                <div className="space-y-1">
                  {savedScenarios.map((s) => (
                    <div key={s.name} className="flex items-center justify-between px-3 py-2 rounded-lg border border-white/5 hover:border-white/10 group">
                      <button onClick={() => applyPreset(s)} className="flex-1 text-left">
                        <div className="text-xs font-medium text-text-secondary group-hover:text-text-primary flex items-center gap-1.5">
                          <FolderOpen className="w-3 h-3" /> {s.name}
                        </div>
                        <div className="text-[10px] text-text-muted">
                          {s.modifications.length} modification{s.modifications.length > 1 ? 's' : ''}
                          {s.created_at && ` — ${s.created_at.slice(0, 10)}`}
                        </div>
                      </button>
                      <button
                        onClick={() => deleteScenarioMut.mutate({ runId: model.run_id, name: s.name })}
                        className="p-1 text-text-muted hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {simulateMut.isError && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <p className="text-xs text-red-400">
                  {simulateMut.error instanceof Error ? simulateMut.error.message : 'Simulation failed. Check backend logs.'}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Results */}
      {simResult && (
        <ScenarioResultsPanel result={simResult} modifications={modifications} />
      )}

      {!simResult && !simulateMut.isPending && (
        <div className="flex items-center justify-center py-16 text-text-muted">
          <div className="text-center space-y-2">
            <FlaskConical className="w-10 h-10 mx-auto text-text-muted/20" />
            <p className="text-sm text-text-secondary">No simulation results yet</p>
            <p className="text-xs max-w-sm">
              Configure your stress modifications above and click "Run Simulation" to see how they affect the groundwater level.
            </p>
          </div>
        </div>
      )}

      {simulateMut.isPending && (
        <div className="flex items-center justify-center py-16 text-text-muted">
          <div className="text-center space-y-3">
            <Loader2 className="w-10 h-10 mx-auto text-accent-cyan animate-spin" />
            <p className="text-sm text-text-secondary">Running simulation...</p>
          </div>
        </div>
      )}
    </div>
  )
}
