import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Loader2, Play, Info, Save, Trash2, FolderOpen } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { usePastasModels, usePastasSimulate, usePastasModel } from '@/hooks/usePastas'
import { useScenarioPresets } from '@/hooks/useScenarioPresets'
import { useSavedScenarios, useSaveScenario, useDeleteScenario } from '@/hooks/useSavedScenarios'
import { ScenarioComposer } from '@/components/pastas/ScenarioComposer'
import { ScenarioResultsPanel } from '@/components/pastas/ScenarioResultsPanel'
import type { PastasScenarioResponse, AquiferFamily } from '@/lib/types'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'
import type { ModificationData } from '@/components/pastas/ModificationCard'
import { api } from '@/lib/api'

function modificationToPayload(mod: ModificationData): Record<string, unknown> {
  if (mod.type === 'pumping_upload') {
    return { type: mod.type, csv_rows: mod.rows, distance_m: mod.distance_m, rfunc: mod.rfunc }
  }
  return mod as unknown as Record<string, unknown>
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

export default function ScenariosPage() {
  const { t } = useTranslation()
  const [searchParams] = useSearchParams()
  const { data: models = [] } = usePastasModels()
  const simulateMutation = usePastasSimulate()

  const [runId, setRunId] = useState(searchParams.get('model') ?? '')
  const [tmin, setTmin] = useState('')
  const [tmax, setTmax] = useState('')
  const [modifications, setModifications] = useState<ModificationData[]>([])
  const [simResult, setSimResult] = useState<PastasScenarioResponse | null>(null)
  const [saveDialogOpen, setSaveDialogOpen] = useState(false)
  const [saveName, setSaveName] = useState('')
  const [aquiferFamily, setAquiferFamily] = useState<AquiferFamily>('sedimentary')

  const { data: selectedModel } = usePastasModel(runId || null)
  const selected = models.find(m => m.run_id === runId)

  // Auto-detect aquifer family from station preview
  const codeBss = selected?.code_bss
  const { data: stationPreview } = useQuery({
    queryKey: ['pastas', 'preview', codeBss],
    queryFn: () => api.pastas.preview(codeBss!),
    enabled: !!codeBss,
    staleTime: 30 * 60 * 1000,
  })

  useEffect(() => {
    if (stationPreview?.metadata) {
      setAquiferFamily(detectAquiferFamily(stationPreview.metadata))
    }
  }, [stationPreview])

  // Load presets referential
  const { data: presetsData } = useScenarioPresets({
    aquifer_family: aquiferFamily,
    tmin: tmin || undefined,
    tmax: tmax || undefined,
  })

  // Saved scenarios
  const { data: savedScenarios = [] } = useSavedScenarios(runId || null)
  const saveScenarioMutation = useSaveScenario()
  const deleteScenarioMutation = useDeleteScenario()

  useEffect(() => {
    if (selectedModel && !tmin && !tmax) {
      const obs = selectedModel.observed
      if (obs?.index?.length > 0) {
        setTmin(obs.index[0].slice(0, 10))
        setTmax(obs.index[obs.index.length - 1].slice(0, 10))
      }
    }
  }, [selectedModel])

  const canSimulate = !!runId && !!tmin && !!tmax

  // Pumping profiles for the detected aquifer family (all usages)
  const pumpingProfiles = useMemo(() => {
    if (!presetsData) return null
    const result: Record<string, typeof presetsData.pumping_profiles.aep.alluvial> = {}
    for (const usage of ['aep', 'irrigation', 'industrial'] as const) {
      const p = presetsData.pumping_profiles[usage]?.[aquiferFamily]
      if (p) result[usage] = p
    }
    return Object.keys(result).length > 0 ? result : null
  }, [presetsData, aquiferFamily])

  async function handleSimulate() {
    if (!canSimulate) return
    try {
      const result = await simulateMutation.mutateAsync({
        run_id: runId,
        tmin,
        tmax,
        modifications: modifications.map(modificationToPayload),
      })
      setSimResult(result)
    } catch { /* Error handled by mutation state */ }
  }

  function applyPreset(preset: { modifications: Record<string, unknown>[] }) {
    const mods = preset.modifications.map(m => {
      const mod = { ...m } as Record<string, unknown>
      if ('start' in mod && !mod.start && tmin) mod.start = tmin
      if ('end' in mod && !mod.end && tmax) mod.end = tmax
      return mod as unknown as ModificationData
    })
    setModifications(mods)
  }

  async function handleSave() {
    if (!runId || !saveName.trim()) return
    await saveScenarioMutation.mutateAsync({
      runId,
      body: {
        name: saveName.trim(),
        modifications: modifications.map(modificationToPayload),
        tmin: tmin || undefined,
        tmax: tmax || undefined,
      },
    })
    setSaveDialogOpen(false)
    setSaveName('')
  }

  const inputClass =
    'w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50'

  return (
    <div className="p-6 flex gap-6 min-h-full">
      {/* Left column */}
      <div className="w-96 shrink-0 space-y-4">
        <OnboardingBanner
          id="scenarios"
          title={t('pastas.scenariosPage.title')}
          description={t('pastas.scenariosPage.description')}
          steps={[
            t('pastas.scenariosPage.steps1'),
            t('pastas.scenariosPage.steps2'),
            t('pastas.scenariosPage.steps3'),
          ]}
        />

        {/* Model picker */}
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-3">{t('pastas.scenariosPage.calibratedModel')}</h2>
          <select
            value={runId}
            onChange={(e) => { setRunId(e.target.value); setSimResult(null) }}
            className={inputClass}
          >
            <option value="">{t('pastas.scenariosPage.pickModel')}</option>
            {models.map((m) => (
              <option key={m.run_id} value={m.run_id}>
                {m.name || m.run_id.slice(0, 8)} — {m.code_bss}
              </option>
            ))}
          </select>

          {selected && (
            <div className="mt-3 space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-accent-cyan">{selected.code_bss}</span>
                {selected.nse != null && (
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${
                    selected.nse > 0.7 ? 'border-green-500/30 text-green-400 bg-green-500/10' :
                    selected.nse > 0.4 ? 'border-accent-cyan/30 text-accent-cyan bg-accent-cyan/10' :
                    'border-red-500/30 text-red-400 bg-red-500/10'
                  }`}>
                    NSE {selected.nse.toFixed(2)}
                  </span>
                )}
                {presetsData?.aquifer_families[aquiferFamily] && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full border border-white/10 text-text-muted">
                    {presetsData.aquifer_families[aquiferFamily]}
                  </span>
                )}
              </div>
              <div className="flex flex-wrap gap-1">
                {[selected.recharge_type, selected.response_type].map(t => (
                  <span key={t} className="text-[10px] px-1.5 py-0.5 rounded border border-white/10 text-text-muted">{t}</span>
                ))}
                {selected.noise_type !== 'unknown' && selected.noise_type !== 'none' && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded border border-white/10 text-text-muted">{selected.noise_type}</span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Simulation window */}
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-3">{t('pastas.scenarios.simulationWindow')}</h2>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-text-muted mb-1">{t('pastas.scenariosPage.start')}</label>
              <input type="date" value={tmin} onChange={(e) => setTmin(e.target.value)} className={inputClass} />
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">{t('pastas.scenariosPage.end')}</label>
              <input type="date" value={tmax} onChange={(e) => setTmax(e.target.value)} className={inputClass} />
            </div>
          </div>
        </div>

        {/* Contextual presets */}
        {runId && presetsData && (
          <div className="bg-bg-card border border-white/5 rounded-xl p-4">
            <h2 className="text-sm font-semibold text-text-primary mb-3">{t('pastas.scenariosPage.presetScenarios')}</h2>
            <div className="grid grid-cols-2 gap-1.5">
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
        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-3">{t('pastas.scenariosPage.modifications')}</h2>
          <ScenarioComposer
            modifications={modifications}
            onChange={setModifications}
            tmin={tmin}
            tmax={tmax}
            pumpingProfiles={pumpingProfiles}
            scaleStressLimits={presetsData?.non_pumping_limits?.scale_stress ?? null}
            linearTrendLimits={presetsData?.non_pumping_limits?.linear_trend ?? null}
          />
        </div>

        {/* Simulate + Save buttons */}
        <div className="flex gap-2">
          <button
            onClick={handleSimulate}
            disabled={!canSimulate || simulateMutation.isPending}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {simulateMutation.isPending ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> {t('pastas.scenariosPage.simulating')}</>
            ) : (
              <><Play className="w-4 h-4" /> {t('pastas.scenariosPage.simulate')}</>
            )}
          </button>
          {modifications.length > 0 && runId && (
            <button
              onClick={() => setSaveDialogOpen(true)}
              className="flex items-center gap-1.5 px-3 py-2.5 rounded-lg bg-white/5 text-text-secondary text-sm border border-white/10 hover:border-white/20 transition-colors"
              title={t('pastas.ui.saveScenarioTitle')}
            >
              <Save className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Save dialog */}
        {saveDialogOpen && (
          <div className="bg-bg-card border border-white/10 rounded-xl p-4 space-y-3">
            <h3 className="text-xs font-semibold text-text-primary">{t('pastas.ui.saveScenarioTitle')}</h3>
            <input
              type="text"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder={t('pastas.ui.scenarioNamePlaceholder')}
              className={inputClass}
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && handleSave()}
            />
            <div className="flex gap-2">
              <button
                onClick={handleSave}
                disabled={!saveName.trim() || saveScenarioMutation.isPending}
                className="flex-1 px-3 py-1.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-xs font-medium border border-accent-cyan/30 disabled:opacity-40"
              >
                {saveScenarioMutation.isPending ? t('pastas.ui.saving') : t('pastas.scenarios.saveScenario')}
              </button>
              <button
                onClick={() => setSaveDialogOpen(false)}
                className="px-3 py-1.5 rounded-lg text-text-muted text-xs border border-white/10"
              >
                {t('common.cancel')}
              </button>
            </div>
          </div>
        )}

        {simulateMutation.isError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            <p className="text-xs text-red-400">
              {simulateMutation.error instanceof Error ? simulateMutation.error.message : t('pastas.scenariosPage.simulationFailed')}
            </p>
          </div>
        )}

        {/* Saved scenarios */}
        {runId && savedScenarios.length > 0 && (
          <div className="bg-bg-card border border-white/5 rounded-xl p-4">
            <h2 className="text-sm font-semibold text-text-primary mb-3">{t('pastas.scenariosPage.savedScenarios')}</h2>
            <div className="space-y-1.5">
              {savedScenarios.map((s) => (
                <div
                  key={s.name}
                  className="flex items-center justify-between px-3 py-2 rounded-lg border border-white/5 hover:border-white/10 group"
                >
                  <button
                    onClick={() => applyPreset(s)}
                    className="flex-1 text-left"
                  >
                    <div className="text-xs font-medium text-text-secondary group-hover:text-text-primary flex items-center gap-1.5">
                      <FolderOpen className="w-3 h-3" />
                      {s.name}
                    </div>
                    <div className="text-[10px] text-text-muted">
                      {s.modifications.length} {s.modifications.length > 1 ? t('pastas.ui.modificationsPlural') : t('pastas.ui.modificationSingular')}
                      {s.created_at && ` — ${s.created_at.slice(0, 10)}`}
                    </div>
                  </button>
                  <button
                    onClick={() => deleteScenarioMutation.mutate({ runId, name: s.name })}
                    className="p-1 text-text-muted hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right column */}
      <div className="flex-1 min-w-0">
        {simResult ? (
          <ScenarioResultsPanel result={simResult} modifications={modifications} codeBss={codeBss} />
        ) : (
          <div className="flex items-center justify-center h-full text-text-muted text-sm">
            <div className="text-center space-y-3">
              <Info className="w-8 h-8 mx-auto text-text-muted/50" />
              <p className="text-text-secondary">{t('pastas.scenariosPage.noSimulation')}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
