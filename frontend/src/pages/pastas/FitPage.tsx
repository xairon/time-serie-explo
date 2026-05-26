import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Loader2, Play, BarChart3, FlaskConical } from 'lucide-react'
import { usePastasFit, usePastasPreview, usePastasModel, usePastasStationInfo } from '@/hooks/usePastas'
import { StationPicker } from '@/components/pastas/StationPicker'
import { PastasConfigForm } from '@/components/pastas/PastasConfigForm'
import { FitResultsPanel } from '@/components/pastas/FitResultsPanel'
import { StationDetailPanel } from '@/components/pastas/StationDetailPanel'
import { CalValToggle } from '@/components/pastas/CalValToggle'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'
import { ScenarioWorkflow } from '@/components/pastas/ScenarioWorkflow'
import type { PastasFitResponse } from '@/lib/types'

const DEMO_STATION = '01584X0023/LV3'

export default function FitPage() {
  const [searchParams] = useSearchParams()
  const fitMutation = usePastasFit()

  // Station picker state — pre-fill from URL query param if present
  const [codeBss, setCodeBss] = useState(searchParams.get('station') ?? '')

  // Station info (instant) + preview (heavy)
  const { data: stationInfo, isLoading: stationInfoLoading } = usePastasStationInfo(codeBss || null)
  const { data: preview, isLoading: previewLoading } = usePastasPreview(codeBss || null)

  // Config form state
  const [recharge, setRecharge] = useState('Linear')
  const [response, setResponse] = useState('Gamma')
  const [noise, setNoise] = useState('ArNoiseModel')
  const [solver, setSolver] = useState('LeastSquares')
  const [tmin, setTmin] = useState('')
  const [tmax, setTmax] = useState('')
  const [modelName, setModelName] = useState('')

  // Cal/val split
  const [valSplit, setValSplit] = useState<number | null>(null)

  // Temperature stress
  const [includeTemp, setIncludeTemp] = useState(false)

  // Result + right panel mode
  const [fitResult, setFitResult] = useState<PastasFitResponse | null>(null)
  const [rightTab, setRightTab] = useState<'results' | 'scenarios'>('results')

  // Load existing model from ?model= query param
  const modelId = searchParams.get('model')
  const { data: loadedModel } = usePastasModel(modelId)

  useEffect(() => {
    if (loadedModel) {
      setFitResult(loadedModel)
      if (loadedModel.code_bss) {
        setCodeBss(loadedModel.code_bss)
      }
    }
  }, [loadedModel])

  // Apply BDLISA preset when station info loads (instant)
  const [presetApplied, setPresetApplied] = useState('')
  useEffect(() => {
    if (stationInfo?.preset && codeBss !== presetApplied) {
      const p = stationInfo.preset as Record<string, string>
      if (p.recharge) setRecharge(p.recharge)
      if (p.response) setResponse(p.response)
      if (p.noise) setNoise(p.noise)
      setPresetApplied(codeBss)
    }
  }, [stationInfo, codeBss, presetApplied])

  const canFit = !!codeBss

  async function handleFit() {
    if (!canFit) return
    try {
      const result = await fitMutation.mutateAsync({
        code_bss: codeBss,
        tmin: tmin || undefined,
        tmax: tmax || undefined,
        recharge: { type: recharge },
        response: { type: response },
        noise: { type: noise },
        solver: { type: solver },
        name: modelName || undefined,
        val_split: valSplit ?? undefined,
        include_temp: includeTemp || undefined,
      })
      setFitResult(result)
    } catch {
      // Error handled by mutation state
    }
  }

  const loadDemo = () => {
    setCodeBss(DEMO_STATION)
    setRecharge('Linear')
    setResponse('Gamma')
    setNoise('ArNoiseModel')
    setSolver('LeastSquares')
    setValSplit(0.3)
    setModelName('demo_craie_champagne')
  }

  return (
    <div className="p-6 flex gap-6 min-h-full">
      {/* Left column — configuration */}
      <div className="w-80 shrink-0 space-y-4">
        <OnboardingBanner
          id="fit"
          title="Calibrer un modèle Pastas"
          description="Un modèle Pastas relie les niveaux piézométriques aux forçages climatiques (précipitations, ETP) via une fonction de transfert. Sélectionnez une station, configurez le modèle et lancez la calibration."
          steps={[
            'Rechercher et sélectionner une station piézométrique',
            "Choisir le modèle de recharge (mode d'infiltration des précipitations) et la fonction de réponse (mode de réponse de l'aquifère)",
            "Activer la validation pour mettre de côté une portion des données pour le test — le modèle est calibré sur les premières années et vérifié sur le reste",
            "Optionnel : ajouter des stress supplémentaires (pompage, rivière, etc.) si vous disposez des données",
            "Cliquer sur « Calibrer le modèle » — les résultats, diagnostics et signatures apparaissent à droite",
          ]}
          exampleAction={{ label: 'Charger une station exemple (craie de Champagne)', onClick: loadDemo }}
        />

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-4">Station</h2>
          <StationPicker codeBss={codeBss} onChange={setCodeBss} />
        </div>

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-2">Configuration du modèle</h2>
          {stationInfo?.preset != null && (stationInfo.preset as Record<string, string>).label && (
            <div className="mb-3 flex items-center gap-2 bg-accent-cyan/5 border border-accent-cyan/20 rounded-lg px-3 py-1.5">
              <span className="text-xs text-accent-cyan font-medium">
                Config recommandée : {(stationInfo.preset as Record<string, string>).label}
              </span>
              <span className="text-[10px] text-text-muted">
                {(stationInfo.preset as Record<string, string>).description}
              </span>
            </div>
          )}
          <PastasConfigForm
            recharge={recharge}
            onRechargeChange={setRecharge}
            response={response}
            onResponseChange={setResponse}
            noise={noise}
            onNoiseChange={setNoise}
            solver={solver}
            onSolverChange={setSolver}
            tmin={tmin}
            onTminChange={setTmin}
            tmax={tmax}
            onTmaxChange={setTmax}
          />
        </div>

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <CalValToggle valSplit={valSplit} onChange={setValSplit} />
        </div>

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={includeTemp}
              onChange={e => setIncludeTemp(e.target.checked)}
              className="accent-accent-cyan w-4 h-4"
            />
            <div>
              <span className="text-sm font-medium text-text-secondary">Inclure la température</span>
              <p className="text-xs text-text-muted">Ajoute la température ERA5 (°C) comme stress supplémentaire. Peut capturer des effets non linéaires que l'ETP seule ne modélise pas.</p>
            </div>
          </label>
        </div>

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Nom de l'exécution (optionnel)
          </label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="ex. station_01_linear"
            className="w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm placeholder:text-text-muted focus:outline-none focus:border-accent-cyan/50"
          />
        </div>

        <button
          onClick={handleFit}
          disabled={!canFit || fitMutation.isPending}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {fitMutation.isPending ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Calibration en cours…
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              {fitResult ? 'Recalibrer avec cette config' : 'Lancer la calibration'}
            </>
          )}
        </button>

        {fitMutation.isError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            <p className="text-xs text-red-400">
              {fitMutation.error instanceof Error
                ? fitMutation.error.message
                : 'Échec de la calibration. Consultez les logs du backend.'}
            </p>
          </div>
        )}
      </div>

      {/* Right column — preview + results/scenarios */}
      <div className="flex-1 min-w-0 space-y-4">
        {codeBss && (
          <div className="flex items-center gap-2 text-xs text-text-muted mb-2">
            <a href={`/station/piezo/${encodeURIComponent(codeBss)}`} className="text-accent-cyan hover:underline">
              &larr; Voir le détail de la station
            </a>
          </div>
        )}
        {/* Rich station detail + lazy time series */}
        {codeBss && (stationInfo || stationInfoLoading) && (
          <StationDetailPanel
            stationInfo={stationInfo}
            stationInfoLoading={stationInfoLoading}
            preview={preview}
            previewLoading={previewLoading}
            onRangeChange={(t0, t1) => { setTmin(t0); setTmax(t1) }}
          />
        )}

        {!codeBss && (
          <div className="flex items-center justify-center h-full text-text-muted text-sm">
            <div className="text-center space-y-2">
              <p className="text-text-secondary">Aucune station sélectionnée</p>
              <p>Sélectionnez une station pour prévisualiser ses données.</p>
            </div>
          </div>
        )}

        {fitResult && (
          <>
            {preview && <div className="border-t border-white/5" />}

            {/* Pipeline tabs */}
            <div className="flex items-center gap-1 bg-bg-card border border-white/5 rounded-xl p-1">
              <button
                onClick={() => setRightTab('results')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors flex-1 justify-center ${
                  rightTab === 'results'
                    ? 'bg-accent-cyan/10 text-accent-cyan'
                    : 'text-text-muted hover:text-text-secondary hover:bg-bg-hover'
                }`}
              >
                <BarChart3 className="w-4 h-4" />
                Résultats et diagnostics
              </button>
              <button
                onClick={() => setRightTab('scenarios')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors flex-1 justify-center ${
                  rightTab === 'scenarios'
                    ? 'bg-purple-500/10 text-purple-400'
                    : 'text-text-muted hover:text-text-secondary hover:bg-bg-hover'
                }`}
              >
                <FlaskConical className="w-4 h-4" />
                Scénarios prospectifs
              </button>
            </div>

            {/* Tab content */}
            {rightTab === 'results' ? (
              <FitResultsPanel result={fitResult} codeBss={codeBss} />
            ) : (
              <ScenarioWorkflow
                model={fitResult}
                codeBss={codeBss}
                onRefit={(newResult) => { setFitResult(newResult); setRightTab('results') }}
              />
            )}
          </>
        )}
      </div>
    </div>
  )
}
