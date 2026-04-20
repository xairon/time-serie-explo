import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Loader2, Play } from 'lucide-react'
import { usePastasFit, usePastasPreview, usePastasModel } from '@/hooks/usePastas'
import { StationPicker } from '@/components/pastas/StationPicker'
import { PastasConfigForm } from '@/components/pastas/PastasConfigForm'
import { FitResultsPanel } from '@/components/pastas/FitResultsPanel'
import { DataPreviewPanel } from '@/components/pastas/DataPreviewPanel'
import { StationMap } from '@/components/pastas/StationMap'
import { CalValToggle } from '@/components/pastas/CalValToggle'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'
import type { PastasFitResponse } from '@/lib/types'

const DEMO_STATION = '01584X0023/LV3'

export default function FitPage() {
  const [searchParams] = useSearchParams()
  const fitMutation = usePastasFit()

  // Station picker state — pre-fill from URL query param if present
  const [codeBss, setCodeBss] = useState(searchParams.get('station') ?? '')

  // Preview
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

  // Result
  const [fitResult, setFitResult] = useState<PastasFitResponse | null>(null)

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

  // Apply BDLISA preset when preview loads
  const [presetApplied, setPresetApplied] = useState('')
  useEffect(() => {
    if (preview?.preset && preview.code_bss !== presetApplied) {
      const p = preview.preset as Record<string, string>
      if (p.recharge) setRecharge(p.recharge)
      if (p.response) setResponse(p.response)
      if (p.noise) setNoise(p.noise)
      setPresetApplied(preview.code_bss)
    }
  }, [preview, presetApplied])

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
          description="Un modèle Pastas relie le niveau piézométrique aux forçages climatiques (pluie, ETP) par une fonction de transfert. Sélectionnez une station, configurez le modèle, et lancez la calibration."
          steps={[
            'Cherchez et sélectionnez une station piézométrique',
            'Choisissez le modèle de recharge (comment la pluie s\'infiltre) et la fonction de réponse (comment l\'aquifère réagit)',
            'Activez la validation pour réserver une partie des données en test — le modèle s\'entraîne sur le début et on vérifie qu\'il prédit bien la fin',
            'Optionnel : ajoutez des stress supplémentaires (pompage, rivière, etc.) si vous avez les données',
            'Cliquez "Fit Model" — résultats, diagnostics et signatures s\'affichent à droite',
          ]}
          exampleAction={{ label: 'Charger la station exemple (Craie de Champagne)', onClick: loadDemo }}
        />

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-4">Station</h2>
          <StationPicker codeBss={codeBss} onChange={setCodeBss} />
        </div>

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-2">Configuration du modèle</h2>
          {preview?.preset && (preview.preset as Record<string, string>).label && (
            <div className="mb-3 flex items-center gap-2 bg-accent-cyan/5 border border-accent-cyan/20 rounded-lg px-3 py-1.5">
              <span className="text-xs text-accent-cyan font-medium">
                Config recommandée : {(preview.preset as Record<string, string>).label}
              </span>
              <span className="text-[10px] text-text-muted">
                {(preview.preset as Record<string, string>).description}
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
              <p className="text-xs text-text-muted">Ajoute la température ERA5 (°C) comme stress supplémentaire. Peut capturer des effets non-linéaires que l'ETP seule ne modélise pas.</p>
            </div>
          </label>
        </div>

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <label className="block text-sm font-medium text-text-secondary mb-1">
            Nom du run (optionnel)
          </label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g. station_01_linear"
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
              {fitResult ? 'Re-calibrer avec cette config' : 'Lancer la calibration'}
            </>
          )}
        </button>

        {fitMutation.isError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            <p className="text-xs text-red-400">
              {fitMutation.error instanceof Error
                ? fitMutation.error.message
                : 'Fit failed. Check backend logs.'}
            </p>
          </div>
        )}
      </div>

      {/* Right column — preview + results */}
      <div className="flex-1 min-w-0 space-y-4">
        {previewLoading && (
          <div className="flex items-center justify-center h-24 text-text-muted text-sm gap-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading preview…
          </div>
        )}

        {preview && !previewLoading && (
          <>
            <StationMap
              lat={typeof preview.metadata.latitude === 'number' ? preview.metadata.latitude : null}
              lon={typeof preview.metadata.longitude === 'number' ? preview.metadata.longitude : null}
              label={preview.code_bss}
            />
            <DataPreviewPanel
              preview={preview}
              onRangeChange={(t0, t1) => {
                setTmin(t0)
                setTmax(t1)
              }}
            />
          </>
        )}

        {!preview && !previewLoading && !codeBss && (
          <div className="flex items-center justify-center h-full text-text-muted text-sm">
            <div className="text-center space-y-2">
              <p className="text-text-secondary">No station selected</p>
              <p>Select a station to preview its data.</p>
            </div>
          </div>
        )}

        {fitResult && (
          <>
            {preview && <div className="border-t border-white/5" />}
            <FitResultsPanel result={fitResult} />
          </>
        )}
      </div>
    </div>
  )
}
