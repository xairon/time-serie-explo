import { useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import { usePastasStationInfo, usePastasPreview, usePastasDiagnose } from '@/hooks/usePastas'
import { usePastasMode } from './PastasLayout'
import { StationPicker } from '@/components/pastas/StationPicker'
import { StationDetailPanel } from '@/components/pastas/StationDetailPanel'
import { PreFitDiagnosticPanel } from '@/components/pastas/PreFitDiagnosticPanel'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'
import type { DiagnosticRecommendation } from '@/lib/types'

const DEMO_STATION = '01584X0023/LV3'

export default function StationStep() {
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()
  const { mode } = usePastasMode()

  const codeBss = searchParams.get('station') ?? ''

  function setCodeBss(value: string) {
    const next = new URLSearchParams(searchParams)
    if (value) {
      next.set('station', value)
    } else {
      next.delete('station')
    }
    setSearchParams(next, { replace: true })
  }

  const { data: stationInfo, isLoading: stationInfoLoading } = usePastasStationInfo(codeBss || null)
  const { data: preview, isLoading: previewLoading } = usePastasPreview(codeBss || null)
  const { data: diagnosis, isLoading: diagnoseLoading } = usePastasDiagnose(codeBss || null)

  // In guided mode, auto-apply recommendations
  useEffect(() => {
    if (mode === 'guided' && diagnosis?.recommendations?.length) {
      // Recommendations are displayed as auto-applied in guided mode
      // The actual params will be forwarded to the calibrate step
    }
  }, [mode, diagnosis])

  function handleApplyRecommendation(rec: DiagnosticRecommendation) {
    // In expert mode, recommendations need explicit application
    // For now, we log the action — the calibrate step reads diagnosis directly
    console.log('Apply recommendation:', rec)
  }

  function handleNext() {
    if (!codeBss) return
    const params = new URLSearchParams({ station: codeBss })
    // Forward recommended tmin/tmax from diagnosis if available
    if (diagnosis?.recommended_tmin) params.set('tmin', diagnosis.recommended_tmin)
    if (diagnosis?.recommended_tmax) params.set('tmax', diagnosis.recommended_tmax)
    navigate(`/pastas/calibrate?${params.toString()}`)
  }

  const loadDemo = () => {
    setCodeBss(DEMO_STATION)
  }

  return (
    <div className="p-6 flex gap-6 min-h-full">
      {/* Left column — station picker + onboarding */}
      <div className="w-80 shrink-0 space-y-4">
        <OnboardingBanner
          id="station-step"
          title="Select a station"
          description="Choose a piezometric station to analyze. The diagnostics panel will check data quality and suggest optimal calibration settings."
          steps={[
            'Search by BSS code, commune, or department',
            'Review station metadata and time series preview',
            'Check pre-fit diagnostics for data quality issues',
            'Click "Next: Calibrate" to proceed with model fitting',
          ]}
          exampleAction={{ label: 'Load example station (Champagne chalk)', onClick: loadDemo }}
        />

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-4">Station</h2>
          <StationPicker codeBss={codeBss} onChange={setCodeBss} />
        </div>

        {/* Next button */}
        <button
          onClick={handleNext}
          disabled={!codeBss}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Next: Calibrate
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Right column — station detail + diagnostics */}
      <div className="flex-1 min-w-0 space-y-4">
        {codeBss && (
          <div className="flex items-center gap-2 text-xs text-text-muted mb-2">
            <a href={`/station/piezo/${encodeURIComponent(codeBss)}`} className="text-accent-cyan hover:underline">
              &larr; View station details
            </a>
          </div>
        )}

        {/* Station detail panel */}
        {codeBss && (stationInfo || stationInfoLoading) && (
          <StationDetailPanel
            stationInfo={stationInfo}
            stationInfoLoading={stationInfoLoading}
            preview={preview}
            previewLoading={previewLoading}
          />
        )}

        {/* Pre-fit diagnostics */}
        {codeBss && (diagnosis || diagnoseLoading) && (
          <div className="bg-bg-card border border-white/5 rounded-xl p-4">
            <PreFitDiagnosticPanel
              diagnosis={diagnosis}
              isLoading={diagnoseLoading}
              onApplyRecommendation={handleApplyRecommendation}
              mode={mode}
            />
          </div>
        )}

        {/* Guided mode summary */}
        {mode === 'guided' && diagnosis && !diagnoseLoading && (
          <div className="bg-accent-cyan/5 border border-accent-cyan/20 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-accent-cyan mb-2">Summary</h3>
            <div className="space-y-1">
              {diagnosis.recommended_tmin && diagnosis.recommended_tmax && (
                <p className="text-xs text-text-secondary">
                  Recommended period: {diagnosis.recommended_tmin} to {diagnosis.recommended_tmax}
                </p>
              )}
              {diagnosis.recommendations.length > 0 ? (
                <p className="text-xs text-text-secondary">
                  {diagnosis.recommendations.length} recommendation(s) will be auto-applied during calibration.
                </p>
              ) : (
                <p className="text-xs text-text-secondary">
                  Data quality looks good. Ready to calibrate.
                </p>
              )}
            </div>
          </div>
        )}

        {!codeBss && (
          <div className="flex items-center justify-center h-full text-text-muted text-sm">
            <div className="text-center space-y-2">
              <p className="text-text-secondary">No station selected</p>
              <p>Search and select a piezometric station to preview its data and diagnostics.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
