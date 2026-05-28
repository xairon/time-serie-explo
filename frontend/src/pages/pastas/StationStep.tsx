import { useSearchParams, useNavigate } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { usePastasStationInfo, usePastasPreview } from '@/hooks/usePastas'
import { StationPicker } from '@/components/pastas/StationPicker'
import { StationDetailPanel } from '@/components/pastas/StationDetailPanel'
import { OnboardingBanner } from '@/components/pastas/OnboardingBanner'

const DEMO_STATION = '01584X0023/LV3'

export default function StationStep() {
  const { t } = useTranslation()
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()

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

  function handleNext() {
    if (!codeBss) return
    navigate(`/pastas/calibrate?${new URLSearchParams({ station: codeBss }).toString()}`)
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
          title={t('pastas.station.title')}
          description={t('pastas.station.description')}
          steps={[
            t('pastas.stationStep.steps1'),
            t('pastas.stationStep.steps2'),
            t('pastas.stationStep.steps3'),
          ]}
          exampleAction={{ label: t('pastas.station.loadExample'), onClick: loadDemo }}
        />

        <div className="bg-bg-card border border-white/5 rounded-xl p-4">
          <h2 className="text-sm font-semibold text-text-primary mb-4">{t('pastas.ui.stationSection')}</h2>
          <StationPicker codeBss={codeBss} onChange={setCodeBss} />
        </div>

        {/* Next button */}
        <button
          onClick={handleNext}
          disabled={!codeBss}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-sm font-medium border border-accent-cyan/30 hover:bg-accent-cyan/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {t('pastas.station.nextCalibrate')}
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Right column — station detail */}
      <div className="flex-1 min-w-0 space-y-4">
        {codeBss && (
          <div className="flex items-center gap-2 text-xs text-text-muted mb-2">
            <a href={`/station/piezo/${encodeURIComponent(codeBss)}`} className="text-accent-cyan hover:underline">
              &larr; {t('pastas.station.viewDetail')}
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

        {!codeBss && (
          <div className="flex items-center justify-center h-full text-text-muted text-sm">
            <div className="text-center space-y-2">
              <p className="text-text-secondary">{t('pastas.station.noSelection')}</p>
              <p>{t('pastas.station.selectPrompt')}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
