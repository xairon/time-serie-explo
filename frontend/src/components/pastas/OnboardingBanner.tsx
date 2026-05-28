import { useState, useEffect } from 'react'
import { X, Lightbulb, Rocket } from 'lucide-react'
import { useTranslation } from 'react-i18next'

interface Props {
  id: string
  title: string
  description: string
  steps?: string[]
  exampleAction?: { label: string; onClick: () => void }
}

export function OnboardingBanner({ id, title, description, steps, exampleAction }: Props) {
  const { t } = useTranslation()
  const storageKey = `pastas_onboarding_${id}`
  const [dismissed, setDismissed] = useState(true)

  useEffect(() => {
    setDismissed(localStorage.getItem(storageKey) === 'dismissed')
  }, [storageKey])

  const dismiss = () => {
    localStorage.setItem(storageKey, 'dismissed')
    setDismissed(true)
  }

  const show = () => {
    localStorage.removeItem(storageKey)
    setDismissed(false)
  }

  if (dismissed) {
    return (
      <button
        onClick={show}
        className="flex items-center gap-1.5 text-xs text-text-muted hover:text-accent-cyan transition-colors mb-3"
      >
        <Lightbulb className="w-3.5 h-3.5" />
        {t('pastas.ui.showGuide')}
      </button>
    )
  }

  return (
    <div className="bg-accent-cyan/5 border border-accent-cyan/20 rounded-xl p-4 mb-4 relative">
      <button
        onClick={dismiss}
        className="absolute top-3 right-3 p-1 text-text-muted hover:text-text-primary transition-colors"
      >
        <X className="w-4 h-4" />
      </button>

      <div className="flex items-start gap-3">
        <div className="shrink-0 w-8 h-8 rounded-lg bg-accent-cyan/10 flex items-center justify-center mt-0.5">
          <Lightbulb className="w-4 h-4 text-accent-cyan" />
        </div>

        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-text-primary">{title}</h3>
          <p className="text-xs text-text-secondary mt-1 leading-relaxed">{description}</p>

          {steps && steps.length > 0 && (
            <ol className="mt-2 space-y-1">
              {steps.map((step, i) => (
                <li key={i} className="text-xs text-text-secondary flex items-start gap-2">
                  <span className="shrink-0 w-4 h-4 rounded-full bg-accent-cyan/10 text-accent-cyan flex items-center justify-center text-[10px] font-semibold mt-0.5">
                    {i + 1}
                  </span>
                  {step}
                </li>
              ))}
            </ol>
          )}

          {exampleAction && (
            <button
              onClick={() => { exampleAction.onClick(); dismiss() }}
              className="mt-3 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent-cyan/20 text-accent-cyan text-xs font-medium hover:bg-accent-cyan/30 transition-colors border border-accent-cyan/30"
            >
              <Rocket className="w-3.5 h-3.5" />
              {exampleAction.label}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
