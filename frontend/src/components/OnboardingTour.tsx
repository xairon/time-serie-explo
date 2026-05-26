import { useState, useEffect, useLayoutEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { X, ChevronLeft, ChevronRight } from 'lucide-react'

const STORAGE_KEY = 'junon.tour.completed.v1'

/**
 * Self-contained first-visit guided tour. A small custom implementation rather
 * than a library so we control the styling end-to-end and don't drag in a
 * peer-dep tree (react-joyride v2 doesn't support React 19, v3 has a very
 * different API).
 *
 * Behaviour: highlights a UI element by drawing a cutout in a dimmed overlay
 * and positions a tooltip card next to it. Each step targets an element via a
 * `data-tour="…"` attribute. Steps with no target render centered.
 *
 * Call window.junonStartTour() to replay the tour at any time.
 */
interface Step {
  target?: string  // data-tour="…" value; omitted = centered welcome
  title: string
  content: React.ReactNode
}

const STEPS: Step[] = [
  {
    title: 'Bienvenue sur Junon',
    content: (
      <>
        <p className="mb-2">Plateforme d'analyse piézométrique et hydrométrique pour les hydrogéologues BRGM.</p>
        <p className="text-text-secondary text-xs">Tour rapide en ~30 secondes pour découvrir les 4 modules.</p>
      </>
    ),
  },
  {
    target: 'nav-observatory',
    title: 'Observatoire',
    content: 'Carte interactive des stations piézométriques et hydrométriques. Filtres, classifications, alertes.',
  },
  {
    target: 'nav-compare',
    title: 'Comparer',
    content: '2 à 5 stations côte à côte : chroniques alignées, indices de sécheresse (SPLI/SSFI), statistiques comparatives.',
  },
  {
    target: 'nav-pastas',
    title: 'Pastas Lab',
    content: 'Calibration de modèles transfer function noise (TFN) sur des piézomètres. Diagnostics, signatures hydrologiques, scénarios prospectifs (pompage, climat).',
  },
  {
    target: 'nav-ai',
    title: 'Lab IA',
    content: 'Prévisions par deep learning (Darts / PyTorch Lightning). Préréglages métier en 1 clic — court terme, sécheresse, débit fluvial.',
  },
  {
    title: 'C\'est parti',
    content: (
      <>
        <p className="mb-2">Cliquez sur une station de la carte pour explorer ses données, ou utilisez le menu du haut pour accéder aux autres modules.</p>
        <p className="text-text-secondary text-xs">Pour relancer ce tour, tapez <code className="bg-bg-input px-1 rounded">junonStartTour()</code> dans la console.</p>
      </>
    ),
  },
]

declare global {
  interface Window { junonStartTour?: () => void }
}

interface Rect { top: number; left: number; width: number; height: number }

function getRect(selector: string | undefined): Rect | null {
  if (!selector) return null
  const el = document.querySelector(`[data-tour="${selector}"]`) as HTMLElement | null
  if (!el) return null
  const r = el.getBoundingClientRect()
  return { top: r.top, left: r.left, width: r.width, height: r.height }
}

export function OnboardingTour() {
  const [open, setOpen] = useState(false)
  const [stepIdx, setStepIdx] = useState(0)
  const [targetRect, setTargetRect] = useState<Rect | null>(null)

  const step = STEPS[stepIdx]

  // Trigger on first visit.
  useEffect(() => {
    if (window.localStorage.getItem(STORAGE_KEY)) return
    const t = setTimeout(() => setOpen(true), 600)  // let nav render
    return () => clearTimeout(t)
  }, [])

  // Expose programmatic replay.
  useEffect(() => {
    window.junonStartTour = () => { setStepIdx(0); setOpen(true) }
    return () => { delete window.junonStartTour }
  }, [])

  // Recompute target position when step changes or window resizes.
  useLayoutEffect(() => {
    if (!open) return
    const update = () => setTargetRect(getRect(step?.target))
    update()
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)
    return () => {
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
    }
  }, [open, step])

  const finish = useCallback(() => {
    window.localStorage.setItem(STORAGE_KEY, '1')
    setOpen(false)
    setStepIdx(0)
  }, [])

  const next = () => {
    if (stepIdx + 1 >= STEPS.length) finish()
    else setStepIdx((i) => i + 1)
  }
  const prev = () => setStepIdx((i) => Math.max(0, i - 1))

  if (!open) return null

  // Position the tooltip card: below the target if there's room, else above,
  // else centered. Card width 320px.
  const CARD_WIDTH = 340
  const CARD_HEIGHT_EST = 200
  const PADDING = 16
  let cardStyle: React.CSSProperties
  if (targetRect) {
    const spaceBelow = window.innerHeight - (targetRect.top + targetRect.height)
    const placeBelow = spaceBelow >= CARD_HEIGHT_EST + PADDING
    const top = placeBelow
      ? targetRect.top + targetRect.height + PADDING
      : Math.max(PADDING, targetRect.top - CARD_HEIGHT_EST - PADDING)
    const targetCenter = targetRect.left + targetRect.width / 2
    const left = Math.max(PADDING, Math.min(window.innerWidth - CARD_WIDTH - PADDING, targetCenter - CARD_WIDTH / 2))
    cardStyle = { top, left, width: CARD_WIDTH, position: 'fixed' }
  } else {
    cardStyle = {
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      width: CARD_WIDTH,
      position: 'fixed',
    }
  }

  return createPortal(
    <>
      {/* Dim overlay; cuts a transparent hole over the target via mask. */}
      <div
        className="fixed inset-0 z-[9998]"
        style={{
          background: 'rgba(0,0,0,0.55)',
          ...(targetRect ? {
            // SVG mask: full black except the target rect (transparent) so the
            // element underneath shines through the dim layer.
            WebkitMask: `radial-gradient(circle at center, transparent 0, transparent 0), linear-gradient(#000, #000)`,
            mask: `linear-gradient(#000, #000)`,
            maskImage: 'none',
          } : {}),
        }}
        onClick={finish}
      />

      {/* Highlight ring around target */}
      {targetRect && (
        <div
          className="fixed z-[9999] pointer-events-none rounded-lg"
          style={{
            top: targetRect.top - 4,
            left: targetRect.left - 4,
            width: targetRect.width + 8,
            height: targetRect.height + 8,
            boxShadow: '0 0 0 4px #06b6d4, 0 0 0 9999px rgba(0,0,0,0.55)',
            borderRadius: 8,
            transition: 'all 200ms ease-out',
          }}
        />
      )}

      {/* Tooltip card */}
      <div
        className="z-[10000] bg-bg-card border border-accent-cyan/30 rounded-xl shadow-2xl"
        style={cardStyle}
      >
        <div className="p-4 space-y-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-[10px] text-accent-cyan uppercase tracking-wide font-semibold">
                Étape {stepIdx + 1} / {STEPS.length}
              </p>
              <h3 className="text-base font-semibold text-text-primary mt-0.5">{step.title}</h3>
            </div>
            <button
              onClick={finish}
              aria-label="Fermer le tour"
              className="p-1 hover:bg-bg-hover rounded text-text-secondary -mr-1 -mt-1"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="text-sm text-text-secondary leading-relaxed">
            {step.content}
          </div>

          {/* Progress dots */}
          <div className="flex items-center justify-center gap-1.5 pt-1">
            {STEPS.map((_, i) => (
              <span
                key={i}
                className={`w-1.5 h-1.5 rounded-full transition-colors ${
                  i === stepIdx ? 'bg-accent-cyan' : i < stepIdx ? 'bg-accent-cyan/40' : 'bg-white/15'
                }`}
              />
            ))}
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-white/5">
            <button
              onClick={prev}
              disabled={stepIdx === 0}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded text-xs text-text-secondary hover:bg-bg-hover disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
              Précédent
            </button>
            {stepIdx === 0 && (
              <button
                onClick={finish}
                className="text-xs text-text-secondary hover:text-text-primary transition-colors"
              >
                Passer
              </button>
            )}
            <button
              onClick={next}
              className="flex items-center gap-1 px-3 py-1.5 rounded bg-accent-cyan text-bg-primary text-xs font-medium hover:bg-accent-cyan/80 transition-colors"
            >
              {stepIdx === STEPS.length - 1 ? 'Terminer' : 'Suivant'}
              {stepIdx < STEPS.length - 1 && <ChevronRight className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </div>
    </>,
    document.body,
  )
}
