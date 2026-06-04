import { Info } from 'lucide-react'
import { useState, useRef } from 'react'
import { createPortal } from 'react-dom'

interface Props {
  text: string
  className?: string
  iconSize?: number
}

// The tooltip is rendered in a portal on document.body with fixed positioning so
// it can never be clipped by an ancestor's `overflow` or trapped behind a sibling's
// stacking context (which a plain absolutely-positioned `z-50` tooltip suffers from).
export function InfoTip({ text, className = '', iconSize = 12 }: Props) {
  const ref = useRef<HTMLSpanElement>(null)
  const [pos, setPos] = useState<{ top: number; left: number; below: boolean } | null>(null)

  function show() {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const below = r.top < 96 // not enough room above → flip below the icon
    setPos({
      top: below ? r.bottom + 6 : r.top - 6,
      left: r.left + r.width / 2,
      below,
    })
  }

  return (
    <span
      ref={ref}
      className={`relative inline-flex items-center ${className}`}
      onMouseEnter={show}
      onMouseLeave={() => setPos(null)}
    >
      <Info
        className="text-text-muted/50 hover:text-text-muted cursor-help transition-colors"
        style={{ width: iconSize, height: iconSize }}
      />
      {pos && createPortal(
        <span
          style={{
            position: 'fixed',
            top: pos.top,
            left: pos.left,
            transform: `translate(-50%, ${pos.below ? '0' : '-100%'})`,
          }}
          className="pointer-events-none z-[9999] w-56 px-2.5 py-1.5 rounded-lg bg-gray-800 border border-white/10 text-[10px] leading-relaxed text-gray-300 shadow-lg"
        >
          {text}
        </span>,
        document.body,
      )}
    </span>
  )
}
