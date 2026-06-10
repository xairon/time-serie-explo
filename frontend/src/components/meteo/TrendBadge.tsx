// frontend/src/components/meteo/TrendBadge.tsx
// DOM twin of the map's trend badge: white 60% circle + black glyph.
export type TrendBadgeKind = 'hausse' | 'stable' | 'baisse' | 'inconnu'

const GLYPHS: Record<TrendBadgeKind, React.ReactNode> = {
  hausse: <path d="M9 12.5 V5.5 M6.2 8.2 L9 5.5 L11.8 8.2" stroke="#000" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />,
  baisse: <path d="M9 5.5 V12.5 M6.2 9.8 L9 12.5 L11.8 9.8" stroke="#000" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />,
  stable: <path d="M5.8 7.5 H12.2 M5.8 10.5 H12.2" stroke="#000" strokeWidth="1.5" fill="none" strokeLinecap="round" />,
  inconnu: <text x="9" y="12.6" textAnchor="middle" fontSize="10" fontWeight="bold" fill="#000">?</text>,
}

export function TrendBadge({ kind, size = 18 }: { kind: TrendBadgeKind; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" aria-hidden="true" style={{ flexShrink: 0 }}>
      <circle cx="9" cy="9" r="8.4" fill="rgba(255,255,255,0.6)" stroke="rgba(0,0,0,0.25)" strokeWidth="0.6" />
      {GLYPHS[kind]}
    </svg>
  )
}
