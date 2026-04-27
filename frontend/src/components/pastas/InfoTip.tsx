import { Info } from 'lucide-react'

interface Props {
  text: string
  className?: string
  iconSize?: number
}

export function InfoTip({ text, className = '', iconSize = 12 }: Props) {
  return (
    <span className={`relative inline-flex items-center group ${className}`}>
      <Info
        className="text-text-muted/50 group-hover:text-text-muted cursor-help transition-colors"
        style={{ width: iconSize, height: iconSize }}
      />
      <span className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-56 px-2.5 py-1.5 rounded-lg bg-gray-800 border border-white/10 text-[10px] leading-relaxed text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity z-50 shadow-lg">
        {text}
      </span>
    </span>
  )
}
