import { useEffect, useRef } from 'react'
import type { ClimatIndexPoint } from '@/lib/observatory-types'
import { SPI_CLASS_COLORS } from '@/lib/era5-colors'
import { projectCellToPixelRect } from '@/lib/climat-minimap'

interface Props {
  points: ClimatIndexPoint[] | undefined
  label: string
  size?: number
}

const DEFAULT_SIZE = 180

/** Non-interactive SPI mini-map for one year of the Comparaison "petits multiples"
 *  (Task B3) — a flat canvas render of the grid squares (see climat-minimap.ts for
 *  why: a full MapLibre/WebGL instance per year would be heavy for a ~180px
 *  thumbnail with no pan/zoom). Colours reuse the server-computed `index_class`
 *  (McKee 7-class) directly — no client-side re-classification. */
export function MiniSpiMap({ points, label, size = DEFAULT_SIZE }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return // jsdom in tests has no 2D canvas backend — render is a no-op there.
    ctx.clearRect(0, 0, size, size)
    ctx.fillStyle = 'rgba(107, 114, 128, 0.08)'
    ctx.fillRect(0, 0, size, size)
    for (const p of points ?? []) {
      if (p.spi == null) continue
      const rect = projectCellToPixelRect(p.latitude, p.longitude, size, size)
      ctx.fillStyle = SPI_CLASS_COLORS[p.index_class] ?? SPI_CLASS_COLORS.UNKNOWN
      ctx.fillRect(rect.x, rect.y, Math.max(rect.w, 1), Math.max(rect.h, 1))
    }
  }, [points, size])

  return (
    <div className="flex flex-col items-center gap-1">
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        data-testid="mini-spi-map"
        aria-label={label}
        role="img"
        className="rounded border border-white/10 bg-black/20"
      />
      <span className="text-[11px] text-text-secondary">{label}</span>
    </div>
  )
}
