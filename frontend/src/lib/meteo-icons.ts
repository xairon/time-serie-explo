// frontend/src/lib/meteo-icons.ts
// Canvas-drawn MapLibre icons for the /meteo clone.
// Trend badges replicate the original's .sector-icon: 18px circle,
// rgba(255,255,255,0.6) background, black glyph (~8px) — drawn at 3x for crispness.

export type TrendKind = 'hausse' | 'stable' | 'baisse' | 'inconnu'

type DrawFn = (ctx: CanvasRenderingContext2D, size: number) => void

function render(draw: DrawFn, size: number): ImageData {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  draw(ctx, size)
  return ctx.getImageData(0, 0, size, size)
}

/** Render a draw function to ImageData for map.addImage().
 *  Pass { sdf: true } to addImage when the icon must be recolored via icon-color. */
export function createIcon(draw: DrawFn, size = 44): ImageData {
  return render(draw, size)
}

/** Station badge: plain circle (tinted by classification color via icon-color).
 *  Intentionally a circle — the original MétéEAU app uses dot markers (clone spec). */
export function drawStationBadge(ctx: CanvasRenderingContext2D, size: number) {
  ctx.beginPath()
  ctx.arc(size / 2, size / 2, size * 0.34, 0, Math.PI * 2)
  ctx.fillStyle = '#fff'
  ctx.fill()
}

/** White type glyph — piezo: borehole stem + downward triangle. */
export function drawPiezoGlyph(ctx: CanvasRenderingContext2D, size: number) {
  const cx = size / 2
  ctx.fillStyle = '#fff'
  ctx.fillRect(cx - size * 0.045, size * 0.30, size * 0.09, size * 0.13)
  const w = size * 0.16, top = size * 0.44, bot = size * 0.66
  ctx.beginPath()
  ctx.moveTo(cx - w, top)
  ctx.lineTo(cx + w, top)
  ctx.lineTo(cx, bot)
  ctx.closePath()
  ctx.fill()
}

/** White type glyph — hydro: water drop. */
export function drawHydroGlyph(ctx: CanvasRenderingContext2D, size: number) {
  const cx = size / 2
  const r = size * 0.15
  ctx.beginPath()
  ctx.arc(cx, size * 0.58, r, 0.12 * Math.PI, 0.88 * Math.PI)
  ctx.lineTo(cx, size * 0.34)
  ctx.closePath()
  ctx.fillStyle = '#fff'
  ctx.fill()
}

/**
 * Trend badge factory: white 60% circle + black glyph.
 * hausse = arrow up, baisse = arrow down, stable = equals, inconnu = '?'.
 * Draw at size 54 → rendered at icon-size 1/3 = 18px.
 */
export function drawTrendBadge(kind: TrendKind): DrawFn {
  return (ctx, size) => {
    const c = size / 2
    ctx.beginPath()
    ctx.arc(c, c, size * 0.46, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255,255,255,0.6)'
    ctx.fill()

    ctx.strokeStyle = '#000'
    ctx.fillStyle = '#000'
    ctx.lineWidth = size * 0.07
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    if (kind === 'stable') {
      // equals sign
      const w = size * 0.18
      for (const dy of [-size * 0.08, size * 0.08]) {
        ctx.beginPath()
        ctx.moveTo(c - w, c + dy)
        ctx.lineTo(c + w, c + dy)
        ctx.stroke()
      }
      return
    }
    if (kind === 'inconnu') {
      ctx.font = `bold ${size * 0.46}px sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('?', c, c + size * 0.02)
      return
    }
    // arrow up / down: vertical shaft + chevron head
    const up = kind === 'hausse'
    const tipY = up ? c - size * 0.20 : c + size * 0.20
    const tailY = up ? c + size * 0.20 : c - size * 0.20
    const head = size * 0.12
    ctx.beginPath()
    ctx.moveTo(c, tailY)
    ctx.lineTo(c, tipY)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(c - head, up ? tipY + head : tipY - head)
    ctx.lineTo(c, tipY)
    ctx.lineTo(c + head, up ? tipY + head : tipY - head)
    ctx.stroke()
  }
}
