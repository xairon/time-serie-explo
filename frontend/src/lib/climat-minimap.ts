// Lightweight canvas projection for the Comparaison "petits multiples" SPI mini-maps
// (Task B3). Deliberately NOT MapLibre: N years selected (up to 6) × a full GL context
// each would be heavy (WebGL context churn, ~1-2s init apiece) for a non-interactive
// ~200px thumbnail — a flat equirectangular projection + canvas fillRect per grid
// square renders all of them in a handful of milliseconds and comfortably meets the
// "< 3s for 4 years" target from the plan. Pure/testable: no canvas here, just maths.
import { ERA5_CELL_HALF } from './era5-grid'

/** Mainland France bounding box (metropolitan grid coverage — Corsica included),
 *  a bit wider than the strict data extent so cells near the coast aren't clipped. */
export const FRANCE_BBOX = { lonMin: -5.2, lonMax: 9.7, latMin: 41.2, latMax: 51.3 }

export interface PixelRect {
  x: number
  y: number
  w: number
  h: number
}

/** Project a grid-cell centre (lat/lon) to a pixel rectangle covering the whole
 *  0.1°×0.1° cell within a `width`×`height` canvas, y flipped (screen space, origin
 *  top-left) since latitude increases northward while canvas rows increase downward. */
export function projectCellToPixelRect(
  lat: number,
  lon: number,
  width: number,
  height: number,
  bbox: { lonMin: number; lonMax: number; latMin: number; latMax: number } = FRANCE_BBOX,
): PixelRect {
  const lonSpan = bbox.lonMax - bbox.lonMin
  const latSpan = bbox.latMax - bbox.latMin
  const cellW = (2 * ERA5_CELL_HALF * width) / lonSpan
  const cellH = (2 * ERA5_CELL_HALF * height) / latSpan
  const cx = ((lon - bbox.lonMin) / lonSpan) * width
  const cy = height - ((lat - bbox.latMin) / latSpan) * height
  return { x: cx - cellW / 2, y: cy - cellH / 2, w: cellW, h: cellH }
}
