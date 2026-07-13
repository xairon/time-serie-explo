import { describe, it, expect, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MiniSpiMap } from './MiniSpiMap'

// jsdom's canvas 2D context is unavailable (no `canvas` npm package installed),
// so the draw calls in the effect are a no-op here — see the component's own
// comment. What IS testable without a real 2D backend, and what this guards,
// is the HiDPI sizing contract: the backing store (width/height attributes)
// must scale with devicePixelRatio while the on-screen CSS size stays fixed,
// otherwise the small SPI multiples render soft on retina screens.
function setDpr(value: number) {
  Object.defineProperty(window, 'devicePixelRatio', { value, configurable: true })
}

afterEach(() => setDpr(1))

describe('MiniSpiMap', () => {
  it('sizes the canvas backing store 1:1 with CSS size at devicePixelRatio 1', () => {
    setDpr(1)
    render(<MiniSpiMap points={[]} label="2022" size={180} />)
    const canvas = screen.getByTestId('mini-spi-map') as HTMLCanvasElement
    expect(canvas.width).toBe(180)
    expect(canvas.height).toBe(180)
    expect(canvas.style.width).toBe('180px')
    expect(canvas.style.height).toBe('180px')
  })

  it('scales the backing store by devicePixelRatio while keeping the CSS size fixed', () => {
    setDpr(2)
    render(<MiniSpiMap points={[]} label="2022" size={180} />)
    const canvas = screen.getByTestId('mini-spi-map') as HTMLCanvasElement
    expect(canvas.width).toBe(360)
    expect(canvas.height).toBe(360)
    expect(canvas.style.width).toBe('180px')
    expect(canvas.style.height).toBe('180px')
  })

  it('respects a custom size prop under a non-integer devicePixelRatio', () => {
    setDpr(1.5)
    render(<MiniSpiMap points={[]} label="2003" size={100} />)
    const canvas = screen.getByTestId('mini-spi-map') as HTMLCanvasElement
    expect(canvas.width).toBe(150)
    expect(canvas.height).toBe(150)
    expect(canvas.style.width).toBe('100px')
  })
})
