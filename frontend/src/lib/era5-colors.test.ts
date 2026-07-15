import { describe, it, expect } from 'vitest'
import { SPI_CLASS_COLORS, STI_CLASS_COLORS } from './era5-colors'

describe('palettes d\'anomalie cohérentes', () => {
  it('Normal est le même gris neutre pour SPI et STI', () => {
    expect(SPI_CLASS_COLORS.NORMAL).toBe('#f7f7f7')
    expect(STI_CLASS_COLORS.NORMAL).toBe(SPI_CLASS_COLORS.NORMAL)
  })
  it('SPI : sec = rouge, humide = bleu', () => {
    expect(SPI_CLASS_COLORS.EXTREMEMENT_BAS).toBe('#b2182b') // très sec = rouge
    expect(SPI_CLASS_COLORS.EXTREMEMENT_HAUT).toBe('#2166ac') // très humide = bleu
  })
  it('STI : chaud = rouge, froid = bleu (axe inversé, même rouge = préoccupant)', () => {
    expect(STI_CLASS_COLORS.EXTREMEMENT_HAUT).toBe('#b2182b') // très chaud = rouge
    expect(STI_CLASS_COLORS.EXTREMEMENT_BAS).toBe('#2166ac')  // très froid = bleu
  })
})
