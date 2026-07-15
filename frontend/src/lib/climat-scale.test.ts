import { describe, it, expect } from 'vitest'
import { classifyBilan } from './climat-scale'
import { SPI_CLASS_COLORS } from './era5-colors'

describe('classifyBilan (binning mm, déficit = rouge)', () => {
  it('déficit sévère → EXTREMEMENT_BAS (rouge)', () => {
    expect(classifyBilan(-200)).toBe('EXTREMEMENT_BAS')
    expect(SPI_CLASS_COLORS[classifyBilan(-200)]).toBe('#b2182b')
  })
  it('équilibré → NORMAL (neutre)', () => {
    expect(classifyBilan(0)).toBe('NORMAL')
    expect(SPI_CLASS_COLORS[classifyBilan(0)]).toBe('#f7f7f7')
  })
  it('fort surplus → EXTREMEMENT_HAUT (bleu)', () => {
    expect(classifyBilan(200)).toBe('EXTREMEMENT_HAUT')
  })
})
