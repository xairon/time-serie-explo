import { describe, it, expect } from 'vitest'
import { computeRollingCumuls, latestSpiPoint, spiClassBadge } from './climate-cumuls'
import { SPI_CLASS_COLORS } from './era5-colors'
import type { ClimatPointSeriesEntry, SPIDataPoint } from './observatory-types'

/** Minimal point-series entry — only the fields the cumuls helper reads. */
function entry(
  month: string,
  precip: number | null,
  normale: number | null,
  complet: boolean | null = true,
): ClimatPointSeriesEntry {
  return {
    month,
    temperature_moyenne: null, temperature_min: null, temperature_max: null,
    precipitation_totale: precip,
    etp_totale: null, bilan_hydrique: null, nb_jours: null,
    mois_complet: complet,
    precipitation_normale: normale,
    temperature_normale: null,
    spi_1: null, sti_1: null, spei_1: null, spi_3: null, sti_3: null, spei_3: null,
    spi_6: null, sti_6: null, spei_6: null, spi_12: null, sti_12: null, spei_12: null,
  }
}

/** Jan→Jun 2026, 10/20/30/40/50/60 mm observed vs a flat 40 mm normal. */
const SIX_MONTHS: ClimatPointSeriesEntry[] = [
  entry('2026-01-01', 10, 40),
  entry('2026-02-01', 20, 40),
  entry('2026-03-01', 30, 40),
  entry('2026-04-01', 40, 40),
  entry('2026-05-01', 50, 40),
  entry('2026-06-01', 60, 40),
]

describe('computeRollingCumuls', () => {
  it('sums observed and normal precipitation over the window, anchored on the last month', () => {
    const [c3] = computeRollingCumuls(SIX_MONTHS, [3])
    expect(c3).not.toBeNull()
    expect(c3!.window).toBe(3)
    expect(c3!.cumul).toBe(40 + 50 + 60)
    expect(c3!.normale).toBe(120)
    expect(c3!.ecartMm).toBe(30)
    expect(c3!.ecartPct).toBeCloseTo(25)
    expect(c3!.from).toBe('2026-04-01')
    expect(c3!.to).toBe('2026-06-01')
  })

  it('returns one result per requested window, in order', () => {
    const out = computeRollingCumuls(SIX_MONTHS, [3, 6, 12])
    expect(out).toHaveLength(3)
    expect(out[0]!.cumul).toBe(150)
    expect(out[1]!.cumul).toBe(210)
    expect(out[2]).toBeNull() // only 6 months available
  })

  it('skips a trailing incomplete month when anchoring', () => {
    const series = [...SIX_MONTHS, entry('2026-07-01', 5, 40, false)]
    const [c3] = computeRollingCumuls(series, [3])
    expect(c3!.to).toBe('2026-06-01')
    expect(c3!.cumul).toBe(150)
  })

  it('skips trailing months with a null observed value (A2: series may include null months)', () => {
    const series = [...SIX_MONTHS, entry('2026-07-01', null, 40)]
    const [c3] = computeRollingCumuls(series, [3])
    expect(c3!.to).toBe('2026-06-01')
  })

  it('returns null when a calendar month is missing inside the window', () => {
    const gappy = [
      entry('2026-01-01', 10, 40),
      entry('2026-02-01', 20, 40),
      // March missing
      entry('2026-04-01', 40, 40),
      entry('2026-05-01', 50, 40),
      entry('2026-06-01', 60, 40),
    ]
    const [c3, c5] = computeRollingCumuls(gappy, [3, 5])
    expect(c3).not.toBeNull() // Apr-Jun is contiguous
    expect(c5).toBeNull() // crosses the gap
  })

  it('returns null when a normal is missing for a month of the window', () => {
    const series = [
      entry('2026-04-01', 40, null),
      entry('2026-05-01', 50, 40),
      entry('2026-06-01', 60, 40),
    ]
    const [c3] = computeRollingCumuls(series, [3])
    expect(c3).toBeNull()
  })

  it('reports a null relative deviation when the normal sums to zero', () => {
    const series = [
      entry('2026-04-01', 1, 0),
      entry('2026-05-01', 2, 0),
      entry('2026-06-01', 3, 0),
    ]
    const [c3] = computeRollingCumuls(series, [3])
    expect(c3!.ecartPct).toBeNull()
    expect(c3!.ecartMm).toBe(6)
  })

  it('tolerates unsorted input', () => {
    const shuffled = [SIX_MONTHS[4], SIX_MONTHS[0], SIX_MONTHS[5], SIX_MONTHS[2], SIX_MONTHS[1], SIX_MONTHS[3]]
    const [c3] = computeRollingCumuls(shuffled, [3])
    expect(c3!.cumul).toBe(150)
  })

  it('returns all nulls for an empty series', () => {
    expect(computeRollingCumuls([], [3, 6, 12])).toEqual([null, null, null])
  })
})

describe('latestSpiPoint', () => {
  const pt = (mois: string, spi: number | null): SPIDataPoint => ({ mois, value: null, spi, classification: spi != null && spi < -1 ? 'TRES_BAS' : 'NORMAL' })

  it('returns the last point with a non-null SPI', () => {
    const data = [pt('2026-04-01', -0.5), pt('2026-05-01', -1.4), pt('2026-06-01', null)]
    expect(latestSpiPoint(data)?.mois).toBe('2026-05-01')
  })

  it('returns null when the series is empty or all-null', () => {
    expect(latestSpiPoint([])).toBeNull()
    expect(latestSpiPoint([pt('2026-06-01', null)])).toBeNull()
  })
})

describe('spiClassBadge', () => {
  it('maps every known class to its SPI colour and i18n key', () => {
    for (const cls of Object.keys(SPI_CLASS_COLORS)) {
      const badge = spiClassBadge(cls)
      expect(badge.color).toBe(SPI_CLASS_COLORS[cls])
      expect(badge.labelKey).toBe(`observatory.spi.${cls}`)
    }
  })

  it('falls back to UNKNOWN for unknown, null or undefined classes', () => {
    for (const cls of ['WAT', null, undefined]) {
      const badge = spiClassBadge(cls as string | null | undefined)
      expect(badge.color).toBe(SPI_CLASS_COLORS['UNKNOWN'])
      expect(badge.labelKey).toBe('observatory.spi.UNKNOWN')
    }
  })

  it('picks a readable text colour: dark on light classes, white on dark ones', () => {
    expect(spiClassBadge('NORMAL').textColor).toBe('#1f2937') // #f5f5f5 background
    expect(spiClassBadge('EXTREMEMENT_BAS').textColor).toBe('#ffffff') // #8c510a background
  })
})
