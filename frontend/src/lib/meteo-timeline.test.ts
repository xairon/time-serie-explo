// frontend/src/lib/meteo-timeline.test.ts
import { describe, it, expect } from 'vitest'
import { addMonths, comparePeriods, buildTimelineWindow, formatPeriodShortFR, formatPeriodLongFR } from './meteo-timeline'

function monthRange(start: string, end: string): string[] {
  const out: string[] = []
  let p = start
  while (comparePeriods(p, end) <= 0) { out.push(p); p = addMonths(p, 1) }
  return out
}

describe('addMonths', () => {
  it('adds within a year', () => expect(addMonths('2026-03', 2)).toBe('2026-05'))
  it('wraps forward across years', () => expect(addMonths('2025-11', 3)).toBe('2026-02'))
  it('wraps backward across years', () => expect(addMonths('2026-01', -1)).toBe('2025-12'))
  it('handles large negative deltas', () => expect(addMonths('2026-06', -18)).toBe('2024-12'))
})

describe('buildTimelineWindow', () => {
  const periods = monthRange('2010-01', '2026-06')

  it('recent selection: 12 data months ending at latest + 3 future slots', () => {
    const cells = buildTimelineWindow(periods, '2026-06')
    expect(cells).toHaveLength(15)
    expect(cells[0].period).toBe('2025-07')
    expect(cells[11].period).toBe('2026-06')
    expect(cells.slice(0, 12).every(c => c.available && !c.future)).toBe(true)
    expect(cells.slice(12).map(c => c.period)).toEqual(['2026-07', '2026-08', '2026-09'])
    expect(cells.slice(12).every(c => c.future && !c.available)).toBe(true)
  })

  it('old selection is centered in the window', () => {
    const cells = buildTimelineWindow(periods, '2015-06')
    const idx = cells.findIndex(c => c.period === '2015-06')
    expect(idx).toBe(5) // floor((12-1)/2)
    expect(cells.every(c => c.available)).toBe(true) // all in-range, no future
  })

  it('clamps at history start', () => {
    const cells = buildTimelineWindow(periods, '2010-02')
    expect(cells[0].period).toBe('2010-01')
  })

  it('marks January cells with showYear', () => {
    const cells = buildTimelineWindow(periods, '2026-06')
    const jan = cells.find(c => c.period === '2026-01')
    expect(jan?.showYear).toBe(true)
    expect(cells.find(c => c.period === '2026-02')?.showYear).toBe(false)
  })

  it('short history starts at first period', () => {
    const cells = buildTimelineWindow(monthRange('2026-01', '2026-06'), '2026-06')
    expect(cells[0].period).toBe('2026-01')
    expect(cells.filter(c => c.available)).toHaveLength(6)
  })

  it('empty history yields no cells', () => {
    expect(buildTimelineWindow([], '2026-06')).toEqual([])
  })
})

describe('formatting', () => {
  it('short FR', () => expect(formatPeriodShortFR('2026-02')).toBe('févr.'))
  it('long FR', () => expect(formatPeriodLongFR('2026-06')).toBe('juin 2026'))
})
