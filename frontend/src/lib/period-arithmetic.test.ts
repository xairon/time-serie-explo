// frontend/src/lib/period-arithmetic.test.ts
import { describe, it, expect } from 'vitest'
import { addMonths, comparePeriods } from './period-arithmetic'

describe('addMonths', () => {
  it('adds within a year', () => expect(addMonths('2026-03', 2)).toBe('2026-05'))
  it('wraps forward across years', () => expect(addMonths('2025-11', 3)).toBe('2026-02'))
  it('wraps backward across years', () => expect(addMonths('2026-01', -1)).toBe('2025-12'))
  it('handles large negative deltas', () => expect(addMonths('2026-06', -18)).toBe('2024-12'))
})

describe('comparePeriods', () => {
  it('orders chronologically within a year', () => expect(comparePeriods('2026-03', '2026-05')).toBe(-1))
  it('orders chronologically across years', () => expect(comparePeriods('2025-12', '2026-01')).toBe(-1))
  it('is symmetric', () => expect(comparePeriods('2026-05', '2026-03')).toBe(1))
  it('is zero for equal periods', () => expect(comparePeriods('2026-06', '2026-06')).toBe(0))
})
