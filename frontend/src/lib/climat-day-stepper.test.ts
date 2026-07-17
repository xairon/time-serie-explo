import { describe, it, expect } from 'vitest'
import { addDays, comparePeriods, resolveDefaultDay, formatDayLabel } from './climat-day-stepper'

describe('addDays', () => {
  // Régression : `day` vaut '' tant que /daily-temp-range n'a pas répondu — un état
  // que ce module CONNAÎT (resolveDefaultDay retourne '' par conception) et contre
  // lequel formatDayLabel se garde déjà. addDays était la seule fonction du fichier
  // à ne pas le faire : ''.split('-').map(Number) donnait [0], d'où
  // new Date(Date.UTC(0, NaN, undefined)) = Invalid Date, et .toISOString() levait
  // une RangeError qui faisait tomber TOUTE la page /climat (écran « Unexpected
  // Application Error! invalid date ») dès qu'on sélectionnait Tx/Tn/T moy avant que
  // la plage ne soit revenue.
  it('ne lève pas et rend l’entrée telle quelle quand le jour n’est pas renseigné', () => {
    expect(() => addDays('', 1)).not.toThrow()
    expect(addDays('', 1)).toBe('')
    expect(addDays('', -1)).toBe('')
  })

  it('ne lève pas sur une date malformée', () => {
    expect(() => addDays('pas-une-date', 1)).not.toThrow()
    expect(addDays('2026-06', 1)).toBe('2026-06')
  })

  it('steps forward one day within a month', () => {
    expect(addDays('2026-06-15', 1)).toBe('2026-06-16')
  })

  it('steps backward one day within a month', () => {
    expect(addDays('2026-06-15', -1)).toBe('2026-06-14')
  })

  it('rolls over to the next month', () => {
    expect(addDays('2026-06-30', 1)).toBe('2026-07-01')
  })

  it('rolls back to the previous month', () => {
    expect(addDays('2026-07-01', -1)).toBe('2026-06-30')
  })

  it('rolls over a year boundary', () => {
    expect(addDays('2026-12-31', 1)).toBe('2027-01-01')
  })

  it('handles a leap-year February correctly', () => {
    expect(addDays('2028-02-28', 1)).toBe('2028-02-29') // 2028 is a leap year
    expect(addDays('2027-02-28', 1)).toBe('2027-03-01') // 2027 is not
  })
})

describe('comparePeriods on day strings', () => {
  it('compares ISO date strings lexicographically (same as month strings)', () => {
    expect(comparePeriods('2026-06-27', '2026-06-28')).toBe(-1)
    expect(comparePeriods('2026-06-28', '2026-06-28')).toBe(0)
    expect(comparePeriods('2026-06-29', '2026-06-28')).toBe(1)
  })
})

describe('resolveDefaultDay', () => {
  it('truncates a full ISO date to YYYY-MM-DD', () => {
    expect(resolveDefaultDay('2026-06-28T00:00:00')).toBe('2026-06-28')
  })

  it('passes through an already-short date unchanged', () => {
    expect(resolveDefaultDay('2026-06-28')).toBe('2026-06-28')
  })

  it('returns an empty string when there is no max_date yet (no ingestion row)', () => {
    expect(resolveDefaultDay(null)).toBe('')
    expect(resolveDefaultDay(undefined)).toBe('')
  })
})

describe('formatDayLabel', () => {
  it('formats a French long date', () => {
    expect(formatDayLabel('2026-06-28', 'fr')).toBe('28 juin 2026')
  })

  it('formats an English long date', () => {
    expect(formatDayLabel('2026-06-28', 'en')).toBe('June 28, 2026')
  })

  it('falls back to the raw string when the input does not match YYYY-MM-DD', () => {
    expect(formatDayLabel('not-a-date', 'fr')).toBe('not-a-date')
  })
})
