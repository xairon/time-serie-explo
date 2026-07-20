// Key-parity guard for the Climat namespace only (not the whole locale files —
// pre-existing drift elsewhere is out of scope here). Every key added for the
// daily-temperature feature (Tx/Tn/Tmoy, DayStepper, daily-temp banner, info
// tooltip) must exist, with the same nesting, in both fr.json and en.json —
// an orphan key in either direction silently breaks the other language.
import { describe, it, expect } from 'vitest'
import fr from './locales/fr.json'
import en from './locales/en.json'

function collectKeys(obj: unknown, prefix = ''): string[] {
  if (obj === null || typeof obj !== 'object') return [prefix]
  return Object.entries(obj as Record<string, unknown>).flatMap(([k, v]) =>
    collectKeys(v, prefix ? `${prefix}.${k}` : k),
  )
}

describe('climat i18n key parity (fr vs en)', () => {
  it('has an identical key set under "climat" in both locales', () => {
    const frKeys = collectKeys(fr.climat).sort()
    const enKeys = collectKeys(en.climat).sort()
    expect(frKeys).toEqual(enKeys)
  })

  it('includes every new daily-temp key introduced for this feature', () => {
    const frKeys = new Set(collectKeys(fr.climat))
    const newKeys = [
      'variables.tmax', 'variables.tmin', 'variables.tmean',
      'picker.dailyTempLabel', 'picker.dailyTempInfo',
      'stepper.prevDay', 'stepper.nextDay',
      // Les clés banner.dailyTemp* ont été retirées avec le bandeau journalier
      // (synthèse narrée supprimée, 2026-07-17) — plus rien à garder ici pour lui.
    ]
    for (const key of newKeys) {
      expect(frKeys.has(key), `missing fr key: climat.${key}`).toBe(true)
    }
  })
})
