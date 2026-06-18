import { describe, expect, it } from 'vitest'
import { observatoryApi } from './observatory-api'

describe('observatoryApi exportUrl', () => {
  it('builds the piezo export url with encoded code', () => {
    expect(observatoryApi.piezo.exportUrl('BSS000/X')).toBe(
      '/api/v1/observatory/piezo/stations/BSS000%2FX/export.csv',
    )
  })
  it('builds the hydro export url with encoded code', () => {
    expect(observatoryApi.hydro.exportUrl('K001 0010')).toBe(
      '/api/v1/observatory/hydro/stations/K001%200010/export.csv',
    )
  })
  it('appends start_date and end_date when provided', () => {
    expect(
      observatoryApi.piezo.exportUrl('BSS000/X', { start_date: '2020-01-01', end_date: '2020-12-31' }),
    ).toBe(
      '/api/v1/observatory/piezo/stations/BSS000%2FX/export.csv?start_date=2020-01-01&end_date=2020-12-31',
    )
  })
  it('omits empty date params', () => {
    expect(observatoryApi.hydro.exportUrl('K001', { start_date: '2020-01-01', end_date: '' })).toBe(
      '/api/v1/observatory/hydro/stations/K001/export.csv?start_date=2020-01-01',
    )
  })
})
