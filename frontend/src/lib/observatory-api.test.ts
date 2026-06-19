import { describe, expect, it } from 'vitest'
import { observatoryApi, EXPORT_COLUMN_GROUPS } from './observatory-api'

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
  it('appends groups when a strict subset is selected', () => {
    expect(
      observatoryApi.piezo.exportUrl('BSS000/X', undefined, ['values', 'index']),
    ).toBe(
      '/api/v1/observatory/piezo/stations/BSS000%2FX/export.csv?groups=values%2Cindex',
    )
  })
  it('omits groups when all five groups are selected', () => {
    expect(
      observatoryApi.hydro.exportUrl('K001', undefined, [...EXPORT_COLUMN_GROUPS]),
    ).toBe('/api/v1/observatory/hydro/stations/K001/export.csv')
  })
  it('combines date range and groups', () => {
    expect(
      observatoryApi.piezo.exportUrl('BSS000/X', { start_date: '2020-01-01', end_date: '' }, ['values']),
    ).toBe(
      '/api/v1/observatory/piezo/stations/BSS000%2FX/export.csv?start_date=2020-01-01&groups=values',
    )
  })
})
