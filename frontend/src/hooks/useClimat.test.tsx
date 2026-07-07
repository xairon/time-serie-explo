import { describe, it, expect, vi, beforeEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

vi.mock('@/lib/observatory-api', () => ({
  observatoryApi: {
    climat: {
      pointSeries: vi.fn().mockResolvedValue({ cell: { latitude: 47.4, longitude: 0.7 }, series: [] }),
      pointEpisodes: vi.fn().mockResolvedValue([]),
      compareYears: vi.fn().mockResolvedValue({ cell: { latitude: 47.4, longitude: 0.7 }, years: {} }),
      gridIndices: vi.fn().mockResolvedValue([]),
    },
  },
}))

import { observatoryApi } from '@/lib/observatory-api'
import {
  useSelectedCellParam,
  useClimatPointSeries,
  useClimatPointEpisodes,
  useClimatCompareYears,
  useClimatCompareGridIndices,
} from './useClimat'

const pointSeriesMock = observatoryApi.climat.pointSeries as unknown as ReturnType<typeof vi.fn>
const pointEpisodesMock = observatoryApi.climat.pointEpisodes as unknown as ReturnType<typeof vi.fn>
const compareYearsMock = observatoryApi.climat.compareYears as unknown as ReturnType<typeof vi.fn>
const gridIndicesMock = observatoryApi.climat.gridIndices as unknown as ReturnType<typeof vi.fn>

function routerWrapper(initialEntry: string) {
  return ({ children }: { children: ReactNode }) => (
    <MemoryRouter initialEntries={[initialEntry]}>{children}</MemoryRouter>
  )
}

function queryWrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
}

describe('useSelectedCellParam', () => {
  it('parses lat/lon from the URL on mount', () => {
    const { result } = renderHook(() => useSelectedCellParam(), { wrapper: routerWrapper('/climat?lat=47.40&lon=0.70') })
    expect(result.current.selectedCell).toEqual({ lat: 47.4, lon: 0.7 })
  })

  it('returns null when lat/lon are absent from the URL', () => {
    const { result } = renderHook(() => useSelectedCellParam(), { wrapper: routerWrapper('/climat') })
    expect(result.current.selectedCell).toBeNull()
  })

  it('returns null when lat/lon are not numeric', () => {
    const { result } = renderHook(() => useSelectedCellParam(), { wrapper: routerWrapper('/climat?lat=abc&lon=0.7') })
    expect(result.current.selectedCell).toBeNull()
  })

  it('selectCell writes rounded coordinates that round-trip back through the URL', () => {
    const { result } = renderHook(() => useSelectedCellParam(), { wrapper: routerWrapper('/climat') })
    act(() => { result.current.selectCell(48.1234, 1.789) })
    expect(result.current.selectedCell).toEqual({ lat: 48.12, lon: 1.79 })
  })

  it('clearSelectedCell removes lat/lon from the URL', () => {
    const { result } = renderHook(() => useSelectedCellParam(), { wrapper: routerWrapper('/climat?lat=47.40&lon=0.70') })
    expect(result.current.selectedCell).not.toBeNull()
    act(() => { result.current.clearSelectedCell() })
    expect(result.current.selectedCell).toBeNull()
  })
})

describe('useClimatPointSeries / useClimatPointEpisodes gating', () => {
  beforeEach(() => { pointSeriesMock.mockClear(); pointEpisodesMock.mockClear() })

  it('does not call the API when lat/lon are undefined', () => {
    renderHook(() => useClimatPointSeries(undefined, undefined), { wrapper: queryWrapper() })
    renderHook(() => useClimatPointEpisodes(undefined, undefined), { wrapper: queryWrapper() })
    expect(pointSeriesMock).not.toHaveBeenCalled()
    expect(pointEpisodesMock).not.toHaveBeenCalled()
  })

  it('calls the API once lat/lon are defined', async () => {
    const { result } = renderHook(() => useClimatPointSeries(47.4, 0.7), { wrapper: queryWrapper() })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(pointSeriesMock).toHaveBeenCalledWith(47.4, 0.7)
  })

  it('calls the episodes API with the fixed 3-month window once lat/lon are defined', async () => {
    const { result } = renderHook(() => useClimatPointEpisodes(47.4, 0.7), { wrapper: queryWrapper() })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(pointEpisodesMock).toHaveBeenCalledWith(47.4, 0.7, 3)
  })
})

describe('useClimatCompareYears gating (Task B3)', () => {
  beforeEach(() => { compareYearsMock.mockClear() })

  it('does not call the API with only 1 selected year (below the 2-6 bound)', () => {
    renderHook(() => useClimatCompareYears(47.4, 0.7, [2003]), { wrapper: queryWrapper() })
    expect(compareYearsMock).not.toHaveBeenCalled()
  })

  it('does not call the API when lat/lon are undefined even with a valid year selection', () => {
    renderHook(() => useClimatCompareYears(undefined, undefined, [1976, 2003]), { wrapper: queryWrapper() })
    expect(compareYearsMock).not.toHaveBeenCalled()
  })

  it('calls the API once lat/lon are defined and years is within bounds', async () => {
    const { result } = renderHook(() => useClimatCompareYears(47.4, 0.7, [1976, 2003, 2022]), { wrapper: queryWrapper() })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(compareYearsMock).toHaveBeenCalledWith(47.4, 0.7, [1976, 2003, 2022])
  })
})

describe('useClimatCompareGridIndices gating (Task B3 mini-maps)', () => {
  beforeEach(() => { gridIndicesMock.mockClear() })

  it('fetches exactly one grid-indices call per selected year, for the given month', async () => {
    const { result } = renderHook(() => useClimatCompareGridIndices([1976, 2003], 6, true), { wrapper: queryWrapper() })
    await waitFor(() => expect(result.current.every((q) => q.isSuccess)).toBe(true))
    expect(gridIndicesMock).toHaveBeenCalledTimes(2)
    expect(gridIndicesMock).toHaveBeenCalledWith('1976-06', 3, 'spi')
    expect(gridIndicesMock).toHaveBeenCalledWith('2003-06', 3, 'spi')
  })

  it('does not fetch anything when disabled', () => {
    renderHook(() => useClimatCompareGridIndices([1976, 2003], 6, false), { wrapper: queryWrapper() })
    expect(gridIndicesMock).not.toHaveBeenCalled()
  })

  it('does not fetch anything for an empty year selection', () => {
    renderHook(() => useClimatCompareGridIndices([], 6, true), { wrapper: queryWrapper() })
    expect(gridIndicesMock).not.toHaveBeenCalled()
  })
})
