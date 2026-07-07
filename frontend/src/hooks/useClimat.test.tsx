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
    },
  },
}))

import { observatoryApi } from '@/lib/observatory-api'
import { useSelectedCellParam, useClimatPointSeries, useClimatPointEpisodes } from './useClimat'

const pointSeriesMock = observatoryApi.climat.pointSeries as unknown as ReturnType<typeof vi.fn>
const pointEpisodesMock = observatoryApi.climat.pointEpisodes as unknown as ReturnType<typeof vi.fn>

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
